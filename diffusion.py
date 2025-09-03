import math
import torch
import torch.nn.functional as F
import argparse

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=100000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyConstantLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument ("--k", type=int, default=1)
    return parser


def make_diffusion(model, n_timestep, time_scale, device):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
    return GaussianDiffusion(model, betas, time_scale=time_scale)


def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise Exception()
    return betas


def E_(input, t, shape):
    #ensuring t is not negative
    #print (f't = {t}')
    t=torch.clamp(t,min=0)
    
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])
        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)
    else:
        return noise_fn(*shape, device=device)


class GaussianDiffusion:

    def __init__(self, net, betas, time_scale=1, sampler="ddim"):
        sampler ='ddpm'
        super().__init__()
        self.net_ = net
        self.time_scale = time_scale
        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64, device=betas.device), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.posterior_variance = posterior_variance
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

        if sampler == "ddim":
            self.p_sample = self.p_sample_ddim
        elif sampler == "ddpm":
            self.p_sample = self.p_sample_ddpm
        else:
            self.p_sample = self.p_sample_loop
   #predicting noise 
    def inference(self, x, t, extra_args):
        return self.net_(x, t * self.time_scale, **extra_args)

    #loss 
    def p_loss(self, x_0, t, extra_args, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_t, sigma_t = self.get_alpha_sigma(x_0, t)
        z = alpha_t * x_0 + sigma_t * noise
        #predicted noise 
        eps_pred = self.inference(z.float(), t.float(), extra_args)
        return F.mse_loss(eps_pred,noise)

    def q_posterior(self, x_0, x_t, t):
        mean = E_(self.posterior_mean_coef1, t, x_t.shape) * x_0 \
               + E_(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = E_(self.posterior_variance, t, x_t.shape)
        log_var_clipped = E_(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_mean_variance(self, x, t, extra_args, clip_denoised):
        v = self.inference(x.float(), t.float(), extra_args).double()
        alpha_t, sigma_t = self.get_alpha_sigma(x, t)
        x_recon = alpha_t * x - sigma_t * v
        if clip_denoised:
            x_recon = x_recon.clamp(min=-1, max=1)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    #not usefull  
    
    
    def p_sample_clipped(self, x, t, extra_args, eta=0, clip_denoised=True, clip_value=3):
        v = self.inference(x.float(), t, extra_args)
        alpha, sigma = self.get_alpha_sigma(x, t)
        # if clip_denoised:
        #     x = x.clip(-1, 1)
        pred = (x * alpha - v * sigma)
        if clip_denoised:
            pred = pred.clip(-clip_value, clip_value)
        eps = (x - alpha * pred) / sigma
        if clip_denoised:
            eps = eps.clip(-clip_value, clip_value)

        t_mask = (t > 0)
        if t_mask.any().item():
            if not t_mask.all().item():
                raise Exception()
            alpha_, sigma_ = self.get_alpha_sigma(x, (t - 1).clip(min=0))
            ddim_sigma = eta * (sigma_ ** 2 / sigma ** 2).sqrt() * \
                         (1 - alpha ** 2 / alpha_ ** 2).sqrt()
            adjusted_sigma = (sigma_ ** 2 - ddim_sigma ** 2).sqrt()
            pred = pred * alpha_ + eps * adjusted_sigma
            if eta:
                pred += torch.randn_like(pred) * ddim_sigma
        return pred


#sampling using DDIM methodology 
    def p_sample_ddim(self, x, t, extra_args, eta=0, clip_denoised=True):
        print('t',t)
        eps = self.inference(x.float(), t.float(), extra_args).double()[:, -3:, :, :]
        alpha_t, sigma_t = self.get_alpha_sigma(x, t)

        # print (x.shape)
        # print (eps.shape) 
        
        x0_pred = (x - sigma_t * eps) / alpha_t
        
        if clip_denoised:
            x0_pred = x0_pred.clamp(min=-1, max=1)
        
        t_prev = t-1
        alpha_t_prev, sigma_t_prev = self.get_alpha_sigma(x, t_prev)
       # ddim_sigma = eta * (sigma_t_prev ** 2 / sigma_t ** 2).sqrt() * (1 - alpha_t ** 2 / alpha_t_prev ** 2).sqrt()
        ddim_sigma = eta * ((1 - alpha_t_prev**2) / (1 - alpha_t**2)).sqrt() * (1 - (alpha_t**2 / alpha_t_prev**2)).sqrt()
        mean_pred = alpha_t_prev * x0_pred + (1-alpha_t_prev**2 - ddim_sigma**2).sqrt() * eps

        # return mean_pred
        if eta == 0:
            return mean_pred
        else:
            return mean_pred + ddim_sigma * torch.randn_like(x)

    

    @torch.no_grad()
    def p_sample_loop(self, x, extra_args, eta=0):
        #print('calling my ahh')
        mode = self.net_.training
        self.net_.eval()
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((x.shape[0],), i, dtype=torch.long).to(x.device)
            x = self.p_sample_ddim(x, t, extra_args, eta=eta)
        self.net_.train(mode)
        return x

    def get_alpha_sigma(self, x, t):
        alpha = E_(self.sqrt_alphas_cumprod, t, x.shape)
        sigma = E_(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha, sigma

    def p_sample_ddpm(self, x, t, extra_args, clip_denoised=True, **kwargs):
            mean, _, log_var = self.p_mean_variance(x, t, extra_args, clip_denoised)
            noise = torch.randn_like(x)
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)
            return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise 






# CLASS FOR DISTILATION (will mainly have to change this code )
class GaussianDiffusionDefault(GaussianDiffusion):

    def __init__(self, args , net, betas, time_scale=1, gamma=0.):
        super().__init__(net, betas, time_scale )
        self.gamma = gamma
        self.args = args 

    def distill_loss_old(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
        with torch.no_grad():
            alpha, sigma = self.get_alpha_sigma(x, t + 1)
            z = alpha * x + sigma * eps #noisy image at t+1
            alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)
            alpha_1, sigma_1 = self.get_alpha_sigma(x, t)
            v = self.inference(z.float(), t.float() + 1, extra_args).double() #velocity predicted at t+1 to t
            rec = (alpha * z - sigma * v).clip(-1, 1) #RECREATED clean image at t 
            z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec) #noisy image at t 
            v_1 = self.inference(z_1.float(), t.float(), extra_args).double() #velocity predicted at t to t-1
            x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1) #recreated image at t-1
            eps_2 = (z - alpha_s * x_2) / sigma_s #nosie predicted at t-1
            v_2 = alpha_s * eps_2 - sigma_s * x_2 #velocity which the student has to predict which contains nosie predicted at t-1 and clean image at t-1 but parameters of t/2 essentially meaning the 
            if self.gamma == 0:
                w = 1
            else:
                w = torch.pow(1 + alpha_s / sigma_s, self.gamma)
        v = student_diffusion.net_(z.float(), t.float() // self.time_scale, **extra_args)
       
        my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
        return F.mse_loss(w * v.float(), w * v_2.float())
           
         
        
    def distill_loss(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
        if eps is None:
            eps = torch.randn_like(x)
            
        with torch.no_grad():
         z_list = []
         alpha_list = []
         sigma_list = []
         rec_list=[]
         v_list=[]
         
         for i in range (self.args.k):
            alpha, sigma = self.get_alpha_sigma(x, t-i)
            alpha_list.append(alpha)
            sigma_list.append(sigma)

            if i == 0:
                z = alpha * x + sigma * eps
                z_list.append(z)

                v = self.inference(z.float(), t.float() , extra_args).double() # function used to get teacher prediction of velocity 
               
                v_list.append(v)

                rec = (alpha * z - sigma * v).clip(-1, 1)
                rec_list.append(rec)

            else :
                z= alpha_list[i] * rec_list[i-1] + (sigma_list[i] / sigma_list[i-1]) * (z_list[i-1] - alpha_list[i-1]* rec_list[i-1])
                z_list.append(z)

                v = self.inference(z_list[i].float(), t.float() -i , extra_args).double()
                v_list.append(v)

                rec = (alpha * z_list[i] - sigma * v_list[i]).clip(-1, 1)
                rec_list.append(rec)
        
         alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // (self.args.k))

       
        v_final =  torch.cat(v_list, dim=1)


        if self.gamma == 0:
                w = 1
        else:
            w = torch.pow(1 + alpha_s / sigma_s, self.gamma)

        
        v = student_diffusion.net_(z.float(), t.float() // (self.args.k), **extra_args) #predcited v by student 
        B, C, H, W = v.shape
        eps_pred_t1, eps_pred_t= torch.split(v, C//(self.args.k), dim=1)
        eps_t1, eps_t= torch.split(v_final, C//(self.args.k), dim=1)
        l1 = F.mse_loss(w*eps_pred_t1.float(),w*eps_t1.float())
        l2 = F.mse_loss(w*eps_pred_t.float(),w*eps_t.float())

      #  print (f'v_final.shape = {v_final.shape}' )
       # print (f'v.shape = {v.shape}' )
        #print (f'size of the weight matrix of student model{student_diffusion.net_.out[2].weight.shape}')
        #return F.mse_loss(w * v.float(), w * v_final.float())  
        return l1 + l2
        # eps_tensor = torch.chunk(student_out, self.args.k , dim = 1)
        # #print (shape(eps_tensor))
        # Loss_list =[]

        # for i , eps in enumerate(eps_tensor):
        #     L = F.mse_loss(eps[i],v_list[i] )
        #     Loss_list.append(L)
        
        # return sum(Loss_list[:]).float()      


        
            
         
       
        #return F.mse_loss(w * v.float(), w * v_final.float())  

    # def distill_loss_(self, student_diffusion, x, t, extra_args, eps=None, student_device=None):
    #     if eps is None:
    #         eps = torch.randn_like(x)
    #     with torch.no_grad():
    #         # Step t+1
    #         alpha_t1, sigma_t1 = self.get_alpha_sigma(x, t + 1)
    #         z = alpha_t1 * x + sigma_t1 * eps
    #         eps_t1 = self.inference(z.float(), (t + 1).float(), extra_args)

    #         # Step t
    #         rec_t = (alpha_t1 * z - sigma_t1 * eps_t1).clip(-1, 1)
    #         alpha_t, sigma_t = self.get_alpha_sigma(x, t)
    #         z_t = alpha_t * rec_t + (sigma_t / sigma_t1) * (z - alpha_t1 * rec_t)
    #         eps_t = self.inference(z_t.float(), t.float(), extra_args)

    #         student_out = student_diffusion.net_(z.float(), t.float(), extra_args)
    #         B, C, H, W = student_out.shape

    #         # Split student output into 3 
    #         eps_pred_t1, eps_pred_t= torch.split(student_out, C//2, dim=1)

    #         l_1 = F.mse_loss(eps_pred_t1, eps_t1) 
    #         l_2 = F.mse_loss(eps_pred_t, eps_t)

    #         return (l_1.float() + l_2.float())
                
            


                


















