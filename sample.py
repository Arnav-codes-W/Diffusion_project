# import argparse
# import importlib
# from diffusion import make_beta_schedule
# from moving_average import init_ema_model
# from torch.utils.tensorboard import SummaryWriter
# from copy import deepcopy
# import torch
# import os
# from train_utils import make_visualization
# import cv2

# def make_argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--module", help="Model module.", type=str, required=True)
#     parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
#     parser.add_argument("--out_file", help="Path to image.", type=str, default="")
#     parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
#     parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
#     parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
#     parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
#     parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
#     parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
#     parser.add_argument("--k", type =int , default = 1)
#     return parser

# def sample_images(args, make_model):
#     device = torch.device("cuda")

#     teacher_ema = make_model(args ).to(device)

#     # teacher is made by loading weights from checkpoint
#     def make_diffusion(args, model, n_timestep, time_scale, device):
#         betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
#         M = importlib.import_module("diffusion")
#         D = getattr(M, args.diffusion)
#         sampler = "ddim"
#         return D(model, betas, time_scale=time_scale, sampler=sampler)

#     teacher = make_model(args ).to(device)

#     # ckpt = torch.load(args.checkpoint)
#     # teacher.load_state_dict(ckpt["G"])
#     # n_timesteps = ckpt["n_timesteps"]//args.time_scale
#     # time_scale = ckpt["time_scale"]*args.time_scale
#     # del ckpt

#     ckpt = torch.load(args.checkpoint)
#     old_state_dict = ckpt["G"]

#     # Target layer names (adjust if your model uses different naming)
#     out_weight_key = "out.2.weight"
#     out_bias_key = "out.2.bias"

#     # Slice last 3 output channels from previous student
#     old_out_weight = old_state_dict[out_weight_key]  # shape: [9, C, 3, 3]
#     old_out_bias = old_state_dict[out_bias_key]      # shape: [9]

#     if old_out_weight.shape[0]== (3*args.k):
#         # Only keep last 3
#         new_out_weight = old_out_weight[-3:]             # shape: [3, C, 3, 3]
#         new_out_bias = old_out_bias[-3:]                 # shape: [3]

#         # Load into teacher model
#         new_state_dict = teacher_ema.state_dict()
#         new_state_dict[out_weight_key] = new_out_weight
#         new_state_dict[out_bias_key] = new_out_bias

#         teacher_ema.load_state_dict(new_state_dict, strict=False)

#     else:
#         #ckpt = torch.load(args.checkpoint)
#         ckpt =  torch.load(args.checkpoint)
#         teacher_ema.load_state_dict(ckpt["G"])

#     teacher.eval()
#     n_timesteps = ckpt["n_timesteps"]//args.time_scale
#     time_scale = ckpt["time_scale"]*args.time_scale
#     del ckpt
#     print("Model loaded.")

    
    
#     teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
#     image_size = deepcopy(teacher.image_size)
#     image_size[0] = args.batch_size

#     with torch.no_grad():
#         img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta)
#     if img.shape[2] == 1:
#         img = img[:, :, 0]
#     cv2.imwrite(args.out_file, img)

#     print("Finished.")

# parser = make_argument_parser()

# args = parser.parse_args()

# M = importlib.import_module(args.module)
# make_model = getattr(M, "make_model")

# sample_images(args, make_model)



import argparse
import importlib
from diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import os
from train_utils import make_visualization
import cv2

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--out_file", help="Path to image.", type=str, default="")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
    return parser

def sample_images(args, make_model):
    device = torch.device("cuda")

    teacher = make_model().to(device)

    # teacher is made by loading weights from checkpoint
    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("diffusion")
        D = getattr(M, args.diffusion)
        sampler = "ddim"
        return D(model, betas, time_scale=time_scale, sampler=sampler)


    # ckpt = torch.load(args.checkpoint)
    # teacher.load_state_dict(ckpt["G"])
    # n_timesteps = ckpt["n_timesteps"]//args.time_scale
    # time_scale = ckpt["time_scale"]*args.time_scale
    # del ckpt

    ckpt = torch.load(args.checkpoint)
    old_state_dict = ckpt["G"]

    # Target layer names (adjust if your model uses different naming)
    out_weight_key = "conv_out.weight"
    out_bias_key = "conv_out.bias"

    # Slice last 3 output channels from previous student
    old_out_weight = old_state_dict[out_weight_key]  # shape: [9, C, 3, 3]
    old_out_bias = old_state_dict[out_bias_key]      # shape: [9]

    if old_out_weight.shape[0] > 3:
        # Only keep last 3
        new_out_weight = old_out_weight[-3:]             # shape: [3, C, 3, 3]
        new_out_bias = old_out_bias[-3:]                 # shape: [3]

        # Load into teacher model
        new_state_dict = teacher.state_dict()
        new_state_dict[out_weight_key] = new_out_weight
        new_state_dict[out_bias_key] = new_out_bias

        teacher.load_state_dict(new_state_dict, strict=False)

    else:
        ckpt = torch.load(args.checkpoint)
        teacher.load_state_dict(ckpt["G"])
    teacher.eval()
    n_timesteps = ckpt["n_timesteps"]//args.time_scale
    print(f'the timestepts are {n_timesteps} and {ckpt["time_scale"]}')

    time_scale = ckpt["time_scale"]*args.time_scale
    del ckpt
    print("Model loaded.")

    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
    image_size = deepcopy(teacher.image_size)
    image_size[0] = args.batch_size

    with torch.no_grad():
        img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)

    print("Finished.")

parser = make_argument_parser()

args = parser.parse_args()

M = importlib.import_module(args.module)
make_model = getattr(M, "make_model")

sample_images(args, make_model)





# import argparse
# import importlib
# from diffusion import make_beta_schedule
# from moving_average import init_ema_model
# from torch.utils.tensorboard import SummaryWriter
# from copy import deepcopy
# import torch
# import os
# from train_utils import make_visualization
# import cv2

# def make_argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--module", help="Model module.", type=str, required=True)
#     parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, default="")
#     parser.add_argument("--out_file", help="Path to image.", type=str, default="")
#     parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
#     parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
#     parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
#     parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
#     parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
#     parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
#     return parser

# def sample_images(args, make_model):
#     device = torch.device("cuda")

#     teacher_ema = make_model().to(device)

#     def make_diffusion(args, model, n_timestep, time_scale, device):
#         betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
#         M = importlib.import_module("diffusion")
#         D = getattr(M, args.diffusion)
#         sampler = "ddim"
#         if args.clipped_sampling:
#             sampler = "clipped"
#         return D(model, betas, time_scale=time_scale, sampler=sampler)

#     teacher = make_model().to(device)

#     ckpt = torch.load(args.checkpoint )
#     teacher.load_state_dict(ckpt["G"])
#     n_timesteps = ckpt["n_timesteps"]//args.time_scale
#     time_scale = ckpt["time_scale"]*args.time_scale
#     del ckpt
#     print("Model loaded.")

#     teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
#     print(f'the timesteps are {n_timesteps}')
#     image_size = deepcopy(teacher.image_size)
#     #print (image_size)
#     image_size[0] = args.batch_size

#     with torch.no_grad():
#         img = make_visualization(teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
#         print (f'image shape is {img.shape}')
#     # if img.shape[2] == 1:
#     #     img = img[:, :, -1]
#     cv2.imwrite(args.out_file, img)

#     print("Finished.")

# parser = make_argument_parser()

# args = parser.parse_args()

# M = importlib.import_module(args.module)
# make_model = getattr(M, "make_model")

# sample_images(args, make_model)