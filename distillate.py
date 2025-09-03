#!/usr/bin/env python
# coding: utf-8
import argparse
import importlib
from diffusion import make_beta_schedule
from train_utils import *
from moving_average import init_ema_model
from torch.utils.tensorboard.writer import SummaryWriter

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--name", help="Experiment name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--dname", help="Distillation name. Data will be saved to ./checkpoints/<name>/<dname>/.", type=str, required=True)
    parser.add_argument("--base_checkpoint", help="Path to base checkpoint.", type=str, required=True)
    parser.add_argument("--gamma", help="Gamma factor for SNR weights.", type=float, default=0)
    parser.add_argument("--checkpoint_to_continue", help="Path to checkpoint.", type=str, default="")
    parser.add_argument("--num_timesteps", help="Num diffusion steps.", type=int, default=1024)
    parser.add_argument("--num_iters", help="Num iterations.", type=int, default=100000)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=5e-5)
    parser.add_argument("--scheduler", help="Learning rate scheduler.", type=str, default="StrategyLinearLR")
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusionDefault")
    parser.add_argument("--log_interval", help="Log interval in minutes.", type=int, default=15)
    parser.add_argument("--ckpt_interval", help="Checkpoints saving interval in minutes.", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument ("--k", type=int, default=1)
    parser.add_argument("--num_steps", type = int , default = 1000)
    return parser

def distill_model(args, make_model, make_dataset):
    if args.num_workers == -1:
        args.num_workers = args.batch_size * 2

    need_student_ema = True
    if args.scheduler.endswith("SWA"):
        need_student_ema = False

    # print(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    device = torch.device("cuda")
    train_dataset = test_dataset = InfinityDataset(make_dataset(), args.batch_size)

     # len(train_dataset), len(test_dataset)

    img, anno = train_dataset[0]

    teacher_ema = make_model().to(device)

    image_size = teacher_ema.image_size

    checkpoints_dir = os.path.join("checkpoints", args.name, args.dname)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


    # ckpt = torch.load(args.base_checkpoint)
    # old_state_dict = ckpt["G"]

   
    # out_weight_key = "out.2.weight"
    # out_bias_key = "out.2.bias"

   
    # old_out_weight = old_state_dict[out_weight_key]  
    # old_out_bias = old_state_dict[out_bias_key] 
    # print("Shape of out.2.weight before loading:", old_out_weight.shape)
    # print("Shape of out.2.bias before loading:", old_out_bias.shape)
    # print (old_out_weight.shape[0])
    # print(3 * args.k)

    # if old_out_weight.shape[0]== (3  ):
    #     # # Only keep last 3
    #     # new_out_weight = old_out_weight[-3:]             # shape: [3, C, 3, 3]
    #     # new_out_bias = old_out_bias[-3:]                 # shape: [3]

    #     # Load into teacher model
    #     new_state_dict = teacher_ema.state_dict()
    #     new_state_dict[out_weight_key] = repeated_weight
    #     new_state_dict[out_bias_key] = repeated_bias

    #     teacher_ema.load_state_dict(new_state_dict, strict=False)

    # else:
    #     ckpt = torch.load(args.base_checkpoint)
    #     teacher_ema.load_state_dict(ckpt["G"])
    # n_timesteps = ckpt["n_timesteps"]
    # time_scale = ckpt["time_scale"]
    # del ckpt
    # print(f"Num timesteps: {n_timesteps}, time scale: {time_scale}.")


    ckpt = torch.load(args.base_checkpoint)
    old_state_dict = ckpt["G"]

   
    out_weight_key = "conv_out.weight"
    out_bias_key = "conv_out.bias"

   
    old_out_weight = old_state_dict[out_weight_key]  
    old_out_bias = old_state_dict[out_bias_key]      

    if old_out_weight.shape[0]== (3 * args.k ):
        print ('1')
        # Only keep last 3
        new_out_weight = old_out_weight[-3:]             # shape: [3, C, 3, 3]
        new_out_bias = old_out_bias[-3:]                 # shape: [3]

        # Load into teacher model
        new_state_dict = teacher_ema.state_dict()
        new_state_dict[out_weight_key] = new_out_weight
        new_state_dict[out_bias_key] = new_out_bias

        teacher_ema.load_state_dict(new_state_dict, strict=False)

    else:
       # print ('0')
        ckpt = torch.load(args.base_checkpoint)
        teacher_ema.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]
    time_scale = ckpt["time_scale"]
    del ckpt
    print(f"Num timesteps: {n_timesteps}, time scale: {time_scale}.")

    def make_scheduler():
        M = importlib.import_module("train_utils")
        D = getattr(M, args.scheduler)
        return D()

    scheduler = make_scheduler()
    distillation_model = DiffusionDistillation(scheduler, args ) #main distillation thingy defined in train_utils

    def make_diffusion( args , model, n_timestep, time_scale, device):
        betas   = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("diffusion")
        D = getattr(M, args.diffusion)
       # r = D(args , model, betas , time_scale=time_scale)
        r = D(args, model , betas , time_scale = time_scale )
        r.gamma = args.gamma
        return r

    #teacher_ema_diffusion = make_diffusion(args, teacher_ema, n_timesteps, time_scale, device)
    teacher_ema_diffusion = make_diffusion(args , teacher_ema , n_timesteps , time_scale , device )

    student = make_model( args ).to(device)
    if need_student_ema:
        student_ema = make_model(args ).to(device)
    else:
        student_ema = None

    if args.checkpoint_to_continue != "":
        ckpt = torch.load(args.checkpoint_to_continue)
        student.load_state_dict(ckpt["G"])
        student_ema.load_state_dict(ckpt["G"])
        del ckpt

    #distill_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tensorboard = SummaryWriter(os.path.join(checkpoints_dir, "tensorboard"))

    if args.checkpoint_to_continue == "":
        init_ema_model (teacher_ema, student, device , k=args.k)
        init_ema_model (teacher_ema , student_ema , device, k =args.k )
        print("Teacher parameters copied.")
    else:
        print("Continue training...")

    # need to change code here also ( 2 ki jagah k)
    student_diffusion = make_diffusion(args , student, teacher_ema_diffusion.num_timesteps // (args.k), teacher_ema_diffusion.time_scale * (args.k), device)
    if need_student_ema:
        student_ema_diffusion = make_diffusion(args , student_ema, teacher_ema_diffusion.num_timesteps // (args.k), teacher_ema_diffusion.time_scale * (args.k), device)

    on_iter = make_iter_callback(student_ema_diffusion, device, checkpoints_dir, image_size, tensorboard, args.log_interval, args.ckpt_interval, False)


    #Main  
    distillation_model.train_student_debug(make_dataset,args, teacher_ema_diffusion, student_diffusion, student_ema, args.lr, device, make_extra_args=make_condition, on_iter=on_iter)
    print("Finished.")


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")
    make_dataset = getattr(M, "make_dataset")

    distill_model(args, make_model, make_dataset)