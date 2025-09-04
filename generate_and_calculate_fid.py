#!/usr/bin/env python3
"""
Single script to generate 20k images and calculate FID score.
"""

import os
import argparse
import importlib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy.linalg
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusion import make_beta_schedule
from copy import deepcopy
from train_utils import make_visualization
import urllib.request

class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder."""
    
    def __init__(self, path, max_size=None, random_seed=0):
        self.path = path
        self.random_seed = random_seed
        
        # List all image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            import glob
            self.image_files.extend(glob.glob(os.path.join(path, ext)))
            self.image_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
        
        self.image_files = sorted(self.image_files)
        
        if max_size is not None:
            np.random.seed(random_seed)
            if len(self.image_files) > max_size:
                indices = np.random.choice(len(self.image_files), max_size, replace=False)
                self.image_files = [self.image_files[i] for i in sorted(indices)]
        
        # Define transforms for Inception-v3
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, 0  # Return dummy label

def download_cifar10_refs():
    """Download CIFAR-10 reference statistics if not available."""
    cifar10_url = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
    cifar10_file = "cifar10-32x32.npz"
    
    if not os.path.exists(cifar10_file):
        print(f"Downloading {cifar10_file} from {cifar10_url}...")
        try:
            urllib.request.urlretrieve(cifar10_url, cifar10_file)
            print(f"Downloaded {cifar10_file}")
        except Exception as e:
            print(f"Failed to download reference statistics: {e}")
            print("Please manually download from:")
            print("https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz")
            return None
    else:
        print(f"{cifar10_file} already exists.")
    
    return cifar10_file

def calculate_inception_stats(image_path, num_expected=None, seed=0, max_batch_size=64, 
                            num_workers=3, device=torch.device('cuda')):
    """Calculate Inception statistics for FID computation."""
    print('Loading Inception-v3 model...')
    
    # Load Inception-v3 model
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove final classification layer
    inception_model.eval()
    inception_model.to(device)
    
    # List images
    print(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    
    if len(dataset_obj) < 2:
        raise ValueError(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')
    
    if num_expected is not None and len(dataset_obj) < num_expected:
        print(f'Warning: Found {len(dataset_obj)} images, but expected at least {num_expected}')
    
    # Create data loader
    data_loader = DataLoader(
        dataset_obj, 
        batch_size=max_batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Accumulate statistics
    print(f'Calculating statistics for {len(dataset_obj)} images...')
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    
    for images, _labels in tqdm(data_loader, unit='batch', desc="Computing Inception features"):
        if images.shape[0] == 0:
            continue
        
        # Get features from Inception model
        with torch.no_grad():
            features = inception_model(images.to(device)).to(torch.float64)
        
        mu += features.sum(0)
        sigma += features.T @ features
    
    # Calculate final statistics
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    
    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    """Calculate FID from Inception statistics."""
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def p_sample_loop_batch(diffusion, noise, extra_args, device, eta=0, samples_to_capture=-1, need_tqdm=True, clip_value=3):
    """Modified sampling loop that handles batches."""
    mode = diffusion.net_.training
    diffusion.net_.eval()
    
    img = noise  # [batch_size, C, H, W]
    imgs = []
    
    iter_ = reversed(range(diffusion.num_timesteps))
    c_step = diffusion.num_timesteps/samples_to_capture
    next_capture = c_step
    
    if need_tqdm:
        iter_ = tqdm(iter_)
    
    for i in iter_:
        # Process entire batch at once
        t = torch.full((img.shape[0],), i, dtype=torch.int64).to(device)
        img = diffusion.p_sample(img, t, extra_args, eta=eta)
        
        if diffusion.num_timesteps - i > next_capture:
            imgs.append(img.clone())
            next_capture += c_step
    
    imgs.append(img)
    diffusion.net_.train(mode)
    return imgs

def make_visualization_batch(diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    """Generate a batch of images using the diffusion model."""
    extra_args = {}
    batch_size = image_size[0]
    
    # Generate noise for entire batch
    noise = torch.randn(image_size, device=device)  # [batch_size, C, H, W]
    
    # Run diffusion sampling on entire batch
    imgs = p_sample_loop_batch(diffusion, noise, extra_args, device, 
                              samples_to_capture=10, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    
    # Process each image in the batch
    batch_images = []
    for i in range(batch_size):
        # Extract single image from batch
        single_img = imgs[-1][i]  # Take final timestep, i-th image
        
        # Convert to numpy format
        img_np = single_img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img_np = (255 * (img_np + 1) / 2).clip(0, 255).astype(np.uint8)
        batch_images.append(img_np)
    
    return batch_images

def generate_images(args, make_model):
    """Generate images using the trained model with batch processing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    teacher = make_model().to(device)
    
    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("diffusion")
        D = getattr(M, args.diffusion)
        sampler = "ddim"
        return D(model, betas, time_scale=time_scale, sampler=sampler)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    old_state_dict = ckpt["G"]
    
    # Handle output layer adaptation
    out_weight_key = "conv_out.weight"
    out_bias_key = "conv_out.bias"
    
    old_out_weight = old_state_dict[out_weight_key]
    old_out_bias = old_state_dict[out_bias_key]
    
    if old_out_weight.shape[0] > 3:
        new_out_weight = old_out_weight[-3:]
        new_out_bias = old_out_bias[-3:]
        
        new_state_dict = teacher.state_dict()
        new_state_dict[out_weight_key] = new_out_weight
        new_state_dict[out_bias_key] = new_out_bias
        
        teacher.load_state_dict(new_state_dict, strict=False)
    else:
        teacher.load_state_dict(ckpt["G"])
    
    teacher.eval()
    n_timesteps = ckpt["n_timesteps"] // args.time_scale
    time_scale = ckpt["time_scale"] * args.time_scale
    del ckpt
    print(f"Model loaded. Timesteps: {n_timesteps}, Time scale: {time_scale}")
    
    # Create diffusion model
    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
    image_size = deepcopy(teacher.image_size)
    
    # Calculate number of batches needed
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    print(f"Generating {args.num_images} images in {num_batches} batches...")
    
    image_count = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Adjust batch size for the last batch
        current_batch_size = min(args.batch_size, args.num_images - batch_idx * args.batch_size)
        
        # Set batch size for this iteration
        batch_image_size = deepcopy(image_size)
        batch_image_size[0] = current_batch_size
        
        # Generate entire batch at once
        with torch.no_grad():
            batch_images = make_visualization_batch(
                teacher_diffusion, 
                device, 
                batch_image_size, 
                need_tqdm=False, 
                eta=args.eta, 
                clip_value=args.clipping_value
            )
        
        # Save all images in batch
        for i, img in enumerate(batch_images):
            img_filename = os.path.join(args.output_dir, f"image_{image_count:06d}.png")
            cv2.imwrite(img_filename, img)
            image_count += 1
            
            if image_count >= args.num_images:
                break
        
        if image_count >= args.num_images:
            break
    
    print(f"Generated {image_count} images in {args.output_dir}")
    return image_count

def main():
    parser = argparse.ArgumentParser(description="Generate images and calculate FID")
    parser.add_argument("--module", help="Model module.", type=str, default="cifar10_u")
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, 
                       default="./checkpoints/cifar10/original/checkpoint.pt")
    parser.add_argument("--output_dir", help="Output directory for images.", type=str, 
                       default="./fid_images")
    parser.add_argument("--batch_size", help="Batch size for generation.", type=int, default=32)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise.", type=float, default=0)
    parser.add_argument("--num_images", help="Number of images to generate.", type=int, default=20000)
    parser.add_argument("--ref_stats", help="Path to reference statistics file.", type=str, 
                       default="./cifar10-32x32.npz")
    parser.add_argument("--skip_generation", action="store_true", 
                       help="Skip image generation and only calculate FID")
    parser.add_argument("--fid_batch_size", help="Batch size for FID calculation.", type=int, default=64)
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Generate images
    if not args.skip_generation:
        print("="*60)
        print("STEP 1: GENERATING IMAGES")
        print("="*60)
        
        # Import model module
        M = importlib.import_module(args.module)
        make_model = getattr(M, "make_model")
        
        generated_count = generate_images(args, make_model)
        
        if generated_count < args.num_images:
            print(f"Warning: Only generated {generated_count} images out of {args.num_images}")
    
    # Step 2: Download reference statistics if needed
    print("\n" + "="*60)
    print("STEP 2: PREPARING REFERENCE STATISTICS")
    print("="*60)
    
    ref_file = download_cifar10_refs()
    if ref_file is None:
        print("Failed to get reference statistics. Exiting.")
        return
    
    # Step 3: Calculate FID
    print("\n" + "="*60)
    print("STEP 3: CALCULATING FID")
    print("="*60)
    
    # Check if reference statistics exist
    if not os.path.exists(args.ref_stats):
        print(f"Error: Reference statistics file not found: {args.ref_stats}")
        return
    
    # Load reference statistics
    print(f'Loading dataset reference statistics from "{args.ref_stats}"...')
    with open(args.ref_stats, 'rb') as f:
        ref = dict(np.load(f))
    
    # Calculate Inception statistics for generated images
    mu, sigma = calculate_inception_stats(
        image_path=args.output_dir, 
        num_expected=args.num_images, 
        seed=0, 
        max_batch_size=args.fid_batch_size,
        device=device
    )
    
    # Calculate FID
    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    
    # Save results
    with open('fid_results.txt', mode='a') as f:
        f.write(f'Generated_Images_{args.num_images}: {fid}\n')
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Generated images: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    print(f"FID Score: {fid:.4f}")
    print(f"Results saved to: fid_results.txt")
    print("="*60)

if __name__ == "__main__":
    main()