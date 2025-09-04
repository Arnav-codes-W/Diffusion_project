pip install tqdm lmdb opencv-python matplotlib tensorboard
pip install cv
python ./train.py --module cifar10_u --name cifar10 --num_timesteps 1024 --dname original --batch_size 128 --num_workers 4 --num_iters 128 --num_steps 100000
python ./train.py --module cifar10_u --name cifar10 --num_timesteps 1024 --dname original --batch_size 128 --num_workers 4 --num_iters 128 --num_steps 100000
# Basic usage (generates 20k images and calculates FID)
python generate_and_calculate_fid.py
# Custom parameters
python generate_and_calculate_fid.py     --checkpoint ./checkpoints/cifar10/original/checkpoint.pt     --output_dir ./my_fid_images     --num_images 20000     --batch_size 128 \
python generate_and_calculate_fid.py \ ----checkpoint ./checkpoints/cifar10/original/checkpoint.pt \ --output_dir ./my_fid_images \ --num_images 20 \ --batch_size 10 \ --fid_batch_size 10
