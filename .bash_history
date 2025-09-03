pip install tqdm lmdb opencv-python matplotlib tensorboard
pip install cv
python ./train.py --module cifar10_u --name cifar10 --num_timesteps 1024 --dname original --batch_size 128 --num_workers 4 --num_iters 128 --num_steps 100000
python ./train.py --module cifar10_u --name cifar10 --num_timesteps 1024 --dname original --batch_size 128 --num_workers 4 --num_iters 128 --num_steps 100000
