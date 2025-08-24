#!/usr/bin/env python
# coding: utf-8

# # Train the base model 

# In[1]:


import os 
os.environ['CUDA_LAUNCH_BLOCKING']='1'
get_ipython().system('python ./train.py --module cifar10_u --name cifar10 --dname original --batch_size 1 --num_workers 4 --num_iters 15')


# # Model distillation

# ## Distillate to 512 steps

# In[11]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_0 --base_checkpoint ./checkpoints/cifar10/original/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 50 --log_interval 5 --k 2')


# ## Distillate to 256 steps

# In[ ]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_1 --base_checkpoint ./checkpoints/cifar10/base_0/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 5000 --log_interval 5')


# ## Distillate to 128 steps

# In[ ]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_2 --base_checkpoint ./checkpoints/cifar10/base_1/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 5000 --log_interval 5')


# ## Distillate to 64 steps

# In[1]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_3 --base_checkpoint ./checkpoints/cifar10/base_2/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 5000 --log_interval 5')


# ## Distillate to 32 steps

# In[ ]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_4 --base_checkpoint ./checkpoints/cifar10/base_3/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 10000 --log_interval 5')


# ## Distillate to 16 steps

# In[ ]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_5 --base_checkpoint ./checkpoints/cifar10/base_4/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 10000 --log_interval 5')


# ## Distillate to 8 steps

# In[ ]:


get_ipython().system('python ./distillate.py --module cifar10_u --diffusion GaussianDiffusionDefault --name cifar10 --dname base_6 --base_checkpoint ./checkpoints/cifar10/base_5/checkpoint.pt --batch_size 1 --num_workers 4 --num_iters 10000 --log_interval 5')


# # Image generation

# In[ ]:


get_ipython().system('python ./sample.py --out_file ./images/cifar10_u_6.png --module cifar10_u --checkpoint ./checkpoints/cifar10/base_6/checkpoint.pt --batch_size 1')


# In[ ]:


get_ipython().system('python ./sample.py --out_file ./images/cifar10_u_6_clipped.png --module cifar10_u --checkpoint ./checkpoints/cifar10/base_6/checkpoint.pt --batch_size 1 --clipped_sampling True --clipping_value 1.2')


# In[ ]:


get_ipython().system('python ./sample.py --out_file ./images/cifar10_original_ts1.png --module cifar10_u --time_scale 1 --checkpoint ./checkpoints/cifar10/original/checkpoint.pt --batch_size 1')


# In[ ]:


# export to script
get_ipython().system('jupyter nbconvert --to script cifar10_u_script.ipynb --output cifar10_u_script_converted')

