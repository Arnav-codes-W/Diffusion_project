from Unets.unet_final import UNet_final
#from unet_ddim import UNet
from cifar10_dataset import CIFAR10Wrapper

def make_model(args=None):
    # Default k = 1, unless args and args.k are provided
    k_value = 1
    if args is not None and hasattr(args, "k") and args.k is not None:
        k_value = args.k

    model = UNet_final(
        in_channel=3,
        channel=256,
        channel_multiplier=[1, 2, 2, 4],
        n_res_blocks=2,
        attn_strides=[8],
        k=k_value,
    )
    model.image_size = [1, 3, 32, 32]
    return model
    
    ## parameters
    num_classes=1,
    ch=256,
    emb_ch=1024,
    out_ch=3,  # e.g., RGB output
    ch_mult=(1, 1, 1),
    num_res_blocks=3,
    attn_resolutions=(8, 16),
    num_heads=1,
    dropout=0.2,
    logsnr_input_type="inv_cos",
    resblock_resample=True,

def make_dataset():
    return CIFAR10Wrapper(train=True)
