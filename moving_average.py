import torch
import torch.nn as nn 


"""def init_ema_model(model, model_ema, device=None):
    with torch.no_grad():
        for (mp, ep) in zip(model.parameters(), model_ema.parameters()):
            data = mp.data
            if device is not None:
                data = data.to(device)
            ep.data.copy_(data)"""

def init_ema_model(model, model_ema,device=None , k=1 ):
    with torch.no_grad():
        conv_layers = [m for m in model_ema.modules() if isinstance(m, nn.Conv2d)]  #list of all conv layers bnali
        target_layer = conv_layers[-1] if k > 1 else None  

        for (mp, ep) in zip(model.parameters(), model_ema.parameters()):
            data = mp.data
            if device is not None:
                data = data.to(device)
            # if target_layer is not None and ep.shape[0] == target_layer.out_channels * k:
            #     if data.dim() == 4 : 
            #         data = data.repeat(k, 1, 1, 1)  #teachermodel weights repeatt
            #     elif data.dim() == 1:  
            #         data = data.repeat(k)

            if ep.data.shape[0] == k * mp.data.shape[0] and mp.data.dim() == 4:
                data = mp.data.repeat(k, 1, 1, 1)
            elif ep.data.shape[0] == k * mp.data.shape[0] and mp.data.dim() == 1:
                data = mp.data.repeat(k)
                
         
            ep.data.copy_(data)

def moving_average(model, model_ema, beta=0.999, device=None):
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            data = param.data
            if device is not None:
                data = data.to(device)
            param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
