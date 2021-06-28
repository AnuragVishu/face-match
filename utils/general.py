import torch
import numpy as np
import os

def tensor2numpy(tensor, save):
    # save stacked tensors
    np.save(save, tensor)
    print(f'Saved to {save}.npy')

def numpy2tensor(np_path):
    tensor = torch.from_numpy(np.load(np_path+'.npy'))
    return tensor