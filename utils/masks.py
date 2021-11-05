import torch
import numpy

def __get_checkerboard_mask(shape, device, dtype):
    mesh_grid = torch.meshgrid(*[torch.arange(s, device=device, dtype=int) for s in shape])
    grid = torch.zeros(shape, device=device, dtype=int)
    for mgrid in mesh_grid:
        grid += mgrid
    return (grid % 2).to(dtype)



def get_mask(shape, pattern, device, dtype=torch.float):
    if pattern == "checkerboard":
        return __get_checkerboard_mask(shape, device, dtype)
    else:
        raise AttributeError("Unknown pattern: {}".format(pattern))
