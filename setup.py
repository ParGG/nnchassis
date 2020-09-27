import torch as th

def dev_setup():
    """
    Sets up cuda and folder structure for the project
    """
    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device