import os
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, map_location, logger=None, path=None):
    if path is not None:
        checkpoint = torch.load(path, map_location)
        print("checkpoint loaded")
    else:    
        path = '/Users/strom/Desktop/monodle/utilities/checkpoints/checkpoint_epoch_140.pth'
        checkpoint = torch.load(path, map_location)
        print("default checkpoint loaded")
    epoch = checkpoint.get('epoch', -1)
    if model is not None and checkpoint['model_state'] is not None:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and checkpoint['optimizer_state'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    logger.info("==> Done")

    return epoch