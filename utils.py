import glob
import os
import matplotlib
import torch
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import functools


def plot_f0_compared(f0, cleanf0):
    f0 = np.array(f0)
    cleanf0 = np.array(cleanf0)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(f0, label="hat")
    plt.plot(cleanf0, label="gt")
    plt.ylim(0,)
    plt.xlim(0,500)
    plt.legend()
    fig.canvas.draw()
    plt.close()
    return fig


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__