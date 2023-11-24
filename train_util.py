from datetime import datetime

import torch
import torch.nn as nn

import os
from PIL import Image


def observe(
    env,
    repr_net: nn.Module,
    render_mode: str,
    train: bool = True,
    save_img_path: str = None,
):
    if render_mode == "rgb_array":
        rgb, seg, depth = env.render()
    elif render_mode == "rgbd_array":
        rgbd = env.render()
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

    if save_img_path is not None:
        rgb_im = Image.fromarray(rgb)
        rgb_im.save(save_img_path)

    rgb = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1)
    rgb = rgb[None, :, :, :]
    rgb.cuda()

    if train:
        state = repr_net(rgb)
    else:
        with torch.no_grad():
            state = repr_net(rgb)

    state = state.cpu()

    return state


def timestamp():
    ts = datetime.timestamp(datetime.now())
    return datetime.fromtimestamp(ts)
