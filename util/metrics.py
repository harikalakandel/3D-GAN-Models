#https://github.com/bryanlimy/clinical-super-mri/blob/main/src/supermri/metrics/metrics.py
#https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/__init__.py
#https://github.com/AhmedIbrahimai/dice-coefficient-Semantic-Segmentation-computer-vison-opencv/blob/main/dice.py

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision
from torchvision.utils import make_grid
from pathlib import Path


import os
import h5py


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

from matplotlib import pyplot as plt




from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.data import to_categorical
from torchmetrics.utilities.distributed import reduce

from matplotlib import pyplot as plt


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def nmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Normalized mean squared error
    Note: normalize by y_true, not y_pred.
    """
    return mse(y_pred, y_true) / (y_true.norm() ** 2 + 1e-6)


def psnr(x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    """Computes peak signal to noise ratio (PSNR)
    Args:
      x: images in (N,C,H,W)
      y: images in (N,C,H,W)
      max_value: the maximum value of the images (usually 1.0 or 255.0)
    Returns:
      PSNR value
    """
    return 10 * torch.log10(max_value**2 / (mse(x, y) + 1e-6))


def visualise(images, name, path='.', slice=(42, 82)):
    if images.size(0) > 1:
        for i, img in enumerate(images):
            ax = plt.subplot()
            img = img[:, slice[0]:slice[1], :, :]
            x_grid = make_grid(img.permute(1, 0, 2, 3))
            x_grid = x_grid.permute(1, 2, 0).numpy()

            ax.imshow(x_grid)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f'{name} {i + 1}', fontsize=5)

            plt.savefig(Path(path, f'{name} {i + 1}.jpg'), dpi=1000, bbox_inches='tight')
            plt.close()

    else:
        ax = plt.subplot()
        images = images.squeeze(dim=0)[:, slice[0]:slice[1], :, :]
        x_grid = make_grid(images.permute(1, 0, 2, 3))
        x_grid = x_grid.permute(1, 2, 0).numpy()

        ax.imshow(x_grid)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=5)
        plt.savefig(Path(path, f'{name}.jpg'), dpi=1000, bbox_inches='tight')
        plt.close()






def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1-D Gaussian kernel with shape (1, 1, size)
    Args:
      size: the size of the Gaussian kernel
      sigma: sigma of normal distribution
    Returns:
      1D kernel (1, 1, size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(inputs: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Apply 1D Gaussian kernel to inputs images
    Args:
      inputs: a batch of images in shape (N,C,H,W)
      win: 1-D Gaussian kernel
    Returns:
      blurred images
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    channel = inputs.shape[1]
    outputs = inputs
    for i, s in enumerate(inputs.shape[2:]):
        if s >= win.shape[-1]:
            outputs = F.conv2d(
                outputs,
                weight=win.transpose(2 + i, -1),
                stride=1,
                padding=0,
                groups=channel,
            )
    return outputs


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def get_gradient_penalty(real_X, fake_X, discriminator,device='cpu'):
    alpha = torch.rand(size=(real_X.size(0), 1, 1, 1, 1), dtype=torch.float32)
    alpha = alpha.repeat(1, *real_X.size()[1:])
    
    alpha=alpha.to(device)

    interpolates = alpha * real_X + ((1 - alpha) * fake_X)
    interpolates = interpolates.requires_grad_(True)
    interpolates = interpolates.to(device)
    output = discriminator(interpolates)

    ones = torch.ones(size=output.size(), dtype=torch.float32)
    ones=ones.to(device)

    gradients = torch.autograd.grad(outputs=output, inputs=interpolates, grad_outputs=ones, create_graph=True, 
                                        retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradient_penalty

def get_gradient_penalty_WithFixSize(real_X, fake_X, discriminator,device='cpu'):
    alpha = torch.rand(size=(real_X.size(0), 1, 1, 2, 1), dtype=torch.float32)
    alpha = alpha.repeat(1, *real_X.size()[1:])
    
    alpha=alpha.to(device)

    interpolates = alpha * real_X + ((1 - alpha) * fake_X)
    interpolates = interpolates.requires_grad_(True)
    interpolates = interpolates.to(device)
    output = discriminator(interpolates)

    ones = torch.ones(size=output.size(), dtype=torch.float32)
    ones=ones.to(device)

    gradients = torch.autograd.grad(outputs=output, inputs=interpolates, grad_outputs=ones, create_graph=True, 
                                        retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradient_penalty


def dice_coefficient(y_true, y_pred, smooth=1):

    # Flatten the segmentation masks
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Calculate the intersection between the segmentation masks
    intersection = np.sum(y_true_f * y_pred_f)

    # Calculate the sum of the segmentation masks
    y_true_sum = np.sum(y_true_f)
    y_pred_sum = np.sum(y_pred_f)

    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
    dice= np.mean(dice)

    return dice


#https://github.com/Lightning-AI/torchmetrics/blob/v0.8.2/torchmetrics/functional/classification/dice.py#L62-L113

def _stat_scores(
    preds: Tensor,
    target: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates the number of true positive, false positive, true negative and false negative for a specific
    class.

    Args:
        preds: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = _stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))
    """
    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def dice_score(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
) -> Tensor:
    """Compute dice score from prediction scores.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor containing dice score

    Example:
        >>> from torchmetrics.functional import dice_score
        >>> pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.85, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.85, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.85]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> dice_score(pred, target)
        tensor(0.3333)
    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
    for i in range(bg_inv, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg_inv] += no_fg_score
            continue

        # TODO: rewrite to use general `stat_scores`
        tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score

        scores[i - bg_inv] += score_cls
    return reduce(scores, reduction=reduction)
