'''
reference: https://github.com/kuoweilai/pixelshuffle3d/blob/master/pixelshuffle3d.py
to try with this later---pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
'''
import torch
from torch import nn
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()       
        nOut = channels // self.scale ** 2

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
       
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
       
        return output.view(batch_size, nOut, out_depth, out_height, out_width)