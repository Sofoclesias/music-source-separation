"""
Módulo para layers.
"""

import torch
from torch import nn

def centre_crop(x, target):
    diff = x.shape[-1] - target.shape[-1]
    assert (diff % 2 == 0)
    crop = diff // 2
    
    if crop == 0:
        return x
    else:
        return x[:, :, crop:-crop].contiguous()

class ConvLayer1D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, act_func, transpose = False):
        super(ConvLayer1D, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = act_func
        
        if self.transpose:
            self.filter = nn.ConvTranspose1d(
                in_channels = n_inputs,
                out_channels = n_outputs,
                kernel_size = kernel_size,
                stride = stride,
                padding = kernel_size - 1
            )
        else:
            self.filter = nn.Conv1d(
                in_channels = n_inputs,
                out_channels = n_outputs,
                kernel_size = kernel_size,
                stride = stride
            )
        
    def forward(self, x):
        return self.activation(self.filter(x))
    
class ConvLayer2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, act_func, transpose = False):
        super(ConvLayer2D, self).__init__()
        self.transpose = transpose
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = act_func
        
        if self.transpose:
            self.filter = nn.ConvTranspose2d(
                in_channels = n_inputs,
                out_channels = n_outputs,
                kernel_size = kernel_size,
                stride = stride,
                padding = kernel_size - 1
            )
        else:
            self.filter = nn.Conv2d(
                in_channels = n_inputs,
                out_channels = n_outputs,
                kernel_size = kernel_size,
                stride = stride
            )
        
    def forward(self, x):
        return self.activation(self.filter(x))
    
class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, act_func, waveform = False):
        super(UpsamplingBlock, self).__init__()
        
        if waveform:
            self.upconv = ConvLayer1D(
                n_inputs=n_inputs,
                n_outputs=n_inputs,
                kernel_size=kernel_size,
                stride=stride,
                act_func=act_func,
                transpose=True
            )
            
            self.pre_shortcut_convs = nn.ModuleList([ConvLayer1D(n_inputs,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer1D(n_inputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])
            
            self.post_shortcut_convs = nn.ModuleList([ConvLayer1D(n_outputs + n_shortcut,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer1D(n_outputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])
        else:
            self.upconv = ConvLayer2D(
                n_inputs=n_inputs,
                n_outputs=n_inputs,
                kernel_size=kernel_size,
                stride=stride,
                act_func=act_func,
                transpose=True
            )
            
            self.pre_shortcut_convs = nn.ModuleList([ConvLayer2D(n_inputs,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer2D(n_inputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])
            
            self.post_shortcut_convs = nn.ModuleList([ConvLayer2D(n_outputs + n_shortcut,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer2D(n_outputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])
    
    def forward(self, x, shortcut):
        upsampled = self.upconv(x)
        
        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)
    
        combined = centre_crop(shortcut, upsampled)
        
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)],dim=1))
    
        return combined
    
class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, act_func, waveform = False):
        super(DownsamplingBlock, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        if waveform:
            self.pre_shortcut_convs = nn.ModuleList([ConvLayer1D(n_inputs,n_shortcut,kernel_size,1,act_func)] +
                                                    [ConvLayer1D(n_shortcut,n_shortcut,kernel_size,1,act_func) for _ in range(depth - 1)])
            
            self.post_shortcut_convs = nn.ModuleList([ConvLayer1D(n_shortcut,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer1D(n_outputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])

            self.downconv = ConvLayer1D(
                n_inputs=n_outputs,
                n_outputs=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                act_func=act_func
            )
        
        else:
            self.pre_shortcut_convs = nn.ModuleList([ConvLayer2D(n_inputs,n_shortcut,kernel_size,1,act_func)] +
                                                    [ConvLayer2D(n_shortcut,n_shortcut,kernel_size,1,act_func) for _ in range(depth - 1)])
            
            self.post_shortcut_convs = nn.ModuleList([ConvLayer2D(n_shortcut,n_outputs,kernel_size,1,act_func)] +
                                                    [ConvLayer2D(n_outputs,n_outputs,kernel_size,1,act_func) for _ in range(depth - 1)])

            self.downconv = ConvLayer2D(
                n_inputs=n_outputs,
                n_outputs=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                act_func=act_func
            )
            
    def forward(self, x):
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)
            
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)
            
        out = self.downconv(out)
        
        return out, shortcut