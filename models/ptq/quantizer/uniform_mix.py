# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer

#mixedprecision
class UniformQuantizer_Mixed(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer_Mixed, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None
        # self.bit_type.bits = 6
    
    def update_quantization_params(self, *args, **kwargs):
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)

    def quant(self, inputs, scale=None, zero_point=None):
        
        if scale is None:
            scale = self.scale
        
        if zero_point is None:
            zero_point = self.zero_point
        dim = inputs.shape[-1]
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        # for j in range(len(scale)):
            # sca
        zero_point = zero_point.reshape(range_shape)
        
        # #아래 3줄 gpu 같은거 쓰도록 설정. 기존  fq-vit는 그냥 됐는데, 왜 이건 설정해줘야하는지모르겟음
        if scale.device!=inputs.device:
            scale = scale.to(inputs.device)
            zero_point = zero_point.to(inputs.device)
        outputs = inputs / scale + zero_point
        # outputs = outputs.round().clamp(self.bit_type.lower_bound,
        #                                 self.bit_type.upper_bound)
        # print("upper bound: ", )
        if self.observer.is_outlier:
            for i in range(dim):
                if self.observer.smallscale_index[i] == 1: #smallscale일때 6비트로 해줌 나머지는 8비트
                    outputs[:,:,i] = outputs[:,:,i].round().clamp(self.bit_type.lower_bound,((self.bit_type.upper_bound+1)/1-1)) #6비트적용시 4로 나누어주면됌..
                    # outputs[:,:,i] = outputs[:,:,i].round().clamp(self.bit_type.lower_bound,self.bit_type.upper_bound )
                else:
                    outputs[:,:,i] = outputs[:,:,i].round().clamp(self.bit_type.lower_bound,((self.bit_type.upper_bound+1)/1-1))
                    # outputs[:,:,i] = outputs[:,:,i].round().clamp(self.bit_type.lower_bound,self.bit_type.upper_bound)
                
        
        # print("scale:", self.scale)
        # print("zero_point: ", self.zero_point)

        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        device = inputs.device
        scale = scale.to(device)
        zero_point = zero_point.to(device)
        outputs = (inputs - zero_point) * scale
        return outputs
