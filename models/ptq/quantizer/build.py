# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer
from .uniform_mix import UniformQuantizer_Mixed
str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer, 'uniform_mix': UniformQuantizer_Mixed}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
