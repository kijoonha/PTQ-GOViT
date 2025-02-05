from models import BIT_TYPE_DICT
# class Config:

#     def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
#         '''
#         ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
#         lis stands for Log-Int-Softmax.
#         These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
#         '''
#         self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
#         self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
#         self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint6']
#         self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

#         self.OBSERVER_W = 'minmax'
#         self.OBSERVER_A = quant_method
#         self.OBSERVER_A_attn = quant_method

#         self.QUANTIZER_W = 'uniform'
#         self.QUANTIZER_A = 'uniform'
#         self.QUANTIZER_A_LN = 'uniform'
#         self.QUANTIZER_A_OUT = 'uniform'

#         self.CALIBRATION_MODE_W = 'channel_wise'
#         self.CALIBRATION_MODE_A = 'layer_wise'
#         self.CALIBRATION_MODE_S = 'layer_wise'
#         self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
#         if lis:
#             self.INT_SOFTMAX = True
#             self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
#             self.OBSERVER_S = 'minmax'
#             self.QUANTIZER_S = 'log2'
#         else:
#             self.INT_SOFTMAX = False
#             self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
#             self.OBSERVER_S = self.OBSERVER_A
#             self.QUANTIZER_S = self.QUANTIZER_A
#         if ptf:
#             self.INT_NORM = True
#             self.OBSERVER_A_LN = 'ptf'
#             # self.OBSERVER_A = 'ptf'
#             # self.CALIBRATION_MODE_A = 'channel_wise'
#             self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
#         else:
#             self.INT_NORM = False
#             self.OBSERVER_A_LN = "percentile"
#             self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
#         if outlier:
#             self.INT_NORM = True
#             self.OBSERVER_A_LN = 'outlier'
#             # self.OBSERVER_A = 'outlier'
            

#             self.CALIBRATION_MODE_A_LN = 'channel_wise'
#             # self.QUANTIZER_A_LN = 'uniform_mix'
#             self.QUANTIZER_A_LN = 'uniform'
#         else:
#             pass
        
#         #실험 configuration
#         self.outlier = outlier    
#         self.ivit_layernorm = False
        
#         # if timeoutlier:
#         #     self.INT_NORM = True
#         #     self.OBSERVER_A_OUT = 'timeoutlier'
#         #     self.CALIBRATION_MODE_A_OUT = 'layer_wise'
#         # else:
#         #     self.INT_NORM = True
#         #     self.OBSERVER_A_OUT = 'ptf'
#         #     self.CALIBRATION_MODE_A_OUT = 'channel_wise'
#         print("observer: ", self.OBSERVER_A_LN)
        
        
# class Config_partially4bit:

#     def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
#         '''
#         ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
#         lis stands for Log-Int-Softmax.
#         These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
#         '''
#         self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
#         self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
#         self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint4']
#         self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

#         self.OBSERVER_W = 'minmax'
#         self.OBSERVER_A = quant_method
#         self.OBSERVER_A_attn = quant_method

#         self.QUANTIZER_W = 'uniform'
#         self.QUANTIZER_A = 'uniform'
#         self.QUANTIZER_A_LN = 'uniform'
#         self.QUANTIZER_A_OUT = 'uniform'

#         self.CALIBRATION_MODE_W = 'channel_wise'
#         self.CALIBRATION_MODE_A = 'layer_wise'
#         self.CALIBRATION_MODE_S = 'layer_wise'
#         self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
#         if lis:
#             self.INT_SOFTMAX = True
#             self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
#             self.OBSERVER_S = 'minmax'
#             self.QUANTIZER_S = 'log2'
#         else:
#             self.INT_SOFTMAX = False
#             self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
#             self.OBSERVER_S = self.OBSERVER_A
#             self.QUANTIZER_S = self.QUANTIZER_A
#         if ptf:
#             self.INT_NORM = True
#             self.OBSERVER_A_LN = 'ptf'
#             # self.OBSERVER_A = 'ptf'
#             # self.CALIBRATION_MODE_A = 'channel_wise'
#             self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
#         else:
#             self.INT_NORM = False
#             self.OBSERVER_A_LN = self.OBSERVER_A
#             self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
#         if outlier:
#             self.INT_NORM = True
#             self.OBSERVER_A_LN = 'outlier'
#             # self.OBSERVER_A = 'outlier'

#             self.CALIBRATION_MODE_A_LN = 'channel_wise'
#             # self.QUANTIZER_A_LN = 'uniform_mix'
#             self.QUANTIZER_A_LN = 'uniform'
#         else:
#             pass
        

#         #실험 configuration
#         self.outlier = outlier    
#         self.ivit_layernorm = False
#         self.ivit_INTSOFTMAX = False
        
class Config_8bit:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            # self.OBSERVER_A = 'ptf'
            # self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            # self.OBSERVER_A = 'outlier'
            

            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False


class Config_partially6bit:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            # self.OBSERVER_A = 'ptf'
            # self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            # self.OBSERVER_A = 'outlier'
            

            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False

        
        
class Config_partially4bit:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint4']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            # self.OBSERVER_A = 'ptf'
            # self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            # self.OBSERVER_A = 'outlier'
            

            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            self.QUANTIZER_A_LN = 'uniform_mix'
            # self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False
        
class Config_6bit_method_ivitsoftmax:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int6']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint6']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.OBSERVER_A = 'ptf'
            self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            self.OBSERVER_A = 'outlier'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = True
        
class Config_6bit_method:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int6']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint6']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.OBSERVER_A = 'ptf'
            self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            self.OBSERVER_A = 'outlier'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False
        
class Config_8bit_method:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint8']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.OBSERVER_A = 'ptf'
            self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            self.OBSERVER_A = 'outlier'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False
        

class Config_4bit_method:

    def __init__(self, ptf=True, lis=True, outlier = False, quant_method='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_LN = BIT_TYPE_DICT['uint6']
        self.BIT_TYPE_A_attn = BIT_TYPE_DICT['uint6']

        self.OBSERVER_W = 'minmax'
        self.OBSERVER_A = quant_method
        self.OBSERVER_A_attn = quant_method

        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'
        self.QUANTIZER_A_OUT = 'uniform'

        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'
        self.CALIBRATION_MODE_A_attn = 'layer_wise'
        
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.OBSERVER_A = 'ptf'
            self.CALIBRATION_MODE_A = 'channel_wise'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
        
        if outlier:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'outlier'
            self.OBSERVER_A = 'outlier'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
            # self.QUANTIZER_A_LN = 'uniform_mix'
            self.QUANTIZER_A_LN = 'uniform'
        else:
            pass
        

        #실험 configuration
        self.outlier = outlier    
        self.ivit_layernorm = False
        self.ivit_INTSOFTMAX = False