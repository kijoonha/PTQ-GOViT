# # Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# import torch

# from .base import BaseObserver
# from .utils import lp_loss

# # method 1
# class OutlierObserver(BaseObserver):

#     def __init__(self, module_type, bit_type, calibration_mode,  percentile_sigma = 0.02):
#         super(OutlierObserver, self).__init__(module_type, bit_type,
#                                           calibration_mode)
#         self.percentile_sigma = 0.01
#         self.is_outlier = False
#         self.max_quantile = None
#         self.min_quantile = None
#         self.symmetric = self.bit_type.signed
#         self.outlier_count = 0

    
#     def fine_outlier(self, v):
#         v = self.reshape_tensor(v)
#         cur_max_channel = v.max(axis=1).values
#         cur_min_channel = v.min(axis=1).values
#         cur_max_quan = torch.quantile(cur_max_channel, 1-self.percentile_sigma)
#         cur_min_quan = torch.quantile(cur_min_channel, self.percentile_sigma) #하위 Percentile
#         layer_max = torch.max(cur_max_quan)
#         layer_min = torch.min(cur_min_quan)
        
#         # layer 전체의 max값이 cur_max_quan의 2배보다 큰지 확인
#         if layer_max > 2 * cur_max_quan:
#             self.is_outlier = True
#         else:
#             if layer_min < 2 * cur_min_quan:
#                 self.is_outlier = True
#             else:
#                 self.is_outlier = False 
            

#     def update(self, v): #minmax구하는것은 percentile 참조

#         v = self.reshape_tensor(v)
        
#         cur_max = v.max(axis=1).values
#         cur_min = v.min(axis=1).values

#         #outlier포함 min, max값을 구함
#         if self.max_val is None:
#             self.max_val = cur_max
#         else:
#             self.max_val = torch.max(cur_max, self.max_val)
#         cur_min = v.min(axis=1).values
#         if self.min_val is None:
#             self.min_val = cur_min
#         else:
#             self.min_val = torch.min(cur_min, self.min_val)
        
#         if self.is_outlier:
#             #percentile에 해당하는 min, max를 구함
#             cur_max_quantile = torch.quantile(cur_max, 1-self.percentile_sigma)
#             cur_min_quantile = torch.quantile(cur_min, self.percentile_sigma)  
#             #max
#             if self.max_quantile is None:
#                 self.max_quantile = cur_max_quantile
#             else:
#                 self.max_quantile = torch.max(cur_max_quantile, self.max_quantile)
#             #min
#             if self.min_quantile is None:
#                 self.min_quantile = cur_min_quantile
#             else:
#                 self.min_quantile = torch.min(cur_min_quantile, self.min_quantile)  
#             self.outlier_count += 1
#         #outlier없을때는 layerwise로 구함
#         else:
#             self.max_val = self.max_val.max()
#             self.min_val = self.min_val.min()
        
    #lp로스 연산을 하지 않아도 되는 장점이 있음. 연산 flops계산 비교할 수있을듯
    # def get_quantization_params(self, inputs, *args, **kwargs):
    #     max_val = self.max_val # ex) [128]
    #     min_val = self.min_val
    #     if self.is_outlier:
    #         print(self.max_val)
    #         num_channel = self.max_val.shape[0]
    #         qmax = self.bit_type.upper_bound
    #         qmin = self.bit_type.lower_bound
        
    #         zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         ##scale 방법 2가지 quantile기준 올라가기 or outlier기준 내리기
    #         scale1 = (self.max_quantile - self.min_quantile) / float(qmax - qmin)    
    #         scale1.clamp_(self.eps)
    #         scale_mask = torch.ones_like(max_val) #channel개수 만큼의 scalemask만듦
    #         #data랑 lploss 활용하는 fq-vit의 부분 없앰
    #         temp_max_quantile = self.max_quantile
    #         temp_min_quantile = self.min_quantile
    #         for j in range(num_channel): #channelwise의미
    #             for i in range(5): # 32배 까지 값을 늘려서해봄 *outlier를 약간 무시하는 효과
    #                 temp_max_quantile *= 2
    #                 temp_min_quantile *= 2
    #                 if temp_max_quantile<self.max_val[j] or temp_min_quantile>self.min_val[j]:
    #                     scale_mask[j] = scale_mask[j] * 2
    #                 else:
    #                     break
    #         if self.symmetric:
    #             zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         else:
    #             zero_point = qmin - torch.round(min_val / scale)
    #             zero_point.clamp_(qmin, qmax)
            
    #         scale = scale1 * scale_mask
    #     else:
    #         max_val = self.max_val
    #         min_val = self.min_val

    #         qmax = self.bit_type.upper_bound
    #         qmin = self.bit_type.lower_bound

    #         scale = torch.ones_like(max_val, dtype=torch.float32)
    #         zero_point = torch.zeros_like(max_val, dtype=torch.int64)

    #         if self.symmetric:
    #             max_val = torch.max(-min_val, max_val)
    #             scale = max_val / (float(qmax - qmin) / 2)
    #             scale.clamp_(self.eps)
    #             zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         else:
    #             scale = (max_val - min_val) / float(qmax - qmin)
    #             scale.clamp_(self.eps)
    #             zero_point = qmin - torch.round(min_val / scale)
    #             zero_point.clamp_(qmin, qmax)
    #     return scale, zero_point

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.

# Method2 - outlier를 살리는 방법
import torch

from .base import BaseObserver
from .utils import lp_loss


class TimeObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode,  percentile_sigma = 0.02):
        super(TimeObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)
        self.percentile_sigma = 0.01
        self.is_outlier = False
        self.max_quantile = None
        self.min_quantile = None
        self.symmetric = self.bit_type.signed
        self.outlier_count = 0

    

    def update(self, v): #minmax구하는것은 percentile 참조

        # v shape : B, T, H, W, C
        print(v.shape) #qact2, 4 :
        B,T,H,W,C = v.shape
         
        cur_max = v.max(axis=1).values
        cur_min = v.min(axis=1).values

        #outlier포함 min, max값을 구함
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        
        if self.is_outlier:
            #percentile에 해당하는 min, max를 구함
            cur_max_quantile = torch.quantile(cur_max, 1-self.percentile_sigma)
            cur_min_quantile = torch.quantile(cur_min, self.percentile_sigma)  
            #max
            if self.max_quantile is None:
                self.max_quantile = cur_max_quantile
            else:
                self.max_quantile = torch.max(cur_max_quantile, self.max_quantile)
            #min
            if self.min_quantile is None:
                self.min_quantile = cur_min_quantile
            else:
                self.min_quantile = torch.min(cur_min_quantile, self.min_quantile)  
           
            
        
    #lp로스 연산을 하지 않아도 되는 장점이 있음. 연산 flops계산 비교할 수있을듯
    # def get_quantization_params(self, inputs, *args, **kwargs):
    #     max_val = self.max_val # ex) [128]
    #     min_val = self.min_val
    #     global_max = max_val.max()
    #     global_min = min_val.min()
        
    #     if self.is_outlier:
    #         num_channel = self.max_val.shape[0]
    #         qmax = self.bit_type.upper_bound
    #         qmin = self.bit_type.lower_bound
        
    #         zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         scale_big = (global_max - global_min) / float(qmax - qmin)
    #         scale_big.clamp_(self.eps)
    #         temp_max = global_max
    #         temp_min = global_min
    #         scale_mask = torch.ones_like(max_val) #channel개수 만큼의 scalemask만듦
    #         for j in range(num_channel): #channelwise의미 ##여기부터 다시해보기
    #             for i in range(4): # 32배 까지 값을 늘려서해봄 *outlier를 약간 무시하는 효과
    #                 temp_max /= 2
    #                 temp_min /= 2
    #                 if max_val[j]>temp_max or min_val[j]<temp_min:
    #                     scale_mask[j] = scale_mask[j] / 2
    #                 else:
    #                     break
    #         scale = scale_big * scale_mask
    #         if self.symmetric:
    #             zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         else:
    #             zero_point = qmin - torch.round(min_val / scale)
    #             zero_point.clamp_(qmin, qmax)
            
            
    #     else:
    #         max_val = self.max_val
    #         min_val = self.min_val

    #         qmax = self.bit_type.upper_bound
    #         qmin = self.bit_type.lower_bound

    #         scale = torch.ones_like(max_val, dtype=torch.float32)
    #         zero_point = torch.zeros_like(max_val, dtype=torch.int64)

    #         if self.symmetric:
    #             max_val = torch.max(-min_val, max_val)
    #             scale = max_val / (float(qmax - qmin) / 2)
    #             scale.clamp_(self.eps)
    #             zero_point = torch.zeros_like(max_val, dtype=torch.int64)
    #         else:
    #             scale = (max_val - min_val) / float(qmax - qmin)
    #             scale.clamp_(self.eps)
    #             zero_point = qmin - torch.round(min_val / scale)
    #             zero_point.clamp_(qmin, qmax)
    #     return scale, zero_point
    
    #group quant
    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val # ex) [128]
        min_val = self.min_val
        global_max = max_val.max()
        global_min = min_val.min()
        
        if self.is_outlier:
            num_channel = self.max_val.shape[0]
            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound
        
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = torch.ones_like(max_val, dtype=torch.float32)

            scale_big = (global_max - global_min) / float(qmax - qmin)
            scale_small = (self.max_quantile - self.min_quantile) / float(qmax - qmin)
            scale_big.clamp_(self.eps)
            scale_small
            temp_max = global_max
            temp_min = global_min
            for j in range(num_channel): #channelwise의미 ##여기부터 다시해보기
                if max_val[j]>self.max_quantile*2 or min_val[j]<self.min_quantile*2:
                    scale[j] = scale_big
                else:
                    scale[j] = scale_small
                    
            if self.symmetric:
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            else:
                zero_point = qmin - torch.round(min_val / scale)
                zero_point.clamp_(qmin, qmax)
        else:
            
            

            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound
            scale_mask = torch.ones_like(max_val, dtype=torch.float32)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            max_val = self.max_val.max() #이미 채널별 최대, 최소값임
            min_val = self.min_val.min()

            
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
            scale = scale*scale_mask
            # print(scale)
        return scale, zero_point
                         
    def print_outlier_count(self):
        print("Outlier Count: ", self.outlier_count)
        
