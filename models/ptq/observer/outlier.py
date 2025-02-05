import torch

from .base import BaseObserver
from .utils import lp_loss


class OutlierObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode,  percentile_sigma = 0.02):
        super(OutlierObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)
        self.percentile_sigma = 0.01
        self.is_outlier = False
        self.max_quantile = None
        self.min_quantile = None
        self.symmetric = self.bit_type.signed
        self.outlier_count = 0
        # self.bit_type.bits = 6  # 4bit로 바꿔줌
        # self.bit_type_small.bits = 6
        self.smallscale_index = []  
        self.outlier_multiplier = 2
        

    
    def find_outlier(self, v):
        v = self.reshape_tensor(v)
        cur_max_channel = v.max(axis=1).values
        cur_min_channel = v.min(axis=1).values
        cur_max_quan = torch.quantile(cur_max_channel, 1-self.percentile_sigma)
        cur_min_quan = torch.quantile(cur_min_channel, self.percentile_sigma) #하위 Percentile
        layer_max = torch.max(cur_max_channel)
        layer_min = torch.min(cur_min_channel)
        
        # layer 전체의 max값이 cur_max_quan의 2배보다 큰지 확인
        if layer_max > self.outlier_multiplier * cur_max_quan or layer_min < self.outlier_multiplier * cur_min_quan:
            self.is_outlier = True
            self.outlier_count += 1
        else:
            self.is_outlier = False 
        self.smallscale_index = [0] * len(cur_max_channel)

    def update(self, v): #minmax구하는것은 percentile 참조
        v = self.reshape_tensor(v)
        
        cur_max = v.max(axis=1).values
        cur_min = v.min(axis=1).values
        #outlier포함 min, max값을 구함
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)
        # if self.is_outlier:
            # print("self max val :", self.max_val)
        #     print("self min val :", self.min_val)
        #quantile updatae
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
           
            
    #group quant
    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val # ex) shape [128]
        min_val = self.min_val
        global_max = max_val.max()
        global_min = min_val.min()
        qmax = self.bit_type.upper_bound
        qmax_big = (self.bit_type.upper_bound+1)/1 -1
        q = [0,0,0,0]
        # qmax_small = qmax #qmax는 8비트, qmax_small은 4비트
        qmin = self.bit_type.lower_bound

        scale_big = (global_max - global_min) / float(qmax_big - qmin)
        if self.is_outlier:
            num_channel = self.max_val.shape[0]
            qmax_small = (self.bit_type.upper_bound+1)/1 -1
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = torch.ones_like(max_val, dtype=torch.float32)
            scale_big = (global_max - global_min) / float(qmax_big - qmin)
            scale_small = (self.max_quantile - self.min_quantile) / (float((qmax_small - qmin)))
            scale_small.clamp_(self.eps)
            scale_big.clamp_(self.eps)
            temp_max = global_max
            temp_min = global_min
            # scale_small*=2
            for j in range(num_channel):
                if max_val[j]>self.max_quantile* self.outlier_multiplier or min_val[j]<self.min_quantile * self.outlier_multiplier: #outlier값인 경우
                    
                    scale[j] = scale_big
                    self.smallscale_index[j] = 0
                    # print("scale_big: ", scale_big)
                    # zero_point[j] = qmin - torch.round(global_min/ scale[j])
                    # zero_point = qmin - torch.round(min_val / scale_big)
                    # zero_point.clamp_(qmin, qmax_big)
                    #스케일 추가구분
                    if max_val[j]<global_max/2 and min_val[j]>global_min/2: 
                        scale[j] = scale_big/2
                        # q[1] += 1
                    else:
                        scale[j] = scale_big
                        # q[0] += 1
                
                elif max_val[j]<self.max_quantile/2 and min_val[j]>self.min_quantile/2: #small scale 보다 더 작은값을 적용하게되면?
                    scale[j] = scale_small/2
                    # q[3] += 1
                else:
                    scale[j] = scale_small
                    self.smallscale_index[j] = 1
                    # q[2] += 1
                    # zero_point = qmin - torch.round(min_val / scale_small)##????끄냥 scale 되어잇었
                    # zero_point.clamp_(qmin, qmax_small)
                    # print("scale_small: ", scale_small)
                    
                    # zero_point[j] = qmin - torch.round(torch.quantile(min_val, self.percentile_sigma)/scale[j])
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax_small)
            #     zero_point.clamp_(qmin, qmax)
            #zero point잘 돌아가는 코드 확인
            if self.symmetric:
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            # else:

        else:
            qmax_small = (self.bit_type.upper_bound+1)/1 -1
            scale_mask = torch.ones_like(max_val, dtype=torch.float32)
            max_val = self.max_val.max() #이미 채널별 최대, 최소값임
            min_val = self.min_val.min()

            scale = (max_val - min_val) / float((qmax_small - qmin)) 
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax_small)
            scale = scale*scale_mask
            self.smallscale_index = [1] * len(self.max_val)
        # print("small scale index : ", self.smallscale_index)
            # print(scale)
        # if self.is_outlier:
        #     print("scale:", scale)
        #     print("zero: ", zero_point)
        # self.bit_type.bits = 6 
        # print("q : ", q)
        return scale, zero_point
                         
    def print_outlier_count(self):
        print("Outlier Count: ", self.outlier_count)
        


