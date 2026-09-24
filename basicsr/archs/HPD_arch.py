import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights
import math
import time
from thop import profile  # 需要安装: pip install thop

class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros"):
        super().__init__()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.dw = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                           dilation=dilation, groups=out_channels, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.dw(self.pw(x))

class PartialBSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1, bias=True, 
                 padding_mode="zeros", scale=2):
        super().__init__()
        self.remaining_channels = in_channels // scale
        self.other_channels = in_channels - self.remaining_channels
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pdw = nn.Conv2d(self.remaining_channels, self.remaining_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, dilation=1, groups=self.remaining_channels, 
                            bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        fea1, fea2 = torch.split(x, [self.remaining_channels, self.other_channels], dim=1)
        fea1 = self.pdw(fea1)
        fea = torch.cat((fea1, fea2), 1)
        return self.pw(fea)

class MultiScaleFrequencyAttention(nn.Module):

    def __init__(self, embed_dim, scales=[1, 2, 4], fft_norm="ortho"):
        super().__init__()
        self.embed_dim = embed_dim
        self.scales = scales
        self.fft_norm = fft_norm
        

        self.scale_branches = nn.ModuleList()
        for scale in scales:
            branch = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim // 4, embed_dim, 1)
            )
            self.scale_branches.append(branch)
        

        self.fusion = nn.Conv2d(embed_dim * len(scales), embed_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        for branch in self.scale_branches:
            for m in branch:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape  
        fft_dim = (-2, -1)
        

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real, imag = ffted.real, ffted.imag  
        
        multi_scale_real = []
        multi_scale_imag = []
        

        for i, scale in enumerate(self.scales):

            scale_h = H // scale
            scale_w = (W // 2 + 1) // scale  
            
            if scale == 1:
                
                real_scaled = real
                imag_scaled = imag
            else:
               
                real_scaled = F.interpolate(real, size=(scale_h, scale_w), 
                                           mode='bilinear', align_corners=True)
                imag_scaled = F.interpolate(imag, size=(scale_h, scale_w), 
                                           mode='bilinear', align_corners=True)
            
            
            real_processed = real_scaled + self.scale_branches[i](real_scaled)
            imag_processed = imag_scaled + self.scale_branches[i](imag_scaled)
            
            
            real_processed = F.interpolate(real_processed, size=(H, W//2 + 1), 
                                          mode='bilinear', align_corners=True)
            imag_processed = F.interpolate(imag_processed, size=(H, W//2 + 1), 
                                          mode='bilinear', align_corners=True)
            
            multi_scale_real.append(real_processed)
            multi_scale_imag.append(imag_processed)
        
       
        real_fused = self.fusion(torch.cat(multi_scale_real, dim=1))
        imag_fused = self.fusion(torch.cat(multi_scale_imag, dim=1))
        

        ffted = torch.complex(real_fused, imag_fused)
        output = torch.fft.irfftn(ffted, s=(H, W), dim=fft_dim, norm=self.fft_norm)
        
        return x + output * self.gamma

class ProgressiveDistillation(nn.Module):

    def __init__(self, in_channels, out_channels, num_stages=3, reduction=4):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        self.out_channels = out_channels
        

        self.stage_channels_list = []
        current_channels = in_channels
        for i in range(num_stages):

            stage_channels = max(current_channels, out_channels)
            self.stage_channels_list.append(stage_channels)

            current_channels = max(current_channels // 2, out_channels)
        

        for stage_channels in self.stage_channels_list:
            stage = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Conv2d(stage_channels, stage_channels // reduction, 1, bias=False), 
                nn.GELU(),
                nn.Conv2d(stage_channels // reduction, stage_channels, 1, bias=False),
                nn.Sigmoid()
            )
            self.stages.append(stage)
        

        total_channels = sum(self.stage_channels_list)
        self.fusion = nn.Conv2d(total_channels, out_channels, 1)
        self.distill_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.GELU()
        
        self._init_weights()
        
    def _init_weights(self):
        for stage in self.stages:
            for m in stage:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fusion.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.distill_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.shape
        distilled_features = []
        current = x
        
        for i in range(self.num_stages):

            stage_channels = self.stage_channels_list[i]

            if current.shape[1] != stage_channels:
                current = nn.Conv2d(current.shape[1], stage_channels, 1).to(current.device)(current)
            

            ca = self.stages[i](current)
            distilled = current * ca
            distilled_features.append(distilled)
            

            if i < self.num_stages - 1:
                next_channels = self.stage_channels_list[i+1]

                if current.shape[1] > next_channels:
                    current = current[:, :next_channels, :, :]  
                else:
                    current = current 
        

        fused = self.fusion(torch.cat(distilled_features, dim=1))
        return self.act(fused)

class DynamicWindowAttention(nn.Module):

    def __init__(self, dim, num_heads=8, max_window_size=16, min_window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        

        self.window_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 2, 1),  
            nn.Sigmoid()
        )
        

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_window_size - 1) * (2 * max_window_size - 1), num_heads)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)
        
        for m in self.window_predictor:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_relative_position_index(self, window_size):

        coords = torch.stack(torch.meshgrid([
            torch.arange(window_size[0]), 
            torch.arange(window_size[1])
        ]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        return relative_coords.sum(-1)

    def forward(self, x):
        B, H, W, C = x.shape
        

        spatial_feat = x.permute(0, 3, 1, 2)
        window_pred = self.window_predictor(spatial_feat)
        

        h_window = max(self.min_window_size, 
                      min(self.max_window_size, 
                          int(H * window_pred[:, 0].mean().clamp(0.1, 0.9))))
        w_window = max(self.min_window_size, 
                      min(self.max_window_size, 
                          int(W * window_pred[:, 1].mean().clamp(0.1, 0.9))))
        

        h_window = (h_window // 4) * 4
        w_window = (w_window // 4) * 4
        

        num_windows_h = (H + h_window - 1) // h_window
        num_windows_w = (W + w_window - 1) // w_window
        

        pad_h = (num_windows_h * h_window - H) % h_window
        pad_w = (num_windows_w * w_window - W) % w_window
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W
        

        x = x.view(B, num_windows_h, h_window, num_windows_w, w_window, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, h_window, w_window, C)
        

        x_windows = x.view(-1, h_window * w_window, C)
        

        qkv = self.qkv(x_windows).view(-1, h_window * w_window, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        

        relative_position_index = self.get_relative_position_index((h_window, w_window))
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            h_window * w_window, h_window * w_window, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        

        x = (attn @ v).transpose(1, 2).reshape(-1, h_window * w_window, C)
        x = self.proj(x)
        

        x = x.view(-1, h_window, w_window, C)
        

        x = x.view(B, num_windows_h, num_windows_w, h_window, w_window, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, C)
        

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        return x

class HPDB(nn.Module):

    def __init__(self, in_channels, out_channels, atten_channels=None, 
                 use_multi_scale_freq=True, use_progressive_distill=True,
                 use_dynamic_window=True, num_distill_stages=3, window_heads=8):
        super().__init__()
        
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if atten_channels is None:
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels
            
        self.use_multi_scale_freq = use_multi_scale_freq
        self.use_progressive_distill = use_progressive_distill
        self.use_dynamic_window = use_dynamic_window


        if use_progressive_distill:

            self.c1_d = ProgressiveDistillation(in_channels, self.dc, num_stages=num_distill_stages)
            self.c2_d = ProgressiveDistillation(self.rc, self.dc, num_stages=num_distill_stages)
            self.c3_d = ProgressiveDistillation(self.rc, self.dc, num_stages=num_distill_stages)
        else:

            self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
            self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
            self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
            

        self.c1_r = PartialBSConvU(in_channels, self.rc, kernel_size=5, padding=2)
        self.c2_r = PartialBSConvU(self.rc, self.rc, kernel_size=5, padding=2)
        self.c3_r = PartialBSConvU(self.rc, self.rc, kernel_size=5, padding=2)
        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()


        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1)
        

        if use_multi_scale_freq:
            self.freq_attention = MultiScaleFrequencyAttention(self.atten_channels, scales=[1, 2, 4])
        else:
            self.freq_attention = nn.Identity()
            
        if use_dynamic_window:
            self.window_attention = DynamicWindowAttention(self.atten_channels, num_heads=window_heads)
        else:
            self.window_attention = nn.Identity()
        
        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1)
        self.pixel_norm = nn.LayerNorm(out_channels)
        

        self.alpha = nn.Parameter(torch.tensor(0.5))  
        self.beta = nn.Parameter(torch.tensor(0.5))  
        
        self._init_weights()

    def _init_weights(self):

        modules_to_init = []
        if not self.use_progressive_distill:
            modules_to_init.extend([self.c1_d, self.c2_d, self.c3_d])
        modules_to_init.extend([self.c5, self.c6])
        
        for m in modules_to_init:
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                

        nn.init.constant_(self.alpha, 0.5)
        nn.init.constant_(self.beta, 0.5)

    def forward(self, x):

        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.act(self.c1_r(x))

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1))

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2))

        r_c4 = self.act(self.c4(r_c3))
        

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)


        out_freq = self.freq_attention(out)
        

        B, C, H, W = out.shape
        out_spatial = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_spatial = self.window_attention(out_spatial)
        out_spatial = out_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)
        

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        out = out + out_freq * alpha + out_spatial * beta
        
        out = self.c6(out)
        
        # LayerNorm
        out = out.permute(0, 2, 3, 1)
        out = self.pixel_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out + x

class UpsampleOneStep(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.upsample = nn.Sequential(*[conv, pixel_shuffle])

    def forward(self, x):
        return self.upsample(x)

class Upsampler_rep(nn.Module):
    """重参数化上采样器"""
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.conv3x3 = nn.Conv2d(in_channels * 2, out_channels * (upscale_factor**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        v1 = F.conv2d(x, self.conv1x1.weight, self.conv1x1.bias, padding=0)
        v1 = F.pad(v1, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.conv1x1.bias.view(1, -1, 1, 1)
        v1[:, :, 0:1, :] = b0_pad
        v1[:, :, -1:, :] = b0_pad
        v1[:, :, :, 0:1] = b0_pad
        v1[:, :, :, -1:] = b0_pad
        v2 = F.conv2d(v1, self.conv3x3.weight, self.conv3x3.bias, padding=0)
        out = self.conv1(x) + self.conv3(x) + v2
        return self.pixel_shuffle(out)

@ARCH_REGISTRY.register()
class HPD(nn.Module):

    
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=56,
        num_atten=56,
        num_block=8,
        upscale=3,
        num_in=4,
        upsampler="pixelshuffledirect",
        rgb_mean=(0.4488, 0.4371, 0.4040),
        use_multi_scale_freq = True,
        use_progressive_distill=True,
        use_dynamic_window=True,
        num_distill_stages=3,  
        window_heads=8
    ):
        super().__init__()
        
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.upscale = upscale
        self.num_block = num_block
        
        self.num_in = num_in
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        

        self.fea_conv = BSConvU(num_in_ch * num_in, num_feat, kernel_size=3, padding=1)
        

        self.blocks = nn.ModuleList()
        for i in range(num_block):
            block = HPDB(
                in_channels=num_feat, 
                out_channels=num_feat, 
                atten_channels=num_atten,
                use_multi_scale_freq=use_multi_scale_freq,
                use_progressive_distill=use_progressive_distill,
                use_dynamic_window=use_dynamic_window,
                num_distill_stages=num_distill_stages,
                window_heads=window_heads
            )
            self.blocks.append(block)
        

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1, 1, 0)
        self.GELU = nn.GELU()
        self.c2 = BSConvU(num_feat, num_feat, kernel_size=3, padding=1)
        

        if upsampler == "pixelshuffledirect":
            self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == "pixelshuffle_rep":
            self.upsampler = Upsampler_rep(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean
        

        input_cat = torch.cat([input] * self.num_in, dim=1)
        out_fea = self.fea_conv(input_cat)
        
   
        block_outputs = []
        x = out_fea
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        

        trunk = torch.cat(block_outputs, dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea
        

        output = self.upsampler(out_lr) + self.mean
        
        return output
        
    def get_flops(self, input_size=(3, 128, 128)):

        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        

        flops, params = profile(self, inputs=(input_tensor,), verbose=False)
        return flops, params
    
    def get_inference_time(self, input_size=(3, 128, 128), warmup=10, repeats=100):

        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        

        self.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(input_tensor)
        

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(repeats):
                _ = self(input_tensor)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time)  / repeats 
        return avg_time
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        

        try:
            flops, params = self.get_flops()
            inference_time = self.get_inference_time()
            
            info_str = f'#Params: {num_parameters / 10 ** 3:<.4f} [K]\n'
            info_str += f'FLOPs: {flops / 10 ** 9:<.2f} [G]\n'
            info_str += f'Inference Time: {inference_time:<.2f} [ms]'
        except Exception as e:
            info_str = f'#Params: {num_parameters / 10 ** 3:<.4f} [K]\n'
            info_str += f'FLOPs/Time calculation failed: {e}'
        
        return f'{self._get_name()}\n{info_str}'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HPD(upscale=4).to(device)
    x = torch.randn(2, 3, 128, 128).to(device)
    

    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    

    print("Testing inference time...")
    inference_time = model.get_inference_time(input_size=(3, 128, 128))
    

    print("Calculating FLOPs...")
    flops, params = model.get_flops(input_size=(3, 128, 128))
    

    output = model(x)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"Inference time: {inference_time:.2f}ms")
