import torch
import torch.nn as nn
import torch.nn.functional as F
import math

FEATURE_DIM_N = 128
N_HEADS = 2 
KERNEL_SIZE = (4, 4)
STRIDES = (2, 2)



# N_FFT = 128  
N_FFT=512
# HOP_LENGTH = 128
HOP_LENGTH = 256
# ORIGINAL_LENGTH = 16384
ORIGINAL_LENGTH = 65280


DROPOUT_RATE = 0.1
I_TRANSFORMER_LAYERS = 2
J_DPTF_STACKS = 2

def compute_stft_safe(x, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=None):
    if win_length is None: win_length = n_fft
    window = torch.hann_window(win_length, device=x.device)
    
    if x.shape[-1] < n_fft:
        pad_amount = n_fft - x.shape[-1]
        x = F.pad(x, (0, pad_amount))  
    
    Xm = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )
    return Xm



def preprocess_signal(xm):
    """
    xm: [B, C, T]  (batch, channels, time)
    Returns:
        Xm2_4d: [B, 4, F_padded, T_new] - LogAmp/Phase stack
        Xm_original_batched: [B, F, T_new] - Complex STFT of the *first* channel
    """
    B, C, T_old = xm.shape
    Xm2_4d_list = []
    Xm_original_batched_list = []



    for b in range(B):
        sig_mix = xm[b, 0] 
        Xm_mix = compute_stft_safe(sig_mix) 
        Xm_original_batched_list.append(Xm_mix)
        
        F, T_new = Xm_mix.shape
        F_padded = F + 1
        
        channel_features = []
        for c in range(C):
            sig = xm[b, c] 
            Xm_c = compute_stft_safe(sig) 
            Xm_c = torch.cat([Xm_c, torch.zeros(1, T_new, device=Xm_c.device)], dim=0)
            Xm_abs = torch.abs(Xm_c)
            Xm_logamp = torch.log10(torch.clamp(Xm_abs, min=1e-10))
            Xm_phase = torch.angle(Xm_c)
            channel_features.append(Xm_logamp)
            channel_features.append(Xm_phase)  
        Xm2_4d_list.append(torch.stack(channel_features, dim=0))
    Xm2_4d = torch.stack(Xm2_4d_list, dim=0)
    Xm_original_batched = torch.stack(Xm_original_batched_list, dim=0)
    return Xm2_4d, Xm_original_batched 



def post_processing_block(Xs_4d,Xm_original_batched):
    Xs_logamp = Xs_4d[:, :2] 
    Xs_phase = Xs_4d[:, 2:]  
    Xs_linamp = torch.pow(10, Xs_logamp)
    M_real = Xs_linamp * torch.cos(Xs_phase)
    M_imag = Xs_linamp * torch.sin(Xs_phase)
    Xo_4d_complex = torch.complex(M_real, M_imag)
    Xo_4d_complex = Xo_4d_complex[:, :, :-1, :]
    Xo1 = Xo_4d_complex[:, 0] 
    Xo2 = Xo_4d_complex[:, 1] 
    Xo1_separated = Xo1 * Xm_original_batched
    Xo2_separated = Xo2 * Xm_original_batched
    
    return Xo1_separated, Xo2_separated

def compute_istft(X, length=ORIGINAL_LENGTH):
    window = torch.hann_window(N_FFT, device=X.device)
    xm_out = torch.istft(
        X, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT,
        window=window, return_complex=False, length=length
    )
    return xm_out.squeeze(0)



class PermuteLayerNorm(nn.Module):
    """Applies LayerNorm over the feature dimension N after permuting [B, C, F, T] -> [B, F, T, C]"""
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) 
        return x

class SeparableConv2d(nn.Module):
    """
    Simulates SeparableConv2D used in the paper (II.C)
    The paper cites Keras/TensorFlow, which has a native separable layer.
    In PyTorch, this is implemented as Depthwise followed by Pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FeatureExtractor(nn.Module):
    def __init__(self, N=FEATURE_DIM_N, K=KERNEL_SIZE, S=STRIDES):
        super().__init__()
      

        self.separable_conv1 = SeparableConv2d(2, N, kernel_size=K, stride=1, padding=(K[0]//2, K[1]//2))

        self.norm1 = PermuteLayerNorm(N)
      
        self.separable_conv_down = SeparableConv2d(N, N, kernel_size=K, stride=S)
        self.norm_down = PermuteLayerNorm(N)
      
        self.conv_final = nn.Conv2d(N, N, kernel_size=1)
        self.norm_final = PermuteLayerNorm(N)

    def forward(self, x):

        x = self.norm1(self.separable_conv1(x))

        x = self.norm_down(self.separable_conv_down(x))
        x = F.relu(x)
        
        Xe = self.norm_final(self.conv_final(x))
        return Xe

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding as required by the paper (II.D, Eq. 8)"""
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, Z):
        seq_len = Z.size(1)
        E = self.pe[:seq_len, :Z.size(2)]
        return Z + E.unsqueeze(0) 


class TransformerEncoder(nn.Module):
    """
    Implements I stacks of vanilla Transformer Encoder layers (Fig. 2c).
    Applies g(.) I times.
    """
    def __init__(self, dim, num_heads, dropout=DROPOUT_RATE, num_layers=I_TRANSFORMER_LAYERS):
        super().__init__()
        self.pos_encoder = PositionalEncoding(dim)

        self.layers = nn.ModuleList([self._make_g_block(dim, num_heads, dropout) for _ in range(num_layers)])

    def _make_g_block(self, dim, num_heads, dropout):
        """Implements the g(.) block (Eq. 9, 10)"""
        return nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
            nn.LayerNorm(dim), 
            nn.Linear(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Dropout(dropout)
        ])
    
    def forward(self, Z):
    
        Z_start = Z
        Z1 = self.pos_encoder(Z) 
        Z_out = Z1
        for mha, norm1, ffn1, ffn2, norm2, drop1, drop2 in self.layers:
            residual1 = Z_out
            attn_out, _ = mha(Z_out, Z_out, Z_out)
            Z_out = norm1(residual1 + drop1(attn_out)) 
            
            residual2 = Z_out
            ffw_out = ffn2(F.relu(ffn1(Z_out)))
            Z_out = norm2(residual2 + drop2(ffw_out))
    
        Z4 = Z_out + Z_start
        return Z4


class DPTFBlock(nn.Module):
    """
    Implements the h(.) module: F-TE followed by T-TE (in series) (Fig. 2b)
    Note: The paper implies shared weights across blocks (II.D)
    """
    def __init__(self, N=FEATURE_DIM_N, H=N_HEADS):
        super().__init__()
        self.f_te = TransformerEncoder(N, H)
        self.t_te = TransformerEncoder(N, H)
        
    def forward(self, x):
        B, N, F_p, T_p = x.shape 
        
        x_freq = x.permute(0, 3, 2, 1).reshape(B * T_p, F_p, N)
        x_freq = self.f_te(x_freq)
        x_out = x_freq.reshape(B, T_p, F_p, N).permute(0, 3, 2, 1)

        x_time = x_out.permute(0, 2, 3, 1).reshape(B * F_p, T_p, N)
        x_time = self.t_te(x_time)
        x_out = x_time.reshape(B, F_p, T_p, N).permute(0, 3, 1, 2)
        
        return x_out


class FeatureTransformer(nn.Module):
    """The Feature Transformer block (Fig. 2a)"""
    def __init__(self, N=FEATURE_DIM_N, num_stacks=J_DPTF_STACKS):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([DPTFBlock(N) for _ in range(num_stacks)])
        
        self.conv_tanh = nn.Conv2d(N, N, kernel_size=1)
        self.conv_sigmoid = nn.Conv2d(N, N, kernel_size=1)
        self.conv_final = nn.Conv2d(N, N, kernel_size=1)
        self.norm_final = PermuteLayerNorm(N)

    def forward(self, x):
        Xt1 = x
        for block in self.transformer_blocks:
            Xt1 = block(Xt1)
            
        tanh_out = torch.tanh(self.conv_tanh(Xt1))
        sigmoid_out = torch.sigmoid(self.conv_sigmoid(Xt1))
        Xt2 = tanh_out * sigmoid_out
        
        Xt = self.norm_final(self.conv_final(Xt2))
        return Xt


class Separator(nn.Module):
    def __init__(self, N=FEATURE_DIM_N, K=KERNEL_SIZE, S=STRIDES):
        super().__init__()
        self.conv_up = nn.ConvTranspose2d(N, N, kernel_size=K, stride=S, output_padding=1)
        self.norm1 = PermuteLayerNorm(N)
    
        self.conv_final = nn.Conv2d(N, 4, kernel_size=1) 

    def forward(self, x, F_padded, T):
       
        
        x = self.conv_up(x)
        
        x = F.interpolate(x, size=(F_padded, T), mode='nearest')
        
        Xs1 = F.relu(self.norm1(x))
        
        Xs = self.conv_final(Xs1)
        return Xs


class DPFTSeparator(nn.Module):
    def __init__(self, num_transformer_stacks=J_DPTF_STACKS):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_transformer = FeatureTransformer(num_stacks=num_transformer_stacks)
        self.separator = Separator()
        
    def forward(self, xm):
        Xm2_4d, Xm_original_batched = preprocess_signal(xm) 
        
        B, C, F_padded, T = Xm2_4d.shape
        Xe = self.feature_extractor(Xm2_4d)
        Xt = self.feature_transformer(Xe)
        Xs = self.separator(Xt, F_padded, T) 
        Xo1_separated, Xo2_separated = post_processing_block(Xs, Xm_original_batched)
        xo1 = compute_istft(Xo1_separated)
        xo2 = compute_istft(Xo2_separated)
        
        return xo1, xo2
