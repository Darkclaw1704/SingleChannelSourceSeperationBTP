import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import your model
from dpft_model import DPFTSeparator, ORIGINAL_LENGTH
import dpft_model

# --- CONFIG ---
H5_PATH = "/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"
SPLIT_DIR = Path("/home/bss/dev/")
MODEL_PATH = "best_dpft_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_PER_SAMPLE = 64
N_FFT = 512
HOP_SIZE = 256

# --- 1. RE-DEFINE DATASET (Must match training) ---
class BSSFromIndicesDataset(Dataset):
    def __init__(self, h5_data, valid_indices, num_mixtures, target_length, frames_per_sample):
        self.data = h5_data
        self.valid_indices = valid_indices
        self.num_mixtures = num_mixtures
        self.target_length = target_length
        self.frames_per_sample = frames_per_sample
        
    def __len__(self):
        return self.num_mixtures

    def __getitem__(self, idx):
        rng = np.random.default_rng(seed=idx + len(self.valid_indices))
        idx_s1 = rng.choice(self.valid_indices, size=self.frames_per_sample, replace=True)
        idx_s2 = rng.choice(self.valid_indices, size=self.frames_per_sample, replace=True)
        
        frames_s1 = self.data[np.sort(idx_s1)][:, :, 0] 
        frames_s2 = self.data[np.sort(idx_s2)][:, :, 0]
        
        S1 = frames_s1.flatten()[:self.target_length]
        S2 = frames_s2.flatten()[:self.target_length]
        
        s1_amp = rng.uniform(0.1, 1.0); S1_scaled = S1 * s1_amp
        s2_amp = rng.uniform(0.1, 1.0); S2_scaled = S2 * s2_amp
        noise = rng.normal(0, rng.uniform(0.001, 0.01), self.target_length).astype(np.float32)
        X_M_i = S1_scaled + S2_scaled + noise
        
        max_amp = np.max(np.abs(X_M_i))
        if max_amp > 1e-8:
            X_M_i /= max_amp; S1_norm = S1_scaled / max_amp; S2_norm = S2_scaled / max_amp
        else:
            S1_norm, S2_norm = S1_scaled, S2_scaled

        return (torch.tensor(X_M_i, dtype=torch.float32).unsqueeze(0), 
                torch.stack([torch.tensor(S1_norm), torch.tensor(S2_norm)]))

# --- 2. SETUP MODEL & DATA ---
# Monkey-patch STFT for model compatibility
def compute_stft_custom(xm, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=None):
    if win_length is None: win_length = n_fft
    window = torch.hann_window(win_length, device=xm.device)
    if xm.dim() > 2: xm = xm.squeeze(1) 
    if xm.shape[-1] < n_fft: xm = F.pad(xm, (0, n_fft - xm.shape[-1])) 
    return torch.stft(xm, n_fft, hop_length, win_length, window, return_complex=True)

dpft_model.compute_stft = compute_stft_custom

print("Loading Data & Indices...")
f = h5py.File(H5_PATH, "r"); X_all = f['X'][:]; f.close()
idx_test = np.load(SPLIT_DIR / "test_indices.npy")
test_ds = BSSFromIndicesDataset(X_all, idx_test, 10, ORIGINAL_LENGTH, FRAMES_PER_SAMPLE) # Load 10 samples
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print("Loading Model...")
model = DPFTSeparator().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Weights Loaded ✔")
except:
    print("⚠️ Could not load weights. Using random init.")

model.eval()

# --- 3. GENERATE PLOTS ---
print("Generating visualizations...")
# Get one batch (1 sample)
xm, ym = next(iter(test_loader))
xm, ym = xm.to(DEVICE), ym.to(DEVICE)

with torch.no_grad():
    xo1, xo2 = model(xm)
    # Handle dimensions
    if xo1.dim()==1: xo1 = xo1.unsqueeze(0)
    if xo2.dim()==1: xo2 = xo2.unsqueeze(0)

# Convert to Numpy for plotting
mix = xm[0].cpu().numpy().flatten()
s1_gt = ym[0,0].cpu().numpy().flatten()
s2_gt = ym[0,1].cpu().numpy().flatten()
s1_pred = xo1[0].cpu().numpy().flatten()
s2_pred = xo2[0].cpu().numpy().flatten()

# Plot 1: Waveforms
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1); plt.plot(mix, color='grey'); plt.title("Input Mixture"); plt.grid(alpha=0.3)
plt.subplot(3, 1, 2); plt.plot(s1_gt, 'g', label='GT', alpha=0.6); plt.plot(s1_pred, 'k--', label='Pred', alpha=0.8); plt.title("Source 1"); plt.legend()
plt.subplot(3, 1, 3); plt.plot(s2_gt, 'b', label='GT', alpha=0.6); plt.plot(s2_pred, 'k--', label='Pred', alpha=0.8); plt.title("Source 2"); plt.legend()
plt.tight_layout()
plt.savefig("eval_waveforms_dpft.png")
print("Saved eval_waveforms_dpft.png")

# Plot 2: Spectrograms
def plot_spec(ax, sig, title):
    ax.specgram(sig, NFFT=256, Fs=1, noverlap=128, cmap='magma')
    ax.set_title(title)

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
# Row 1: Mix
plot_spec(axes[0,0], mix, "Mixture")
axes[0,1].axis('off')
# Row 2: Source 1
plot_spec(axes[1,0], s1_gt, "GT Src 1")
plot_spec(axes[1,1], s1_pred, "Pred Src 1")
# Row 3: Source 2
plot_spec(axes[2,0], s2_gt, "GT Src 2")
plot_spec(axes[2,1], s2_pred, "Pred Src 2")
plt.tight_layout()
plt.savefig("eval_spectrograms_dpft.png")
print("Saved eval_spectrograms_dpft.png")