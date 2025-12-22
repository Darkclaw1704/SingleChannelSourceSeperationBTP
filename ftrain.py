# import torch
# import torch.nn.functional as F
# import itertools
# import h5py
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path

# # Import your model definition
# # Assuming the first code block you provided is saved as dpft_model.py
# from dpft_model import DPFTSeparator, ORIGINAL_LENGTH
# import dpft_model

# # --- HYPERPARAMETERS ---
# FRAMES_PER_SAMPLE = 64  # 64 * 1024 = 65,536 samples
# BATCH_SIZE = 1          # Small batch size due to memory
# NUM_EPOCHS = 2
# N_FFT = 512
# HOP_SIZE = 256
# FS = 50e6
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- PATHS ---
# H5_PATH = "/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"
# SPLIT_DIR = Path("/home/bss/dev/") # Updated to your dev folder

# # --- 1. NEW DATASET CLASS FOR DETERMINISTIC SPLITS ---
# class BSSFromIndicesDataset(Dataset):
#     def __init__(self, h5_data, valid_indices, num_mixtures, target_length, frames_per_sample):
#         """
#         h5_data: The full loaded X array (N, 1024, 2)
#         valid_indices: The specific row numbers allowed for this set (from .npy file)
#         """
#         self.data = h5_data
#         self.valid_indices = valid_indices
#         self.num_mixtures = num_mixtures
#         self.target_length = target_length
#         self.frames_per_sample = frames_per_sample
        
#     def __len__(self):
#         return self.num_mixtures

#     def __getitem__(self, idx):
#         # DETERMINISM: Seed numpy using the sample index so the mixture is always consistent
#         rng = np.random.default_rng(seed=idx + len(self.valid_indices))

#         # 1. Select distinct frames for Source 1 and Source 2 from the ALLOWED indices
#         idx_s1 = rng.choice(self.valid_indices, size=self.frames_per_sample, replace=True)
#         idx_s2 = rng.choice(self.valid_indices, size=self.frames_per_sample, replace=True)
        
#         # 2. Fetch frames (Assuming I-channel only based on previous context)
#         # Using [:, :, 0] for I component.
#         frames_s1 = self.data[np.sort(idx_s1)][:, :, 0] # (64, 1024)
#         frames_s2 = self.data[np.sort(idx_s2)][:, :, 0] # (64, 1024)
        
#         # 3. Concatenate to make long sequence
#         S1 = frames_s1.flatten()[:self.target_length]
#         S2 = frames_s2.flatten()[:self.target_length]
        
#         # 4. Scale
#         s1_amp = rng.uniform(0.1, 1.0)
#         s2_amp = rng.uniform(0.1, 1.0)
#         S1_scaled = S1 * s1_amp
#         S2_scaled = S2 * s2_amp
        
#         # 5. Noise
#         noise_level = rng.uniform(0.001, 0.01)
#         noise = rng.normal(0, noise_level, self.target_length).astype(np.float32)
        
#         # 6. Mix
#         X_M_i = S1_scaled + S2_scaled + noise
        
#         # 7. Normalize
#         max_amp = np.max(np.abs(X_M_i))
#         if max_amp > 1e-8:
#             X_M_i /= max_amp
#             S1_norm = S1_scaled / max_amp
#             S2_norm = S2_scaled / max_amp
#         else:
#             S1_norm, S2_norm = S1_scaled, S2_scaled

#         # Return Tensors
#         X_M_tensor = torch.tensor(X_M_i, dtype=torch.float32).unsqueeze(0) # (1, L)
#         Y_GT_tensor = torch.stack([
#             torch.tensor(S1_norm, dtype=torch.float32), 
#             torch.tensor(S2_norm, dtype=torch.float32)
#         ], dim=0)
        
#         return X_M_tensor, Y_GT_tensor

# # --- 2. LOSS FUNCTIONS ---
# def sdsdr_loss(true_signal, estimated_signal):
#     if true_signal.dim() == 0: true_signal = true_signal.unsqueeze(0)
#     if estimated_signal.dim() == 0: estimated_signal = estimated_signal.unsqueeze(0)
#     if true_signal.dim() == 1: true_signal = true_signal.unsqueeze(0)
#     if estimated_signal.dim() == 1: estimated_signal = estimated_signal.unsqueeze(0)
#     s = true_signal.view(-1)
#     s_hat = estimated_signal.view(-1)
    
#     s_norm_sq = torch.sum(s * s)
#     s_s_hat_dot = torch.sum(s * s_hat)
    
#     if s_norm_sq < 1e-8:
#         s_target = torch.zeros_like(s)
#     else:
#         scale = s_s_hat_dot / s_norm_sq
#         s_target = scale * s
        
#     s_error = s_hat - s_target
#     s_target_norm_sq = torch.sum(s_target * s_target)
#     s_error_norm_sq = torch.sum(s_error * s_error)
#     epsilon = 1e-8
    
#     sdsdr = 10 * torch.log10(s_target_norm_sq / (s_error_norm_sq + epsilon) + epsilon)
#     return sdsdr

# def uPIT_SDSDR_Loss(true_signals, estimated_signals):
#     if true_signals.dim() == 2: true_signals = true_signals.unsqueeze(0)
#     s_hat = torch.stack(estimated_signals, dim=1) 
#     batch_size, num_channels, _ = true_signals.shape
    
#     sd_sdr_matrix = torch.zeros(batch_size, num_channels, num_channels, device=true_signals.device)
#     for i in range(num_channels):
#         for j in range(num_channels):
#             s_i = true_signals[:, i, :].reshape(batch_size, -1)
#             s_hat_j = s_hat[:, j, :].reshape(batch_size, -1)
#             sdsdr_batch = []
#             for b in range(batch_size):
#                 sdsdr_val = sdsdr_loss(s_i[b].flatten(), s_hat_j[b].flatten()) 
#                 sdsdr_batch.append(sdsdr_val)
#             sd_sdr_matrix[:, i, j] = torch.stack(sdsdr_batch)
            
#     permutations = list(itertools.permutations(range(num_channels)))
#     max_avg_sdsdr = []
    
#     for b in range(batch_size):
#         sdsdr_per_perm = []
#         for perm in permutations:
#             sdsr_sum = 0
#             for i in range(num_channels):
#                 sdsr_sum += sd_sdr_matrix[b, i, perm[i]]
#             avg_sdsdr = sdsr_sum / num_channels
#             sdsdr_per_perm.append(avg_sdsdr)
#         max_avg_sdsdr.append(torch.max(torch.stack(sdsdr_per_perm)))
        
#     final_loss = -torch.mean(torch.stack(max_avg_sdsdr))
#     return final_loss

# # --- 3. MAIN SETUP ---

# # Load Data Once
# try:
#     print(f"Loading data from {H5_PATH}...")
#     f = h5py.File(H5_PATH, "r")
#     # Load all X data into memory (required for speed)
#     X_all_data = f['X'][:] 
#     f.close()
#     print("Data loaded into memory.")
# except Exception as e:
#     print(f"Error loading HDF5: {e}")
#     exit()

# # Load Indices from Step 1
# try:
#     print("Loading split indices...")
#     idx_train = np.load(SPLIT_DIR / "train_indices.npy")
#     idx_val   = np.load(SPLIT_DIR / "val_indices.npy")
#     idx_test  = np.load(SPLIT_DIR / "test_indices.npy")
#     print(f"Indices loaded: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")
# except FileNotFoundError:
#     print(f"Error: Index files not found in {SPLIT_DIR}. Please run the 'Step 1' script first!")
#     exit()

# # Determine number of mixtures
# n_train_mixtures = len(idx_train) // FRAMES_PER_SAMPLE
# n_val_mixtures   = len(idx_val)   // FRAMES_PER_SAMPLE
# n_test_mixtures  = len(idx_test)  // FRAMES_PER_SAMPLE

# # Create Datasets using Specific Indices
# train_dataset = BSSFromIndicesDataset(X_all_data, idx_train, n_train_mixtures, ORIGINAL_LENGTH, FRAMES_PER_SAMPLE)
# val_dataset   = BSSFromIndicesDataset(X_all_data, idx_val,   n_val_mixtures,   ORIGINAL_LENGTH, FRAMES_PER_SAMPLE)
# test_dataset  = BSSFromIndicesDataset(X_all_data, idx_test,  n_test_mixtures,  ORIGINAL_LENGTH, FRAMES_PER_SAMPLE)

# # Loaders
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
# val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# # --- 4. MODEL INIT ---
# # Fix for custom STFT in model
# def compute_stft_custom(xm, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=None):
#     if win_length is None: win_length = n_fft
#     window = torch.hann_window(win_length, device=xm.device)
#     if xm.dim() > 2: xm = xm.squeeze(1) 
#     if xm.shape[-1] < n_fft:
#         pad_amount = n_fft - xm.shape[-1]
#         xm = F.pad(xm, (0, pad_amount)) 
#     Xm = torch.stft(xm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
#                     window=window, return_complex=True, normalized=False)
#     return Xm

# dpft_model.compute_stft = compute_stft_custom

# print("Initializing Model...")
# model = DPFTSeparator()
# model.to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # --- 5. TRAINING LOOP ---
# print("\nStarting Training...")
# train_losses = []
# val_losses = []
# best_val = float('inf')

# for epoch in range(NUM_EPOCHS):
#     model.train()
#     running_loss = 0.0
    
#     current_lr = 1e-4 * (0.90 ** epoch)
#     for pg in optimizer.param_groups: pg['lr'] = current_lr
    
#     for i, (xm, ym) in enumerate(train_loader):
#         xm, ym = xm.to(DEVICE), ym.to(DEVICE)
        
#         optimizer.zero_grad()
#         xo1, xo2 = model(xm)

#         # --- FIX: Handle 1D output when BATCH_SIZE=1 ---
#         if xo1.dim() == 1: xo1 = xo1.unsqueeze(0)
#         if xo2.dim() == 1: xo2 = xo2.unsqueeze(0)
#         # -----------------------------------------------
        
#         # Reshape for loss (B, L)
#         loss = uPIT_SDSDR_Loss(ym, [xo1.flatten(1), xo2.flatten(1)])
        
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#         if i % 10 == 0:
#             print(f"  Ep {epoch} Step {i} Loss: {loss.item():.4f}", end='\r')
            
#     avg_train = running_loss / len(train_loader)
#     train_losses.append(avg_train)
    
#     # Validation
#     model.eval()
#     run_val = 0.0
#     with torch.no_grad():
#         for xm, ym in val_loader:
#             xm, ym = xm.to(DEVICE), ym.to(DEVICE)
#             xo1, xo2 = model(xm)
            
#             # --- FIX: Handle 1D output here too ---
#             if xo1.dim() == 1: xo1 = xo1.unsqueeze(0)
#             if xo2.dim() == 1: xo2 = xo2.unsqueeze(0)
#             # --------------------------------------

#             run_val += uPIT_SDSDR_Loss(ym, [xo1.flatten(1), xo2.flatten(1)]).item()
            
#     avg_val = run_val / len(val_loader)
#     val_losses.append(avg_val)
    
#     print(f"\nEpoch {epoch+1}: Train={avg_train:.4f} dB, Val={avg_val:.4f} dB")
    
#     if avg_val < best_val:
#         best_val = avg_val
#         torch.save(model.state_dict(), "best_dpft_model.pth")
#         print("  -> Saved Best Model")

# print("Training Done.")

# # --- 6. TESTING LOOP (Added for completeness) ---
# print("\nStarting Testing...")
# # Load best weights
# try:
#     model.load_state_dict(torch.load("best_dpft_model.pth", map_location=DEVICE))
#     print("Loaded best weights.")
# except:
#     print("Using final weights (best model not found).")

# model.eval()
# run_test = 0.0
# with torch.no_grad():
#     for xm, ym in test_loader:
#         xm, ym = xm.to(DEVICE), ym.to(DEVICE)
#         xo1, xo2 = model(xm)

#         # --- FIX: Handle 1D output here too ---
#         if xo1.dim() == 1: xo1 = xo1.unsqueeze(0)
#         if xo2.dim() == 1: xo2 = xo2.unsqueeze(0)
#         # --------------------------------------

#         run_test += uPIT_SDSDR_Loss(ym, [xo1.flatten(1), xo2.flatten(1)]).item()

# avg_test = run_test / len(test_loader)
# print(f"Final Test Loss: {avg_test:.4f} dB")



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