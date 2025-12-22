# # # import h5py
# # # import torch
# # # import numpy as np
# # # from dpft_model import DPFTSeparator 

# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print("Using device:", DEVICE)

# # # h5_path = r"/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"
# # # with h5py.File(h5_path, "r") as f:
# # #     print("Keys in HDF5 file:", list(f.keys()))
    

# # #     X = f['X'][:]  
# # #     Y = f['Y'][:]  


# # # subset_indices = np.arange(4) 
# # # X_subset = X[subset_indices]   
# # # Y_subset = Y[subset_indices]  
# # # print("Subset shape: X={}, Y={}".format(X_subset.shape, Y_subset.shape))


# # # xm_batch = torch.tensor(X_subset, dtype=torch.float32, device=DEVICE)

# # # xm_batch = xm_batch.permute(0, 2, 1).contiguous()  

# # # from dpft_model import compute_stft_safe

# # # N_FFT_SMALL = 256
# # # HOP_SMALL = 128

# # # def compute_stft_small(xm, n_fft=N_FFT_SMALL, hop_length=HOP_SMALL, win_length=None):
# # #     if win_length is None: win_length = n_fft
# # #     window = torch.hann_window(win_length, device=xm.device)
# # #     if xm.dim() > 1: xm = xm.squeeze(0)
# # #     Xm = torch.stft(
# # #         xm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
# # #         window=window, return_complex=True
# # #     )
# # #     return Xm

# # # import dpft_model
# # # dpft_model.compute_stft = compute_stft_small

# # # model = DPFTSeparator()
# # # model.to(DEVICE)
# # # model.eval()

# # # from torch.utils.data import TensorDataset, DataLoader

# # # num_samples = 25000  

# # # X = X[:num_samples]
# # # Y = Y[:num_samples]

# # # X_train = torch.tensor(X, dtype=torch.float32)
# # # Y_train = torch.tensor(Y, dtype=torch.float32)
# # # train_dataset = TensorDataset(X_train.permute(0, 2, 1).contiguous(), Y_train)
# # # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# # # criterion = torch.nn.L1Loss() 
# # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

# # # num_epochs = 50
# # # train_losses_history = [] #

# # # # 3. Training Loop
# # # model.train() 
# # # for epoch in range(num_epochs):
# # #     running_loss = 0.0
# # #     for batch_idx, (xm_batch, ym_batch) in enumerate(train_loader):
# # #         xm_batch = xm_batch.to(DEVICE)
        
 
# # #         optimizer.zero_grad()
# # #         xo1, xo2 = model(xm_batch)
        

# # #         dummy_target = xm_batch[:, 1, :].to(DEVICE) 
# # #         loss = criterion(xo1, dummy_target) 
# # #         loss.backward()
# # #         optimizer.step()

# # #         running_loss += loss.item()

# # #     avg_epoch_loss = running_loss / len(train_loader)
# # #     train_losses_history.append(avg_epoch_loss)
    
# # #     print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_epoch_loss:.4f}")

# # # print("Training complete.")

# # # import matplotlib.pyplot as plt

# # # plt.figure(figsize=(8, 5))
# # # epochs_range = range(1, num_epochs + 1)
# # # plt.plot(epochs_range, train_losses_history, label='Training Loss', color='blue', marker='o')

# # # plt.title('Training Loss Over Epochs')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Loss')
# # # plt.xticks(epochs_range)
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()





# # import torch
# # import torch.nn.functional as F
# # import itertools
# # import h5py
# # import numpy as np
# # from dpft_model import DPFTSeparator 
# # from dpft_model import ORIGINAL_LENGTH
# # import matplotlib.pyplot as plt
# # from torch.utils.data import TensorDataset, DataLoader
# # import dpft_model

# # def sdsdr_loss(true_signal, estimated_signal):
# #     if true_signal.dim() == 0: true_signal = true_signal.unsqueeze(0)
# #     if estimated_signal.dim() == 0: estimated_signal = estimated_signal.unsqueeze(0)
# #     if true_signal.dim() == 1: true_signal = true_signal.unsqueeze(0)
# #     if estimated_signal.dim() == 1: estimated_signal = estimated_signal.unsqueeze(0)
# #     s = true_signal.view(-1)
# #     s_hat = estimated_signal.view(-1)
# #     s_norm_sq = torch.sum(s * s)
# #     s_s_hat_dot = torch.sum(s * s_hat)
# #     if s_norm_sq < 1e-8:
# #         s_target = torch.zeros_like(s)
# #     else:
# #         scale = s_s_hat_dot / s_norm_sq
# #         s_target = scale * s
# #     s_error = s_hat - s_target
# #     s_target_norm_sq = torch.sum(s_target * s_target)
# #     s_error_norm_sq = torch.sum(s_error * s_error)
# #     epsilon = 1e-8
# #     sdsdr = 10 * torch.log10(s_target_norm_sq / (s_error_norm_sq + epsilon) + epsilon)
# #     return sdsdr

# # def uPIT_SDSDR_Loss(true_signals, estimated_signals):
# #     if true_signals.dim() == 2:
# #         true_signals = true_signals.unsqueeze(0)
# #     s_hat = torch.stack(estimated_signals, dim=1) 
# #     batch_size, num_channels, _ = true_signals.shape
# #     sd_sdr_matrix = torch.zeros(batch_size, num_channels, num_channels, device=true_signals.device)
# #     for i in range(num_channels):
# #         for j in range(num_channels):
# #             s_i = true_signals[:, i, :].reshape(batch_size, -1)
# #             s_hat_j = s_hat[:, j, :].reshape(batch_size, -1)
# #             sdsdr_batch = []
# #             for b in range(batch_size):
# #                 sdsdr_val = sdsdr_loss(s_i[b], s_hat_j[b])
# #                 sdsdr_batch.append(sdsdr_val)
# #             sd_sdr_matrix[:, i, j] = torch.stack(sdsdr_batch)
# #     permutations = list(itertools.permutations(range(num_channels)))
# #     max_avg_sdsdr = []
# #     for b in range(batch_size):
# #         sdsdr_per_perm = []
# #         for perm in permutations:
# #             sdsr_sum = 0
# #             for i in range(num_channels):
# #                 sdsr_sum += sd_sdr_matrix[b, i, perm[i]]
# #             avg_sdsdr = sdsr_sum / num_channels
# #             sdsdr_per_perm.append(avg_sdsdr)
# #         max_avg_sdsdr.append(torch.max(torch.stack(sdsdr_per_perm)))
# #     final_loss = -torch.mean(torch.stack(max_avg_sdsdr))
# #     return final_loss

# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("Using device:", DEVICE)

# # h5_path = r"/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"

# # # h5_path = r"C:\Users\Dev\Desktop\s\btp\GOLD_XYZ_OSC.0001_1024.hdf5"
# # try:
# #     with h5py.File(h5_path, "r") as f:
# #         print("Keys in HDF5 file:", list(f.keys()))
# #         X_all = f['X'][:]
# #         Y_all = f['Y'][:]
# # except FileNotFoundError:
# #     print(f"Error: HDF5 file not found at {h5_path}. Using dummy data.")
# #     num_samples, num_mix_channels, L = 100, 2, 1024
# #     num_sources = 2
# #     X_all = np.random.randn(num_samples, num_mix_channels, L).astype(np.float32)
# #     Y_all = np.random.randn(num_samples, num_sources, L).astype(np.float32)

# # num_samples = 30000
# # if num_samples > X_all.shape[0]: num_samples = X_all.shape[0]
# # X_mixture = X_all[:num_samples, 0, :].reshape(num_samples, 1, -1)
# # Y_sources = Y_all[:num_samples]
# # if Y_sources.ndim == 2:
# #     print("WARNING: Y_sources is 2D. Assuming shape (N, L) and adding a placeholder Channel dimension.")
# #     Y_sources = Y_sources[:, np.newaxis, :]
# # if Y_sources.ndim == 3 and Y_sources.shape[1] == 1 and Y_sources.shape[0] > 1:
# #     print("WARNING: Y_sources has C=1, but C=2 is required. Duplicating source dimension.")
# #     Y_sources = np.concatenate([Y_sources, Y_sources], axis=1)
# # B, C, L_current = Y_sources.shape
# # if L_current != ORIGINAL_LENGTH:
# #     print(f"WARNING: Resizing Y data (Ground Truth) from length {L_current} to {ORIGINAL_LENGTH}")
# #     Y_sources_tensor = torch.from_numpy(Y_sources).to(torch.float32)
# #     pad_amount = ORIGINAL_LENGTH - L_current
# #     if pad_amount > 0:
# #         Y_sources_tensor = F.pad(Y_sources_tensor, (0, pad_amount))
# #     elif pad_amount < 0:
# #         Y_sources_tensor = Y_sources_tensor[..., :ORIGINAL_LENGTH]
# #     Y_sources = Y_sources_tensor.numpy()

# # train_split = int(0.8 * num_samples)
# # val_split = int(0.9 * num_samples)
# # X_train = X_mixture[:train_split]
# # Y_train = Y_sources[:train_split]
# # X_val = X_mixture[train_split:val_split]
# # Y_val = Y_sources[train_split:val_split]
# # X_test = X_mixture[val_split:]
# # Y_test = Y_sources[val_split:]
# # print(f"Dataset Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).contiguous()
# # Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
# # X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).contiguous()
# # Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
# # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).contiguous()
# # Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# # train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# # val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
# # test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# # BATCH_SIZE = 4
# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # N_FFT = 512
# # HOP_SIZE = 256
# # def compute_stft_custom(xm, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=None):
# #     if win_length is None: win_length = n_fft
# #     window = torch.hann_window(win_length, device=xm.device)
# #     if xm.dim() > 2: xm = xm.squeeze(-1)
# #     Xm = torch.stft(xm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
# #                     window=window, return_complex=True, normalized=False)
# #     return Xm

# # dpft_model.compute_stft = compute_stft_custom
# # model = DPFTSeparator()
# # model.to(DEVICE)
# # criterion = uPIT_SDSDR_Loss
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# # num_epochs = 50
# # train_losses_history = [] 
# # val_losses_history = []
# # best_val_loss = float('inf')

# # print("\nStarting Training with uPIT-SDSDR Loss...")
# # for epoch in range(num_epochs):
# #     model.train() 
# #     running_loss = 0.0
# #     current_lr = 1e-4 * (0.90 ** epoch)
# #     for param_group in optimizer.param_groups:
# #         param_group['lr'] = current_lr
# #     for batch_idx, (xm_batch, ym_batch) in enumerate(train_loader):
# #         xm_batch = xm_batch.to(DEVICE)
# #         ym_batch = ym_batch.to(DEVICE)
# #         optimizer.zero_grad()
# #         xo1, xo2 = model(xm_batch)
# #         xo1_safe = torch.atleast_2d(xo1)
# #         xo2_safe = torch.atleast_2d(xo2)
# #         loss = criterion(ym_batch, [xo1_safe, xo2_safe])
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item()
# #     avg_train_loss = running_loss / len(train_loader)
# #     train_losses_history.append(avg_train_loss)
# #     model.eval()
# #     val_loss = 0.0
# #     with torch.no_grad():
# #         for xm_batch, ym_batch in val_loader:
# #             xm_batch = xm_batch.to(DEVICE)
# #             ym_batch = ym_batch.to(DEVICE)
# #             xo1, xo2 = model(xm_batch)
# #             xo1_safe = torch.atleast_2d(xo1)
# #             xo2_safe = torch.atleast_2d(xo2)
# #             loss = criterion(ym_batch, [xo1_safe, xo2_safe])
# #             val_loss += loss.item()
# #     avg_val_loss = val_loss / len(val_loader)
# #     val_losses_history.append(avg_val_loss)
# #     print(f"Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f} dB, Val Loss: {avg_val_loss:.4f} dB")
# #     if avg_val_loss < best_val_loss:
# #         best_val_loss = avg_val_loss
# #         torch.save(model.state_dict(), 'best_dpft_model.pth')
# #         print("Model saved!")
# # print("Training complete.")

# # print("\nStarting Model Testing...")
# # try:
# #     model.load_state_dict(torch.load('best_dpft_model.pth'))
# # except FileNotFoundError:
# #     print("Warning: Best model weights not found. Using final epoch weights.")
# # model.eval()
# # test_loss = 0.0
# # total_sdsdr = 0.0
# # with torch.no_grad():
# #     for xm_batch, ym_batch in test_loader:
# #         xm_batch = xm_batch.to(DEVICE)
# #         ym_batch = ym_batch.to(DEVICE)
# #         xo1, xo2 = model(xm_batch)
# #         xo1_safe = torch.atleast_2d(xo1)
# #         xo2_safe = torch.atleast_2d(xo2)
# #         loss = criterion(ym_batch, [xo1_safe, xo2_safe])
# #         test_loss += loss.item()
# #         total_sdsdr += -loss.item() * xm_batch.shape[0]
# # avg_test_loss = test_loss / len(test_loader)
# # avg_test_sdsdr = total_sdsdr / X_test.shape[0]
# # print(f"\n--- Test Results ---")
# # print(f"Test Loss (Negative Avg SD-SDR): {avg_test_loss:.4f} dB")
# # print(f"Average Max SD-SDR on Test Set: {avg_test_sdsdr:.4f} dB")

# # plt.figure(figsize=(10, 6))
# # epochs_range = range(1, num_epochs + 1)
# # plt.plot(epochs_range, train_losses_history, label='Training Loss (-Avg SD-SDR)', color='blue')
# # plt.plot(epochs_range, val_losses_history, label='Validation Loss (-Avg SD-SDR)', color='red')
# # plt.title('Learning Curve (Negative Average SD-SDR)')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss (dB)')
# # plt.legend()
# # plt.grid(True)


# # plt.savefig("training_curve.png", dpi=300, bbox_inches='tight')


# # plt.show()




# import torch
# import torch.nn.functional as F
# import itertools
# import h5py
# import numpy as np
# from dpft_model import DPFTSeparator 
# from dpft_model import ORIGINAL_LENGTH
# import matplotlib.pyplot as plt
# from torch.utils.data import TensorDataset, DataLoader
# import dpft_model

# # --- Loss Functions (SDSDR and uPIT-SDSDR are retained as they match the paper) ---
# def sdsdr_loss(true_signal, estimated_signal):
#     if true_signal.dim() == 0: true_signal = true_signal.unsqueeze(0)
#     if estimated_signal.dim() == 0: estimated_signal = estimated_signal.unsqueeze(0)
#     if true_signal.dim() == 1: true_signal = true_signal.unsqueeze(0)
#     if estimated_signal.dim() == 1: estimated_signal = estimated_signal.unsqueeze(0)
#     s = true_signal.view(-1)
#     s_hat = estimated_signal.view(-1)
    
#     # Target signal projection (scaled version of true signal)
#     s_norm_sq = torch.sum(s * s)
#     s_s_hat_dot = torch.sum(s * s_hat)
    
#     # Scale calculation per Eq. 20 (Note: The paper's formulation is slightly different but this is the standard scale-dependent SDR implementation)
#     if s_norm_sq < 1e-8:
#         s_target = torch.zeros_like(s)
#     else:
#         scale = s_s_hat_dot / s_norm_sq
#         s_target = scale * s
        
#     s_error = s_hat - s_target
#     s_target_norm_sq = torch.sum(s_target * s_target)
#     s_error_norm_sq = torch.sum(s_error * s_error)
#     epsilon = 1e-8
    
#     # SD-SDR calculation
#     sdsdr = 10 * torch.log10(s_target_norm_sq / (s_error_norm_sq + epsilon) + epsilon)
#     return sdsdr

# def uPIT_SDSDR_Loss(true_signals, estimated_signals):
#     """Calculates uPIT loss based on SD-SDR (Eq. 21)"""
#     if true_signals.dim() == 2:
#         true_signals = true_signals.unsqueeze(0)
#     s_hat = torch.stack(estimated_signals, dim=1) 
#     batch_size, num_channels, _ = true_signals.shape
    
#     # 1. Compute SD-SDR matrix (true vs. estimated)
#     sd_sdr_matrix = torch.zeros(batch_size, num_channels, num_channels, device=true_signals.device)
#     for i in range(num_channels): # True source index
#         for j in range(num_channels): # Estimated source index
#             s_i = true_signals[:, i, :].reshape(batch_size, -1)
#             s_hat_j = s_hat[:, j, :].reshape(batch_size, -1)
#             sdsdr_batch = []
#             for b in range(batch_size):
#                 # Ensure sdsdr_loss operates on 1D tensor per batch item
#                 sdsdr_val = sdsdr_loss(s_i[b].flatten(), s_hat_j[b].flatten()) 
#                 sdsdr_batch.append(sdsdr_val)
#             sd_sdr_matrix[:, i, j] = torch.stack(sdsdr_batch)
            
#     # 2. Find best permutation for each batch item (Maximizes average SD-SDR)
#     permutations = list(itertools.permutations(range(num_channels)))
#     max_avg_sdsdr = []
    
#     for b in range(batch_size):
#         sdsdr_per_perm = []
#         for perm in permutations:
#             # Sum of SD-SDRs for the current permutation
#             sdsr_sum = 0
#             for i in range(num_channels):
#                 sdsr_sum += sd_sdr_matrix[b, i, perm[i]]
            
#             # Average SD-SDR for the current permutation
#             avg_sdsdr = sdsr_sum / num_channels
#             sdsdr_per_perm.append(avg_sdsdr)
            
#         # Select the max average SD-SDR over all permutations
#         max_avg_sdsdr.append(torch.max(torch.stack(sdsdr_per_perm)))
        
#     # 3. Final loss is the negative mean of the maximum average SD-SDRs
#     final_loss = -torch.mean(torch.stack(max_avg_sdsdr))
#     return final_loss

# # # --- Data Generation and Utility Functions ---
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("Using device:", DEVICE)

# # h5_path = r"/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"

# # try:
# #     with h5py.File(h5_path, "r") as f:
# #         print("Keys in HDF5 file:", list(f.keys()))
# #         X_all_np = f['X'][:]
# # except FileNotFoundError:
# #     print(f"Error: HDF5 file not found at {h5_path}. Cannot proceed without real data.")
# #     exit()

# # # Extract the Real (I) component, as the paper typically works with real signals or I/Q components separately.
# # # The DeepSig data has shape (N_frames, 1024, 2) [I, Q]
# # I_all_np = X_all_np[:, :, 0].astype(np.float32) # Shape: (N_frames, 1024)
# # num_total_frames = I_all_np.shape[0]

# # # --- PARAMETERS FOR DATA GENERATION ---
# # NUM_MIXTURES = 120
# # if NUM_MIXTURES > num_total_frames // 2: 
# #     NUM_MIXTURES = num_total_frames // 2

# # def generate_mixtures(I_all, N_mixtures, target_length):
# #     """Generates synthetic BSS dataset (Mixture and Sources) from individual signals."""
# #     N_frames, L_original = I_all.shape
    
# #     # Initialize arrays for the mixture (X_M) and two sources (Y_GT)
# #     X_M = np.zeros((N_mixtures, 1, target_length), dtype=np.float32) # (N, 1, L_target)
# #     Y_GT = np.zeros((N_mixtures, 2, target_length), dtype=np.float32) # (N, 2, L_target)
    
# #     for i in range(N_mixtures):
# #         # Randomly select two distinct indices for source 1 and source 2
# #         idx1, idx2 = np.random.choice(N_frames, size=2, replace=False)
# #         S1 = I_all[idx1] 
# #         S2 = I_all[idx2]
        
# #         # --- Data Preprocessing as suggested by the paper (Section III.C, steps 3-6) ---
        
# #         # 3. Scale signals (Paper: -10 dBm to -80 dBm, modeled here by relative scaling and amplitude normalization)
# #         s1_amp = np.random.uniform(0.1, 1.0)
# #         s2_amp = np.random.uniform(0.1, 1.0)
# #         S1_scaled = S1 * s1_amp
# #         S2_scaled = S2 * s2_amp
        
# #         # 4. Corrupt them by adding Gaussian noise (Paper: various levels)
# #         noise_level = np.random.uniform(0.01, 0.1) # Example noise level
# #         noise = np.random.normal(0, noise_level, L_original).astype(np.float32)
        
# #         # 5. Mix the signals by adding them together (X_M = S1_scaled + S2_scaled + Noise)
# #         X_M_i = S1_scaled + S2_scaled + noise 
        
# #         # 6. Normalize the mixture and ground truth signals by dividing their amplitudes by the maximum amplitude of the input mixture.
# #         max_amp = np.max(np.abs(X_M_i))
# #         if max_amp > 1e-8:
# #             X_M_i /= max_amp
# #             S1_norm = S1_scaled / max_amp
# #             S2_norm = S2_scaled / max_amp
# #         else:
# #             S1_norm = S1_scaled
# #             S2_norm = S2_scaled

# #         # Cut out chunks of size ORIGINAL_LENGTH (Paper: step 2)
# #         L_final = min(L_original, target_length)
        
# #         # Ensure correct shape (1, L_final) for X_M and (2, L_final) for Y_GT
# #         X_M[i, 0, :L_final] = X_M_i[:L_final]
# #         Y_GT[i, 0, :L_final] = S1_norm[:L_final]
# #         Y_GT[i, 1, :L_final] = S2_norm[:L_final]
        
# #     return X_M, Y_GT

# # print(f"Generating {NUM_MIXTURES} synthetic BSS samples...")
# # X_mixture, Y_sources = generate_mixtures(I_all_np, NUM_MIXTURES, ORIGINAL_LENGTH)
# # num_samples = X_mixture.shape[0]

# # # --- Splitting and Tensor Conversion ---
# # train_split = int(0.8 * num_samples)
# # val_split = int(0.9 * num_samples)

# # X_train = X_mixture[:train_split]
# # Y_train = Y_sources[:train_split]
# # X_val = X_mixture[train_split:val_split]
# # Y_val = Y_sources[train_split:val_split]
# # X_test = X_mixture[val_split:]
# # Y_test = Y_sources[val_split:]

# # print(f"Dataset Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# # # 🚨 CORRECTION: Removed the unnecessary .permute(0, 2, 1) to keep the shape (B, C=1, L)
# # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# # Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
# # X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# # Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
# # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# # Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# # train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# # val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
# # test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# # BATCH_SIZE = 4
# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# # --- IMPORTS AND SETUP (No Change) ---
# import torch
# import torch.nn.functional as F
# import itertools
# import h5py
# import numpy as np
# from dpft_model import DPFTSeparator 
# from dpft_model import ORIGINAL_LENGTH
# import matplotlib.pyplot as plt
# from torch.utils.data import TensorDataset, DataLoader
# import dpft_model

# # ... (sdsdr_loss and uPIT_SDSDR_Loss functions remain unchanged) ...

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", DEVICE)

# h5_path = r"/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"

# # --- REVISED DATA LOADING AND FILTERING ---
# try:
#     with h5py.File(h5_path, "r") as f:  
#         print("Keys in HDF5 file:", list(f.keys()))
#         X_all_np = f['X'][:]
#         # Load SNR information (Z)
#         # Z_all_np = f['Z'][:] 
# except FileNotFoundError:
#     print(f"Error: HDF5 file not found at {h5_path}. Cannot proceed without real data.")
#     exit()

# # Extract the Real (I) component: Shape (N_frames, 1024)
# I_all_np = X_all_np[:, :, 0].astype(np.float32) 

# # Convert one-hot SNR (Z) to actual SNR value
# # Find the index of the '1' in the one-hot array (or simply the maximum index for simplicity)
# # Assuming Z_all_np shape is (N_frames, 26) where 26 is the number of SNR levels (-20 to +30 in 2dB steps)
# # snr_indices = np.argmax(Z_all_np, axis=1)
# # snr_values = -20 + snr_indices * 2 # Map index 0 to -20dB, index 25 to +30dB

# # 🚨 Action: Filter for High-SNR Frames (SNR >= 0 dB)
# MIN_SNR = 0
# # high_snr_indices = np.where(snr_values >= MIN_SNR)[0]
# I_high_snr = I_all_np
# num_high_snr_frames = I_high_snr.shape[0]

# print(f"Total frames: {I_all_np.shape[0]}. High SNR ({MIN_SNR}dB+) frames selected: {num_high_snr_frames}")
# # -----------------------------------------------

# # --- PARAMETERS FOR DATA GENERATION ---
# NUM_MIXTURES = 12000
# # 🚨 Action: Define Concatenation Factor for Data Length
# FRAMES_PER_SAMPLE = 64 # 64 * 1024 = 65,536 samples, close to 65,280
# MIN_FRAMES_REQUIRED = FRAMES_PER_SAMPLE * 2 # Need enough frames for two sources

# if NUM_MIXTURES > num_high_snr_frames // MIN_FRAMES_REQUIRED: 
#     NUM_MIXTURES = num_high_snr_frames // MIN_FRAMES_REQUIRED
# print(f"Number of mixtures to generate: {NUM_MIXTURES}")

# def generate_mixtures(I_high_snr, N_mixtures, target_length):
#     """Generates synthetic BSS dataset using concatenated high-SNR frames."""
#     N_frames_total, L_frame = I_high_snr.shape # L_frame = 1024
    
#     # Initialize arrays
#     X_M = np.zeros((N_mixtures, 1, target_length), dtype=np.float32)
#     Y_GT = np.zeros((N_mixtures, 2, target_length), dtype=np.float32) 
    
#     current_frame_index = 0
    
#     for i in range(N_mixtures):
#         # 1. Select and Concatenate Frames for S1 and S2
        
#         # Ensure we don't run out of frames, wrap around if necessary (less ideal but prevents crash)
#         if current_frame_index + FRAMES_PER_SAMPLE * 2 >= N_frames_total:
#              current_frame_index = 0
#              np.random.shuffle(I_high_snr) # Reshuffle data pool
             
#         # Select 64 consecutive frames for Source 1
#         start_idx_s1 = current_frame_index
#         end_idx_s1 = start_idx_s1 + FRAMES_PER_SAMPLE
#         S1_frames = I_high_snr[start_idx_s1:end_idx_s1]
#         S1 = S1_frames.flatten()[:target_length] # Concatenate and truncate/pad if necessary
        
#         # Select 64 consecutive frames for Source 2 (must be distinct from S1's frames)
#         start_idx_s2 = end_idx_s1
#         end_idx_s2 = start_idx_s2 + FRAMES_PER_SAMPLE
#         S2_frames = I_high_snr[start_idx_s2:end_idx_s2]
#         S2 = S2_frames.flatten()[:target_length] # Concatenate and truncate/pad if necessary
        
#         # Move index pointer for next iteration
#         current_frame_index = end_idx_s2

#         # --- Data Preprocessing as suggested by the paper (Section III.C, steps 3-6) ---
        
#         # 3. Scale signals 
#         s1_amp = np.random.uniform(0.1, 1.0)
#         s2_amp = np.random.uniform(0.1, 1.0)
#         S1_scaled = S1 * s1_amp
#         S2_scaled = S2 * s2_amp
        
#         # 4. Corrupt them by adding Gaussian noise
#         noise_level = np.random.uniform(0.001, 0.01) # Reduced noise level to focus on separation
#         noise = np.random.normal(0, noise_level, target_length).astype(np.float32)
        
#         # 5. Mix the signals
#         X_M_i = S1_scaled + S2_scaled + noise 
        
#         # 6. Normalize the mixture and ground truth
#         max_amp = np.max(np.abs(X_M_i))
#         if max_amp > 1e-8:
#             X_M_i /= max_amp
#             S1_norm = S1_scaled / max_amp
#             S2_norm = S2_scaled / max_amp
#         else:
#             S1_norm = S1_scaled
#             S2_norm = S2_scaled

#         # Store data
#         X_M[i, 0, :] = X_M_i
#         Y_GT[i, 0, :] = S1_norm
#         Y_GT[i, 1, :] = S2_norm
        
#     return X_M, Y_GT

# print(f"Generating {NUM_MIXTURES} synthetic BSS samples by concatenating {FRAMES_PER_SAMPLE} frames.")
# # 🚨 Action: Pass the filtered data set I_high_snr
# X_mixture, Y_sources = generate_mixtures(I_high_snr, NUM_MIXTURES, ORIGINAL_LENGTH)
# num_samples = X_mixture.shape[0]

# # --- SPLITTING AND TENSOR CONVERSION (Need to re-define tensors without .permute) ---

# train_split = int(0.8 * num_samples)
# val_split = int(0.9 * num_samples)

# X_train = X_mixture[:train_split]
# Y_train = Y_sources[:train_split]
# X_val = X_mixture[train_split:val_split]
# Y_val = Y_sources[train_split:val_split]
# X_test = X_mixture[val_split:]
# Y_test = Y_sources[val_split:]

# print(f"Dataset Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# # 🚨 CORRECTION: Ensure shape is (B, C, L)
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
# test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# BATCH_SIZE = 4
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # ... (rest of the code, including plotting functions and training loop, remains unchanged) ...




# # --- Plotting Function (NEW) ---
# N_FFT = 512
# HOP_SIZE = 256
# FS = 50e6 # Sample rate, assumed from paper's section III.B (50 MHz)

# def compute_spectrogram(x, n_fft=N_FFT, hop_length=HOP_SIZE, fs=FS):
#     """Computes and returns the magnitude spectrogram and frequency/time axis."""
#     # x must be 1D (time)
#     if x.dim() > 1: x = x.squeeze()
    
#     window = torch.hann_window(n_fft, device=x.device)
#     # Ensure signal is long enough for STFT
#     if x.shape[-1] < n_fft:
#         pad_amount = n_fft - x.shape[-1]
#         x = F.pad(x, (0, pad_amount)) 
        
#     Xm = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
#                     window=window, return_complex=True, normalized=False)
    
#     # Use only positive frequency bins, convert to magnitude, then log-scale (dB)
#     S_mag = torch.abs(Xm)
#     S_log = 10 * torch.log10(torch.clamp(S_mag, min=1e-10))
    
#     # Create axes for plotting
#     freq_bins = S_log.shape[0]
#     time_bins = S_log.shape[1]
    
#     freq_axis = np.linspace(0, fs / 2, freq_bins) / 1e6 # MHz
#     time_axis = np.linspace(0, (time_bins - 1) * hop_length / fs, time_bins) * 1e3 # ms
    
#     return S_log.cpu().numpy(), freq_axis, time_axis

# def plot_spectrograms(xm, s1, s2, xo1, xo2, sdsdr_val):
#     """Plots the mixture, ground truth signals, and predicted separated signals as Log-Spectrograms."""
    
#     # The paper plots 4 plots: Mixture, S1, S2, P1, P2. We'll plot 5.
#     fig, axes = plt.subplots(3, 2, figsize=(12, 12))
#     plt.suptitle(f'Blind Source Separation Results\nAvg Max SD-SDR: {sdsdr_val:.4f} dB', fontsize=16)
    
#     signals = {
#         'Log-Spectrogram mixture': xm,
#         'Log-Spectrogram signal 1 (GT)': s1,
#         'Log-Spectrogram prediction 1': xo1,
#         'Log-Spectrogram signal 2 (GT)': s2,
#         'Log-Spectrogram prediction 2': xo2,
#     }

#     # Axes mapping: [0,0] Mixture, [0,1] S1, [1,0] P1, [1,1] S2, [2,0] P2
#     # We will adjust the order to match the paper's visualization structure (Mixture top, then GT/Pred pairs)
#     plot_order = [
#         ('Log-Spectrogram mixture', axes[0, 0]),
#         ('Log-Spectrogram signal 1 (GT)', axes[1, 0]),
#         ('Log-Spectrogram prediction 1', axes[1, 1]),
#         ('Log-Spectrogram signal 2 (GT)', axes[2, 0]),
#         ('Log-Spectrogram prediction 2', axes[2, 1]),
#     ]
    
#     # Remove the redundant 6th subplot
#     fig.delaxes(axes[0, 1]) 
    
#     # Process and plot Mixture
#     S_log_mix, freq_axis, time_axis = compute_spectrogram(signals['Log-Spectrogram mixture'])
#     S_log_mix = S_log_mix[:N_FFT//2 + 1, :] # Use positive frequency bins
    
#     ax = axes[0, 0]
#     im = ax.imshow(S_log_mix, aspect='auto', origin='lower',
#                    extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()],
#                    cmap='viridis')
#     ax.set_title(plot_order[0][0])
#     ax.set_xlabel('time (ms)')
#     ax.set_ylabel('freq (MHz)')
#     # Add colorbar for mixture (top-right of the plot)
#     cbar_mix = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
#     cbar_mix.set_label('dB')
    
    
#     # Process and plot Sources/Predictions
#     for i in range(1, len(plot_order)):
#         title, ax = plot_order[i]
#         signal = signals[title]
#         S_log, freq_axis, time_axis = compute_spectrogram(signal)
#         S_log = S_log[:N_FFT//2 + 1, :]

#         im = ax.imshow(S_log, aspect='auto', origin='lower',
#                        extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()],
#                        cmap='viridis')
#         ax.set_title(title)
#         ax.set_xlabel('time (ms)')
#         ax.set_ylabel('freq (MHz)')
        
#         cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
#         cbar.set_label('dB')
        
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig("separation_spectrograms.png", dpi=300)
#     plt.show()

# # --- Model Instantiation and Training Setup ---
# # The logic for compute_stft_custom is only used for overriding the function 
# # in the dpft_model file, ensuring the correct N_FFT and HOP_SIZE are used there.
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

# dpft_model.compute_stft = compute_stft_custom # Override the STFT in the model file

# model = DPFTSeparator()
# model.to(DEVICE)
# criterion = uPIT_SDSDR_Loss
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# num_epochs = 40
# train_losses_history = [] 
# val_losses_history = []
# best_val_loss = float('inf')

# # --- Training Loop (Retained with minor fixes) ---
# print("\nStarting Training with uPIT-SDSDR Loss...")
# for epoch in range(num_epochs):
#     model.train() 
#     running_loss = 0.0
    
#     # Decaying learning rate schedule (Eq. 22)
#     current_lr = 1e-4 * (0.90 ** epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = current_lr
        
#     for batch_idx, (xm_batch, ym_batch) in enumerate(train_loader):
#         xm_batch = xm_batch.to(DEVICE) # Input mixture (B, 1, L)
#         ym_batch = ym_batch.to(DEVICE) # Ground truth sources (B, 2, L)
        
#         optimizer.zero_grad()
        
#         xo1, xo2 = model(xm_batch) 
        
#         # Ensure outputs are (B, L) for stacking in the loss function
#         xo1_safe = xo1.view(xm_batch.shape[0], -1) 
#         xo2_safe = xo2.view(xm_batch.shape[0], -1)
        
#         loss = criterion(ym_batch, [xo1_safe, xo2_safe])
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#     avg_train_loss = running_loss / len(train_loader)
#     train_losses_history.append(avg_train_loss)
    
#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for xm_batch, ym_batch in val_loader:
#             xm_batch = xm_batch.to(DEVICE)
#             ym_batch = ym_batch.to(DEVICE)
            
#             xo1, xo2 = model(xm_batch)
#             xo1_safe = xo1.view(xm_batch.shape[0], -1)
#             xo2_safe = xo2.view(xm_batch.shape[0], -1)
            
#             loss = criterion(ym_batch, [xo1_safe, xo2_safe])
#             val_loss += loss.item()
            
#     avg_val_loss = val_loss / len(val_loader)
#     val_losses_history.append(avg_val_loss)
    
#     print(f"Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f} dB, Val Loss: {avg_val_loss:.4f} dB")
    
#     # Save best model
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(model.state_dict(), 'best_dpft_model.pth')
#         print("Model saved!")

# print("Training complete.")

# # --- Testing and Plotting ---
# print("\nStarting Model Testing...")
# try:
#     model.load_state_dict(torch.load('best_dpft_model.pth'))
# except FileNotFoundError:
#     print("Warning: Best model weights not found. Using final epoch weights.")
# model.eval()
# test_loss = 0.0
# total_sdsdr = 0.0
# sample_for_plotting = None
# sdsdr_for_plotting = 0.0

# with torch.no_grad():
#     for xm_batch, ym_batch in test_loader:
#         xm_batch = xm_batch.to(DEVICE)
#         ym_batch = ym_batch.to(DEVICE)
        
#         xo1, xo2 = model(xm_batch)
        
#         xo1_safe = xo1.view(xm_batch.shape[0], -1)
#         xo2_safe = xo2.view(xm_batch.shape[0], -1)
        
#         loss = criterion(ym_batch, [xo1_safe, xo2_safe])
#         test_loss += loss.item()
#         total_sdsdr += -loss.item() * xm_batch.shape[0] # Convert loss back to positive SD-SDR
        
#         # Capture the first sample for plotting
#         if sample_for_plotting is None:
#             # xm: (1, L), ym: (2, L), xo: (1, L)
#             sample_for_plotting = (
#                 xm_batch[0, 0].cpu(), 
#                 ym_batch[0, 0].cpu(), 
#                 ym_batch[0, 1].cpu(), 
#                 xo1_safe[0].cpu(), 
#                 xo2_safe[0].cpu()
#             )
#             # Calculate actual Max SD-SDR for this specific sample for the plot title
#             single_sample_loss = -criterion(ym_batch[0].unsqueeze(0), [xo1_safe[0].unsqueeze(0), xo2_safe[0].unsqueeze(0)]).item()
#             sdsdr_for_plotting = single_sample_loss

# avg_test_loss = test_loss / len(test_loader)
# avg_test_sdsdr = total_sdsdr / X_test.shape[0]
# print(f"\n--- Test Results ---")
# print(f"Test Loss (Negative Avg SD-SDR): {avg_test_loss:.4f} dB")
# print(f"Average Max SD-SDR on Test Set: {avg_test_sdsdr:.4f} dB")

# # --- Plotting the Learning Curve ---
# plt.figure(figsize=(10, 6))
# epochs_range = range(1, num_epochs + 1)
# plt.plot(epochs_range, train_losses_history, label='Training Loss (-Avg SD-SDR)', color='blue')
# plt.plot(epochs_range, val_losses_history, label='Validation Loss (-Avg SD-SDR)', color='red')
# plt.title('Learning Curve (Negative Average SD-SDR) [cite: 256]')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (dB)')
# plt.legend()
# plt.grid(True)
# plt.savefig("training_curve.png", dpi=300, bbox_inches='tight')
# plt.show()

# # --- Plotting the Separation Results ---
# if sample_for_plotting is not None:
#     xm, s1, s2, xo1, xo2 = sample_for_plotting
#     plot_spectrograms(xm, s1, s2, xo1, xo2, sdsdr_for_plotting)


import torch
import torch.nn.functional as F
import itertools
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset 
from dpft_model import DPFTSeparator 
from dpft_model import ORIGINAL_LENGTH
import dpft_model

# Model capacity is high (N=256, I=4, J=4) - ASSUMING these are set in dpft_model.py
FRAMES_PER_SAMPLE = 64 # 64 * 1024 = 65,536 samples (Close to ORIGINAL_LENGTH)
MAX_MIXTURES = 30 # Target max number of unique mixture samples
BATCH_SIZE = 1# Keep this small (4-8) due to the large sequence length
NUM_EPOCHS = 2 # Increased to allow for convergence

# --- STFT/PLOTTING PARAMETERS ---
N_FFT = 512
HOP_SIZE = 256
FS = 50e6 # Sample rate, assumed from paper's section III.B

# --- LOSS FUNCTIONS (SD-SDR and uPIT-SDSDR) ---

def sdsdr_loss(true_signal, estimated_signal):
    # SD-SDR calculation (Standard Scale-Dependent SDR)
    if true_signal.dim() == 0: true_signal = true_signal.unsqueeze(0)
    if estimated_signal.dim() == 0: estimated_signal = estimated_signal.unsqueeze(0)
    if true_signal.dim() == 1: true_signal = true_signal.unsqueeze(0)
    if estimated_signal.dim() == 1: estimated_signal = estimated_signal.unsqueeze(0)
    s = true_signal.view(-1)
    s_hat = estimated_signal.view(-1)
    
    s_norm_sq = torch.sum(s * s)
    s_s_hat_dot = torch.sum(s * s_hat)
    
    if s_norm_sq < 1e-8:
        s_target = torch.zeros_like(s)
    else:
        scale = s_s_hat_dot / s_norm_sq
        s_target = scale * s
        
    s_error = s_hat - s_target
    s_target_norm_sq = torch.sum(s_target * s_target)
    s_error_norm_sq = torch.sum(s_error * s_error)
    epsilon = 1e-8
    
    sdsdr = 10 * torch.log10(s_target_norm_sq / (s_error_norm_sq + epsilon) + epsilon)
    return sdsdr

def uPIT_SDSDR_Loss(true_signals, estimated_signals):
    """Calculates uPIT loss based on SD-SDR (Eq. 21)"""
    if true_signals.dim() == 2:
        true_signals = true_signals.unsqueeze(0)
    s_hat = torch.stack(estimated_signals, dim=1) 
    batch_size, num_channels, _ = true_signals.shape
    
    sd_sdr_matrix = torch.zeros(batch_size, num_channels, num_channels, device=true_signals.device)
    for i in range(num_channels):
        for j in range(num_channels):
            s_i = true_signals[:, i, :].reshape(batch_size, -1)
            s_hat_j = s_hat[:, j, :].reshape(batch_size, -1)
            sdsdr_batch = []
            for b in range(batch_size):
                sdsdr_val = sdsdr_loss(s_i[b].flatten(), s_hat_j[b].flatten()) 
                sdsdr_batch.append(sdsdr_val)
            sd_sdr_matrix[:, i, j] = torch.stack(sdsdr_batch)
            
    permutations = list(itertools.permutations(range(num_channels)))
    max_avg_sdsdr = []
    
    for b in range(batch_size):
        sdsdr_per_perm = []
        for perm in permutations:
            sdsr_sum = 0
            for i in range(num_channels):
                sdsr_sum += sd_sdr_matrix[b, i, perm[i]]
            avg_sdsdr = sdsr_sum / num_channels
            sdsdr_per_perm.append(avg_sdsdr)
            
        max_avg_sdsdr.append(torch.max(torch.stack(sdsdr_per_perm)))
        
    final_loss = -torch.mean(torch.stack(max_avg_sdsdr))
    return final_loss

# --- CUSTOM DATASET FOR ON-THE-FLY MIXING (MEMORY FIX) ---

class BSSGenerationDataset(Dataset):
    def __init__(self, I_all_frames, num_mixtures, target_length, frames_per_sample):
        self.I_all_frames = I_all_frames # All 1024-sample frames
        self.num_mixtures = num_mixtures
        self.target_length = target_length
        self.frames_per_sample = frames_per_sample
        self.num_total_frames = I_all_frames.shape[0]

        # Pre-generate random indices for each mixture to ensure fixed data split
        # This requires 2 * frames_per_sample frames per mixture
        self.frame_start_indices = np.random.randint(
            0, self.num_total_frames - self.frames_per_sample, size=(num_mixtures, 2)
        )
        
    def __len__(self):
        return self.num_mixtures

    def __getitem__(self, idx):
        # Retrieve pre-generated indices
        start_idx1 = self.frame_start_indices[idx, 0]
        start_idx2 = self.frame_start_indices[idx, 1]
        
        # 1. Concatenate Frames for S1 and S2
        S1_frames = self.I_all_frames[start_idx1 : start_idx1 + self.frames_per_sample]
        S2_frames = self.I_all_frames[start_idx2 : start_idx2 + self.frames_per_sample]
        
        # Flatten and truncate/pad to target_length (65280)
        S1 = S1_frames.flatten()[:self.target_length]
        S2 = S2_frames.flatten()[:self.target_length]
        
        # --- Data Preprocessing (Scaling, Noise, Normalization) ---
        
        # 3. Scale signals 
        s1_amp = np.random.uniform(0.1, 1.0)
        s2_amp = np.random.uniform(0.1, 1.0)
        S1_scaled = S1 * s1_amp
        S2_scaled = S2 * s2_amp
        
        # 4. Corrupt with Gaussian noise
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level, self.target_length).astype(np.float32)
        
        # 5. Mix the signals
        X_M_i = S1_scaled + S2_scaled + noise 
        
        # 6. Normalize
        max_amp = np.max(np.abs(X_M_i))
        if max_amp > 1e-8:
            X_M_i /= max_amp
            S1_norm = S1_scaled / max_amp
            S2_norm = S2_scaled / max_amp
        else:
            S1_norm = S1_scaled
            S2_norm = S2_scaled

        # Convert to PyTorch tensors and ensure (C, L) shape
        X_M_tensor = torch.tensor(X_M_i, dtype=torch.float32).unsqueeze(0) # (1, L)
        Y_GT_tensor = torch.stack([
            torch.tensor(S1_norm, dtype=torch.float32), 
            torch.tensor(S2_norm, dtype=torch.float32)
        ], dim=0) # (2, L)
        
        return X_M_tensor, Y_GT_tensor

# --- PLOTTING UTILITIES ---

def compute_spectrogram(x, n_fft=N_FFT, hop_length=HOP_SIZE, fs=FS):
    """Computes and returns the magnitude spectrogram and frequency/time axis."""
    if x.dim() > 1: x = x.squeeze()
    
    window = torch.hann_window(n_fft, device=x.device)
    if x.shape[-1] < n_fft:
        pad_amount = n_fft - x.shape[-1]
        x = F.pad(x, (0, pad_amount)) 
        
    Xm = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                    window=window, return_complex=True, normalized=False)
    
    S_mag = torch.abs(Xm)
    S_log = 10 * torch.log10(torch.clamp(S_mag, min=1e-10))
    
    freq_bins = S_log.shape[0]
    time_bins = S_log.shape[1]
    
    # Frequency axis (MHz) and Time axis (ms)
    freq_axis = np.linspace(0, fs / 2, freq_bins) / 1e6 
    time_axis = np.linspace(0, (time_bins - 1) * hop_length / fs, time_bins) * 1e3 
    
    return S_log.cpu().numpy(), freq_axis, time_axis

def plot_spectrograms(xm, s1, s2, xo1, xo2, sdsdr_val):
    """Plots the mixture, ground truth signals, and predicted separated signals as Log-Spectrograms."""
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    plt.suptitle(f'Blind Source Separation Results\nAvg Max SD-SDR: {sdsdr_val:.4f} dB', fontsize=16)
    
    signals = {
        'Log-Spectrogram mixture': xm,
        'Log-Spectrogram signal 1 (GT)': s1,
        'Log-Spectrogram prediction 1': xo1,
        'Log-Spectrogram signal 2 (GT)': s2,
        'Log-Spectrogram prediction 2': xo2,
    }

    plot_order = [
        ('Log-Spectrogram mixture', axes[0, 0]),
        ('Log-Spectrogram signal 1 (GT)', axes[1, 0]),
        ('Log-Spectrogram prediction 1', axes[1, 1]),
        ('Log-Spectrogram signal 2 (GT)', axes[2, 0]),
        ('Log-Spectrogram prediction 2', axes[2, 1]),
    ]
    
    fig.delaxes(axes[0, 1]) 
    
    # Plot Mixture
    S_log_mix, freq_axis, time_axis = compute_spectrogram(signals['Log-Spectrogram mixture'])
    S_log_mix = S_log_mix[:N_FFT//2 + 1, :]
    ax = axes[0, 0]
    im = ax.imshow(S_log_mix, aspect='auto', origin='lower',
                   extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()],
                   cmap='viridis')
    ax.set_title(plot_order[0][0])
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('freq (MHz)')
    cbar_mix = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar_mix.set_label('dB')
    
    # Plot Sources/Predictions
    for i in range(1, len(plot_order)):
        title, ax = plot_order[i]
        signal = signals[title]
        S_log, freq_axis, time_axis = compute_spectrogram(signal)
        S_log = S_log[:N_FFT//2 + 1, :]

        im = ax.imshow(S_log, aspect='auto', origin='lower',
                       extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()],
                       cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('freq (MHz)')
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('dB')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("separation_spectrograms.png", dpi=300)
    plt.show()

# --- MAIN EXECUTION BLOCK ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

h5_path = r"/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5"

try:
    with h5py.File(h5_path, "r") as f:
        print("Keys in HDF5 file:", list(f.keys()))
        X_all_np = f['X'][:]
    I_all_frames = X_all_np[:, :, 0].astype(np.float32) 
    num_total_frames = I_all_frames.shape[0]

except FileNotFoundError:
    print(f"Error: HDF5 file not found at {h5_path}. Cannot proceed without real data.")
    exit()

# Filter is removed (using all frames) to ensure maximum data volume
I_high_snr = I_all_frames 

max_feasible_mixtures = num_total_frames // (2 * FRAMES_PER_SAMPLE)
NUM_MIXTURES = min(MAX_MIXTURES, max_feasible_mixtures)

print(f"Total frames: {num_total_frames}. High SNR (0dB+) frames selected: {num_total_frames}")
print(f"Number of mixtures to generate: {NUM_MIXTURES}")
print(f"Generating {NUM_MIXTURES} synthetic BSS samples by concatenating {FRAMES_PER_SAMPLE} frames.")

# --- DATASET AND LOADER SETUP ---

# Instantiate the full dataset
full_dataset = BSSGenerationDataset(
    I_all_frames=I_high_snr,
    num_mixtures=NUM_MIXTURES,
    target_length=ORIGINAL_LENGTH,
    frames_per_sample=FRAMES_PER_SAMPLE
)

all_indices = np.arange(NUM_MIXTURES)
np.random.shuffle(all_indices)

# Define splits
train_split = int(0.8 * NUM_MIXTURES)
val_split = int(0.9 * NUM_MIXTURES)

train_indices = all_indices[:train_split]
val_indices = all_indices[train_split:val_split]
test_indices = all_indices[val_split:]

print(f"Dataset Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

# Use Subset for splitting
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# DataLoaders with multiprocessing (num_workers) for speed
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

# --- MODEL AND TRAINING SETUP ---

def compute_stft_custom(xm, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=None):
    if win_length is None: win_length = n_fft
    window = torch.hann_window(win_length, device=xm.device)
    if xm.dim() > 2: xm = xm.squeeze(1) 
    
    if xm.shape[-1] < n_fft:
        pad_amount = n_fft - xm.shape[-1]
        xm = F.pad(xm, (0, pad_amount)) 
        
    Xm = torch.stft(xm, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    window=window, return_complex=True, normalized=False)
    return Xm

dpft_model.compute_stft = compute_stft_custom

model = DPFTSeparator()
model.to(DEVICE)
criterion = uPIT_SDSDR_Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = NUM_EPOCHS
train_losses_history = [] 
val_losses_history = []
best_val_loss = float('inf')

# --- TRAINING LOOP ---
print("\nStarting Training with uPIT-SDSDR Loss...")
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    
    # Decaying learning rate schedule (Eq. 22)
    current_lr = 1e-4 * (0.90 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
        
    for batch_idx, (xm_batch, ym_batch) in enumerate(train_loader):
        xm_batch = xm_batch.to(DEVICE) 
        ym_batch = ym_batch.to(DEVICE) 
        
        optimizer.zero_grad()
        
        xo1, xo2 = model(xm_batch) 
        
        xo1_safe = xo1.view(xm_batch.shape[0], -1) 
        xo2_safe = xo2.view(xm_batch.shape[0], -1)
        
        loss = criterion(ym_batch, [xo1_safe, xo2_safe])
        
        # 🚨 Memory Optimization: Use torch.autocast/scaler if using mixed precision (optional but helpful)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses_history.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xm_batch, ym_batch in val_loader:
            xm_batch = xm_batch.to(DEVICE)
            ym_batch = ym_batch.to(DEVICE)
            
            xo1, xo2 = model(xm_batch)
            xo1_safe = xo1.view(xm_batch.shape[0], -1)
            xo2_safe = xo2.view(xm_batch.shape[0], -1)
            
            loss = criterion(ym_batch, [xo1_safe, xo2_safe])
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_losses_history.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.2e}, Train Loss: {avg_train_loss:.4f} dB, Val Loss: {avg_val_loss:.4f} dB")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_dpft_model.pth')
        print("Model saved!")

print("Training complete.")

# --- TESTING AND PLOTTING ---
print("\nStarting Model Testing...")
try:
    # 🚨 Memory Safety: Ensure the model loads to the correct device
    model.load_state_dict(torch.load('best_dpft_model.pth', map_location=DEVICE)) 
except FileNotFoundError:
    print("Warning: Best model weights not found. Using final epoch weights.")
model.eval()
test_loss = 0.0
total_sdsdr = 0.0
sample_for_plotting = None
sdsdr_for_plotting = 0.0

with torch.no_grad():
    for xm_batch, ym_batch in test_loader:
        xm_batch = xm_batch.to(DEVICE)
        ym_batch = ym_batch.to(DEVICE)
        
        xo1, xo2 = model(xm_batch)
        
        xo1_safe = xo1.view(xm_batch.shape[0], -1)
        xo2_safe = xo2.view(xm_batch.shape[0], -1)
        
        loss = criterion(ym_batch, [xo1_safe, xo2_safe])
        test_loss += loss.item()
        total_sdsdr += -loss.item() * xm_batch.shape[0] 
        
        # Capture the first sample for plotting
        if sample_for_plotting is None:
            sample_for_plotting = (
                xm_batch[0, 0].cpu(), 
                ym_batch[0, 0].cpu(), 
                ym_batch[0, 1].cpu(), 
                xo1_safe[0].cpu(), 
                xo2_safe[0].cpu()
            )
            single_sample_loss = -criterion(ym_batch[0].unsqueeze(0), [xo1_safe[0].unsqueeze(0), xo2_safe[0].unsqueeze(0)]).item()
            sdsdr_for_plotting = single_sample_loss

avg_test_loss = test_loss / len(test_loader)
avg_test_sdsdr = total_sdsdr / len(test_indices)
print(f"\n--- Test Results ---")
print(f"Test Loss (Negative Avg SD-SDR): {avg_test_loss:.4f} dB")
print(f"Average Max SD-SDR on Test Set: {avg_test_sdsdr:.4f} dB")

# --- Plotting ---
plt.figure(figsize=(10, 6))
epochs_range = range(1, num_epochs + 1)
plt.plot(epochs_range, train_losses_history, label='Training Loss (-Avg SD-SDR)', color='blue')
plt.plot(epochs_range, val_losses_history, label='Validation Loss (-Avg SD-SDR)', color='red')
plt.title('Learning Curve (Negative Average SD-SDR) [cite: 256]')
plt.xlabel('Epoch')
plt.ylabel('Loss (dB)')
plt.legend()
plt.grid(True)
plt.savefig("training_curve.png", dpi=300, bbox_inches='tight')
plt.show()

if sample_for_plotting is not None:
    xm, s1, s2, xo1, xo2 = sample_for_plotting
    plot_spectrograms(xm, s1, s2, xo1, xo2, sdsdr_for_plotting)