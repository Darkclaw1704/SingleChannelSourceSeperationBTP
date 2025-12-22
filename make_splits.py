import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
filepath = '/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5'
# OUTPUT DIRECTORY: Explicitly set to your dev folder
OUTPUT_DIR = '/home/bss/dev/'

TARGET_MODS = ["BPSK", "QPSK"] 
DATA_FRACTION = 0.1 
RANDOM_SEED = 42

MOD_CLASSES = [
    "OOK", "ASK4", "ASK8", "BPSK", "QPSK", "PSK8", "PSK16", "PSK32",
    "APSK16", "APSK32", "APSK64", "APSK128", "QAM16", "QAM32", "QAM64",
    "QAM128", "QAM256", "AM_SSB_WC", "AM_SSB_SC", "AM_DSB_WC", "AM_DSB_SC",
    "FM", "GMSK", "OQPS"
]

print("Loading HDF5...")
f = h5py.File(filepath, 'r')
y_all = f['Y'][:] 
n_total = y_all.shape[0]
indices_all = np.arange(n_total)

# Identify Rows
y_indices_all = np.argmax(y_all, axis=1)
target_class_ids = [MOD_CLASSES.index(m) for m in TARGET_MODS]
mask_mods = np.isin(y_indices_all, target_class_ids)
valid_indices = indices_all[mask_mods]
valid_labels = y_indices_all[mask_mods]

# Subsample
if DATA_FRACTION < 1.0:
    valid_indices, _, valid_labels, _ = train_test_split(
        valid_indices, valid_labels, train_size=DATA_FRACTION,
        random_state=RANDOM_SEED, stratify=valid_labels
    )

# Create Splits
X_train_val_idx, X_test_idx, y_train_val, y_test = train_test_split(
    valid_indices, valid_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=valid_labels
)
X_train_idx, X_val_idx, y_train, y_val = train_test_split(
    X_train_val_idx, y_train_val, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_val
)

# --- SAVE TO DEV FOLDER ---
# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'train_indices.npy'), X_train_idx)
np.save(os.path.join(OUTPUT_DIR, 'val_indices.npy'), X_val_idx)
np.save(os.path.join(OUTPUT_DIR, 'test_indices.npy'), X_test_idx)

print(f"SAVED to {OUTPUT_DIR} -> Train: {len(X_train_idx)}, Val: {len(X_val_idx)}, Test: {len(X_test_idx)}")
f.close()