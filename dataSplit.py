import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
filepath = '/home/bss/data/GOLD_XYZ_OSC.0001_1024.hdf5'
TARGET_MODS = ["BPSK", "QPSK"]  # The specific mods you want
DATA_FRACTION = 0.1             # Use only 10% of the available data for these mods
RANDOM_SEED = 42

# 24 Classes as defined in RadioML 2018.01A
MOD_CLASSES = [
    "OOK", "ASK4", "ASK8", "BPSK", "QPSK", "PSK8", "PSK16", "PSK32",
    "APSK16", "APSK32", "APSK64", "APSK128", "QAM16", "QAM32", "QAM64",
    "QAM128", "QAM256", "AM_SSB_WC", "AM_SSB_SC", "AM_DSB_WC", "AM_DSB_SC",
    "FM", "GMSK", "OQPS"
]

# 1. Open File
f = h5py.File(filepath, 'r')
y_all = f['Y'][:]  # One-hot encoded labels (N, 24)
n_total = y_all.shape[0]
indices_all = np.arange(n_total)

# 2. Identify Target Rows
#    Convert one-hot to index (0..23)
y_indices_all = np.argmax(y_all, axis=1)

#    Find which class indices correspond to our target strings
target_class_ids = [MOD_CLASSES.index(m) for m in TARGET_MODS]

#    Create a mask for rows that match our target mods
mask_mods = np.isin(y_indices_all, target_class_ids)
valid_indices = indices_all[mask_mods]
valid_labels = y_indices_all[mask_mods]  # Keep labels for stratification

print(f"Total samples in file: {n_total}")
print(f"Samples matching {TARGET_MODS}: {len(valid_indices)}")

# 3. Subsample (Fraction of Data)
#    We stratify by label to ensure we keep equal ratios of BPSK/QPSK
if DATA_FRACTION < 1.0:
    valid_indices, _, valid_labels, _ = train_test_split(
        valid_indices, valid_labels,
        train_size=DATA_FRACTION,
        random_state=RANDOM_SEED,
        stratify=valid_labels
    )
    print(f"Subsampled to {DATA_FRACTION*100}%: {len(valid_indices)} samples")

# 4. Create Train/Val/Test Split
#    Split Test (20%)
X_train_val_idx, X_test_idx, y_train_val, y_test = train_test_split(
    valid_indices, valid_labels,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=valid_labels
)

#    Split Train/Val (80% / 20% of remaining -> 0.25 split)
X_train_idx, X_val_idx, y_train, y_val = train_test_split(
    X_train_val_idx, y_train_val,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y_train_val
)

# 5. Save
np.save('train_indices.npy', X_train_idx)
np.save('val_indices.npy', X_val_idx)
np.save('test_indices.npy', X_test_idx)

print(f"SAVED -> Train: {len(X_train_idx)}, Val: {len(X_val_idx)}, Test: {len(X_test_idx)}")
f.close()