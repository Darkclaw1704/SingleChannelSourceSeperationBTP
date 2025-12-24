# SingleChannelSourceSeperationBTP

## Steps for running the transformer code


## Steps for running the CTDCRN code

- **first run 'dataSplit.py'**
  - **How it works:** This script filters the original HDF5 dataset for specific modulation types (e.g., BPSK, QPSK) and takes a defined fraction (10%) of the data. It performs a stratified split into Train, Validation, and Test sets to ensure balanced classes.
  - **Mapping & Model Usage:** Instead of duplicating large datasets, it creates an **index mapping** by saving the numerical row indices from the HDF5 file. Both the **CTDCRN** and **Transformer** models load these files to ensure they train and test on the exact same samples, allowing for fair comparison and memory efficiency.
  - **Output Files:** The indices are saved in the **root directory** as:
    - `train_indices.npy`
    - `val_indices.npy`
    - `test_indices.npy`

- **then run 'CTDCRN code.ipynb'**
  - **How it works:** This model implements a **Complex-Valued Temporal Dilated Convolutional Recurrent Network**. It uses specialized layers (Complex Conv1d, Complex LSTM, and Phase-Preserving Norm) to process the Real and Imaginary (I/Q) components of the signal while maintaining their phase relationships. 
  - **Architecture & Logic:** The model uses a shared **CHE encoder** and independent **separation links** for each source. It utilizes **Permutation Invariant Training (PIT)** to solve the source-matching problem and optimizes using a combination of **Complex MSE** and **Negative SI-SNR** loss.
  - **Data Usage:** It automatically loads the `.npy` files from the previous step to identify which HDF5 rows to use for training and evaluation.
  - **Output Files:** All results are saved in a numbered experiment folder within the **`ctdcrn_runs/`** directory (e.g., `exp1/`). This includes:
    - `best.pt` (Trained model weights)
    - `train.log` (Training history and detailed test metrics)
    - `waveform_gt_vs_pred.png` (Visual verification of signal separation)
