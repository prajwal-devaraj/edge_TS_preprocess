import os

import torch
import numpy as np
from matplotlib import pyplot as plt

from cnnlstm_autoencoder import CNNLSTMAutoencoder

val_archive = np.load('data/processed/validation_cycles.npz')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)
model.load_state_dict(torch.load('data/models/cnnlstm_autoencoder_op07.pth', map_location=device, weights_only=True))
model.eval()

VALIDATION_PLOTS_FOLDER = 'data/plots/validation'
os.makedirs(VALIDATION_PLOTS_FOLDER, exist_ok=True)

for cycle_name in val_archive.files:
    cycle_numpy = val_archive[cycle_name]
    cycle_tensor = torch.tensor(cycle_numpy, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed_tensor = model(cycle_tensor)

    error_array = torch.mean(torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)).cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(error_array, label='Reconstruction Error', color='blue')
    plt.title(f'Reconstruction Error for {cycle_name}')
    plt.xlabel('Window Index')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(VALIDATION_PLOTS_FOLDER, f'{cycle_name}_MAE.png')
    plt.savefig(save_path, dpi=150)

    plt.close()

print("Validation reconstruction error plots saved to data/plots/validation/")
