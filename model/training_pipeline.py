import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from cnnlstm_autoencoder import CNNLSTMAutoencoder


def model_training(tensor_path, visualize_loss=False):
    """Train the model using tensor data and optionally visualize loss."""
    data_numpy = np.load(tensor_path)
    x_tensor = torch.tensor(data_numpy, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, x_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    os.makedirs('data/models', exist_ok=True)
    model_save_path = 'data/models/cnnlstm_autoencoder_op07.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"[*] Model saved to {model_save_path}")

    if visualize_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, marker='o', color='blue')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return model, loss_history


def visualize_sample_reconstruction(model, tensor_path):
    """
    Randomly pick one window from the dataset and plot Real vs Reconstructed.
    """
    device = next(model.parameters()).device
    model.eval()

    data_numpy = np.load(tensor_path)
    random_idx = np.random.randint(0, data_numpy.shape[0])
    single_window_real = torch.tensor(data_numpy[random_idx], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        single_window_recon = model(single_window_real)

    real_data = single_window_real.cpu().numpy().squeeze()
    recon_data = single_window_recon.cpu().numpy().squeeze()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axis_names = ['X-Axis', 'Y-Axis', 'Z-Axis']

    for i in range(3):
        axes[i].plot(real_data[:, i], label='Real (SymLog)', color='blue', alpha=0.7)
        axes[i].plot(recon_data[:, i], label='Reconstructed', color='red', linestyle='--', alpha=0.9)
        axes[i].set_title(f'{axis_names[i]} Reconstruction (Window Index: {random_idx})')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model_training('data/processed/training_tensor.npy', visualize_loss=False)

    trained_model, _ = model_training('data/processed/training_tensor.npy', visualize_loss=False)

    # Can be run multiple times to see different random samples
    visualize_sample_reconstruction(trained_model, 'data/processed/training_tensor.npy')