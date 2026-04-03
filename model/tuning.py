import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from cnnlstm_autoencoder import CNNLSTMAutoencoder

def model_tuning(tensor_path, save_path, old_model_path='data/models/cnnlstm_autoencoder_op07.pth', visualize_loss=False, epochs=20, batch_size=64, lr=0.0001):
    """Tune the model using existing model and mixed cycle data."""
    data_numpy = np.load(tensor_path)
    x_tensor = torch.tensor(data_numpy, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, x_tensor)
    tune_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)
    model.load_state_dict(torch.load(old_model_path, map_location=device, weights_only=True))

    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in tune_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(tune_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"[*] Tuned model saved to {save_path}")

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


if __name__ == '__main__':
    model_tuning(tensor_path='data/processed/tuning_tensor.npy',
                 save_path='data/models/cnnlstm_autoencoder_op07_v1_1.pth',
                 visualize_loss=True)
    

    
