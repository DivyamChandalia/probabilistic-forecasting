from utils import Data, plot_losses, calculate_probability, flip_from_probability, save_data, load_data
from model import StockGPT2Model
import torch
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch import nn
from torch.utils.data import DataLoader

fig = plt.figure()
plt.ion()
file_path = os.path.dirname(os.path.abspath(__file__)) + "/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# data.completed_tickers.append("LDO-USD")
data = load_data(file_path)
# data.plot_pair(1, fig)

learning_rate = 0.001


if os.path.exists(file_path + "model.pickle") and os.path.exists(file_path + "optimizer.pickle"):
    with open(file_path + "model.pickle", "rb") as file:
        model = pickle.load(file)
    with open(file_path + "optimizer.pickle", "rb") as file:
        optimizer = pickle.load(file)
else:
    model = StockGPT2Model(num_features = data.num_features, num_tickers = data.num_tickers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
batch_size = 12

data.mode = "train"
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
data.mode = "val"
val_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
data.mode = "test"
test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)


criterion = nn.MSELoss()

train_losses = []
val_losses = []

num_epochs = 1000
k = 10
# Initialize variables to track best validation loss and best model weights
best_val_loss = float('inf')
best_model_weights = None
# torch.autograd.set_detect_anomaly(False)

# Training loop
epoch_tqdm = tqdm(range(num_epochs), unit="epoch", desc="Epochs: ", position=0, leave=False)

for epoch in epoch_tqdm:
    model.train()
    train_loss = 0.0
    p = calculate_probability(k,epoch)

    data.mode = "train"
    len_batch = len(train_loader)

    for batch_idx, (src, sos) in enumerate(train_loader):
        src, sos = src.to(device), sos.to(device)
        optimizer.zero_grad()

        temp_src = src[:,:data.len_input]
        # print(src.shape)
        # Forward pass
        output, past_key_values = model(src = temp_src, sos = sos)
        temp_src = output[:,-1,:].unsqueeze(1)
        # for i in range(data.len_input, data.len_input + data.len_output - 1):
        #     # mask = flip_from_probability(p = p, batch_size = src.shape[0], num_features = data.num_features).to(device)
        #     # next_input = torch.where(mask, src[:,i], output[:,i])
        #     # temp_src = next_input.unsqueeze(1)
        #     temp_output, past_key_values = model(src = temp_src, past_key_values=past_key_values)
        #     temp_src = temp_output.to(device)
        #     output = torch.cat((output,temp_output), dim = 1)

        # Compute loss and backpropagate
        output,past_key_values = model(src = src, sos = sos)
        loss = criterion(output[:,:-1,:].to(device), src)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        epoch_tqdm.set_description(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx}/{len_batch}] - Train Loss: {loss.item():.4f}")
    train_loss /= len(train_loader)

    model.eval()
    data.mode = "val"
    len_batch = len(val_loader)
    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (src, sos) in enumerate(val_loader):
            src, sos = src.to(device), sos.to(device)
            temp_src = src[:,:data.len_input]
            # Forward pass
            output, past_key_values = model(src = temp_src, sos = sos)
            temp_src = output[:,-1,:].unsqueeze(1)
            # for i in range(data.len_input, data.len_input + data.len_output - 1):
            #     temp_output, past_key_values = model(src = temp_src, past_key_values=past_key_values)
            #     temp_src = temp_output
            #     output = torch.cat((output,temp_output), dim = 1)
            #loss
            output, past_key_values = model(src = src, sos = sos)
            loss = criterion(output[:,:-1,:].to(device), src)
            val_loss += loss.item()
            epoch_tqdm.set_description(f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx}/{len_batch}] - Val Loss: {loss.item():.4f}")
        val_loss = val_loss / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model weights if validation loss improves
    if train_loss < best_val_loss:
        best_val_loss = train_loss
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
        with open(file_path + "model.pickle", "wb") as file:
            pickle.dump(model, file)
        with open(file_path + "optimizer.pickle", "wb") as file:
            pickle.dump(optimizer, file)
    epoch_tqdm.set_description(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    plot_losses(train_losses, val_losses, fig)

    # Print progress

# Save the best model weights to a file
torch.save(best_model_weights, file_path + "best_model_weights.pth")