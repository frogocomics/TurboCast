import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import rnn_utils
from rnn_utils import WeatherDataset, CNN, Model


def train_model(model, train_loader, val_loader, epochs):
    """
    Train the model and evaluate the model on the validation set for each epoch.
    """

    # Define the Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        total_train_loss = 0

        model.train()
        for i, (weather_input, image_input, target) in enumerate(train_loader):

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = model(weather_input, image_input)

            loss = criterion(outputs, target)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            total_train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch + 1}:', total_train_loss / len(train_loader))
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for i, (weather_input, image_input, target) in enumerate(val_loader):
                outputs = model(weather_input, image_input)
                loss = criterion(outputs, target)

                total_val_loss += loss.item()

            print(f'Epoch {epoch + 1}:', total_val_loss / len(val_loader))
            val_losses.append(total_val_loss / len(val_loader))

    # Return train and val losses to plot learning curve later
    return np.array(train_losses), np.array(val_losses)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaled_train_df, scaled_validation_df, scaled_test_df = rnn_utils.get_scaled_dfs()

    print(scaled_train_df.columns)
    print(len(scaled_train_df.columns))

    # Get all radar tensors onto device
    images_array = np.load('../data/radar/processed/all_images.npy')[24510:]
    image_tensor = torch.tensor(images_array, dtype=torch.float).to(device) / 100  # Scale image just like precipitation

    # Create dataloaders
    print('Creating dataloaders')
    input_hours, target_hours = 6, 2
    exclude_columns = ['UTC_DATE', 'IMAGE_INDEX']
    batch_size = 256

    # Create datasets
    train_dataset = WeatherDataset(scaled_train_df, input_hours, target_hours, image_tensor, exclude_columns)
    val_dataset = WeatherDataset(scaled_validation_df, input_hours, target_hours, image_tensor, exclude_columns)
    test_dataset = WeatherDataset(scaled_test_df, input_hours, target_hours, image_tensor, exclude_columns)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    # Create the CNN and LSTM model
    print('Creating model')
    cnn_model = CNN().to(device)

    # MODEL HYPERPARAMETERS
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    MODEL_TYPE = 'GRU'

    model = Model(cnn_model, HIDDEN_DIM, NUM_LAYERS, MODEL_TYPE).to(device)
    num_epochs = 100

    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, num_epochs)

    # Save training and validation losses for model so can be plotted over time
    np.save('outputs/train_losses.npy', train_losses)
    np.save('outputs/val_losses.npy', val_losses)

    # Save model weights
    torch.save(model.state_dict(), 'outputs/trained_model.pth')
