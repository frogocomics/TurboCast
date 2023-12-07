import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import rnn_utils
from rnn_utils import WeatherDataset, CNN, Model


def get_test_loss(model, dataloader):
    """
    Get the test loss for a model and dataloader.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        criterion = nn.MSELoss()
        total_loss = 0
        for i, (weather_input, image_input, target) in enumerate(dataloader):
            # print(weather_input.shape, image_input.shape, target.shape)
            # Forward pass, backward pass, and optimize
            outputs = model(weather_input, image_input)
            loss = criterion(F.relu(outputs), target)

            total_loss += loss.item()

        return total_loss / len(dataloader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaled_train_df, scaled_validation_df, scaled_test_df = rnn_utils.get_scaled_dfs()

    # Get all radar tensors onto device, if possible.
    images_array = np.load('../data/radar/processed/all_images.npy')[24510:]
    image_tensor = torch.tensor(images_array, dtype=torch.float).to(device) / 100

    # Create dataloaders
    print('Creating dataloaders')
    input_hours, target_hours = 6, 2
    exclude_columns = ['UTC_DATE', 'IMAGE_INDEX']
    batch_size = 256

    # Create model
    # Create the CNN and LSTM model
    cnn_model = CNN().to(device)
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    MODEL_TYPE = 'GRU'
    model = Model(cnn_model, HIDDEN_DIM, NUM_LAYERS, MODEL_TYPE).to(device)

    state = torch.load('outputs/trained_model.pth')
    model.load_state_dict(state)

    # Create dataset and dataloader
    test_dataset = WeatherDataset(scaled_test_df, 6, 2, image_tensor, exclude_columns)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 1. Want to evaluate testing loss for different input/output combinations
    losses = []
    for j in range(2, 16):
        print('Testing for', j)
        test_dataset = WeatherDataset(scaled_test_df, j, 2, image_tensor, exclude_columns)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        losses.append(get_test_loss(model, test_dataloader))

    np.save('outputs/length_losses.npy', np.array(losses))

    # 2. Look inside CNN:
    # Define a function to capture intermediate images
    print('Looking inside CNN')
    intermediate_images = []


    def get_intermediate_images(module, input, output):
        intermediate_output = output.detach()  # Detach to avoid backpropagation
        intermediate_images.append(intermediate_output)


    # Register a forward hook on the desired layer
    layer_to_extract = cnn_model.conv2  # Change this to the desired layer
    hook_handle = layer_to_extract.register_forward_hook(get_intermediate_images)

    # Prepare your input data
    print(image_tensor.shape)
    input_data = image_tensor[70000, :, :]
    input_data = input_data[None, None, :, :]

    # Forward pass through the model
    cnn_model(input_data)
    # Detach the hook
    hook_handle.remove()

    input_data = input_data.detach().cpu().numpy()
    intermediate_images = torch.cat(intermediate_images).detach().cpu().numpy()

    np.save('outputs/image_input.npy', input_data)
    np.save('outputs/cnn_output2.npy', intermediate_images)

    intermediate_images = []

    # Register a forward hook on the desired layer
    layer_to_extract = cnn_model.pool1  # Change this to the desired layer
    hook_handle = layer_to_extract.register_forward_hook(get_intermediate_images)

    # Prepare your input data
    print(image_tensor.shape)
    input_data = image_tensor[70000, :, :]
    input_data = input_data[None, None, :, :]

    # Forward pass through the model
    cnn_model(input_data)
    # Detach the hook
    hook_handle.remove()

    intermediate_images = torch.cat(intermediate_images).detach().cpu().numpy()

    np.save('outputs/cnn_output1.npy', intermediate_images)

    # 3. Get test features and targets and save
    # Saving test inputs and outputs
    print('Saving test inputs and outputs')

    inputs = []
    outputs = []
    targets = []

    test_dataset = WeatherDataset(scaled_test_df, 6, 2, image_tensor, exclude_columns)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():

        for i, (weather_input, image_input, target) in enumerate(test_dataloader):
            # Forward pass, backward pass, and optimize
            output = model(weather_input, image_input)
            inputs.append(weather_input)
            outputs.append(output)
            targets.append(target)

    inputs = torch.cat(inputs).detach().cpu().numpy()
    outputs = torch.cat(outputs).detach().cpu().numpy()
    targets = torch.cat(targets).detach().cpu().numpy()

    np.save('outputs/test_inputs.npy', inputs)
    np.save('outputs/test_outputs.npy', outputs)
    np.save('outputs/test_targets.npy', targets)
