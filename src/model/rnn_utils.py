import os
from datetime import datetime, timedelta

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def pre_split_feature_engineering(df, weather_images):
    def numerical_feat_eng(data):
        # Removing Unobserved Data
        data = data.drop(columns=['PRECIP_AMOUNT_ST CATHARINES A'])

        # Removing Correlated Features
        data = data.drop(columns=['TEMP_EGBERT CS'])
        data = data.drop(columns=['TEMP_TORONTO CITY'])
        data = data.drop(columns=['STATION_PRESSURE_TORONTO CITY'])
        data = data.drop(columns=['STATION_PRESSURE_TORONTO CITY CENTRE'])
        data = data.drop(columns=['STATION_PRESSURE_PORT WELLER (AUT)'])
        data = data.drop(columns=['WIND_X_TORONTO CITY'])
        data = data.drop(columns=['WIND_Y_TORONTO CITY'])

        # Columns for standard scaling, vapor pressure already very low
        numerical_columns = [var for var in data.columns if is_numeric_dtype(data[var]) and
                             "RELATIVE_HUMIDITY" not in var and
                             "STATION_PRESSURE" not in var and
                             'PRECIP_AMOUNT' not in var]

        return data, numerical_columns

    df, numerical_columns = numerical_feat_eng(df)

    # Merging image index if required
    tmp_df = pd.DataFrame(data=weather_images['UTC_DATE'])
    tmp_df['IMAGE_INDEX'] = tmp_df.index

    df = df.merge(tmp_df, on='UTC_DATE')

    return df, numerical_columns


def split_year_into_chunks(year):
    """
    Get validation chunks for a given year, as [[start, end], [start, end], etc.].
    """

    # Define the start date and end date for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59, 59)  # Set end time to the last second of the year

    # Calculate the total number of hours in the year
    total_hours = int((end_date - start_date).total_seconds() / 3600) + 1

    # Calculate the number of hours in each chunk
    hours_per_chunk = total_hours // 8

    # Initialize a list to store the first and last date of each chunk
    chunks = []

    # Split the year into 8 equal chunks rounded to the nearest hour
    for i in range(8):
        # Calculate the start and end date for the current chunk
        chunk_start_date = start_date + timedelta(hours=i * hours_per_chunk)
        chunk_end_date = chunk_start_date + timedelta(hours=hours_per_chunk)

        # Adjust the end date to the nearest hour within the year
        if chunk_end_date > end_date:
            chunk_end_date = end_date

        # Append the first and last date of the chunk to the list
        chunks.append([chunk_start_date, chunk_end_date])

    return chunks


def post_feature_engineering(data, set_type, scaler, pca, kmeans, exclude_columns, numerical_columns):
    """
    Function for additional feature engineering.
    """

    data = data.copy()

    for column in exclude_columns:
        if column in numerical_columns:
            numerical_columns.remove(column)

    precip = []

    for column in data.columns:
        if 'PRECIP_AMOUNT' in column:
            precip.append(data[column].to_numpy())

    precip = np.array(precip).T

    if set_type == 'Training':
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        transformed_precip = pca.fit_transform(precip)
        precip_label = kmeans.fit_predict(transformed_precip)
    else:
        data[numerical_columns] = scaler.transform(data[numerical_columns])
        transformed_precip = pca.transform(precip)
        precip_label = kmeans.predict(transformed_precip)

    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(precip_label, prefix='Cluster')
    one_hot_encoded.index = data.index

    # Add the one-hot encoded columns to the original DataFrame
    data = pd.concat([data, one_hot_encoded], axis=1)

    return data


def get_scaled_dfs():
    """
    Function to load all data and process it into a scaled train set, validation set, and test set.
    """

    # Load all weather stations
    weather_stations = [
        pd.read_csv('../data/weather/processed/' + file, parse_dates=['UTC_DATE'], index_col='UTC_DATE')
        .drop(columns=['Unnamed: 0', 'WIND_SPEED', 'WIND_DIRECTION', 'VAPOR_PRESSURE', 'x', 'y', 'STATION_NAME'])
        for file in os.listdir('../data/weather/processed')
    ]

    station_names = [file[:-4] for file in os.listdir('../data/weather/processed')]

    # Load all weather images
    weather_images = pd.read_csv('../data/radar/processed/image_dates.csv', parse_dates=['UTC_DATE']).iloc[
                     24510:].reset_index(drop=True)

    for i, weather_station in enumerate(weather_stations):
        weather_stations[i] = weather_station.loc[weather_station.index.isin(weather_images['UTC_DATE'])]

    weather_data = pd.DataFrame()

    for i, df in enumerate(weather_stations):

        df = df.copy()

        df['RELATIVE_HUMIDITY'] = df['RELATIVE_HUMIDITY'] / 100    # Normalize between 0 and 1
        df['STATION_PRESSURE'] = df['STATION_PRESSURE'] / 101.325  # Normalize to atm
        df['PRECIP_AMOUNT'] = df['PRECIP_AMOUNT'] / 100            # Arbitrary scale

        for column in df.columns:
            if column != 'UTC_DATE':  # Avoid renaming the UTC_DATE column
                df.rename(columns={column: f'{column}_{station_names[i]}'}, inplace=True)

        if weather_data.empty:
            weather_data = df
        else:
            weather_data = pd.merge(weather_data, df, on='UTC_DATE', how='left')

    weather_data = weather_data.sort_values(by='UTC_DATE').reset_index()

    weather_data_clean, numerical_columns = pre_split_feature_engineering(weather_data, weather_images)

    chunks = split_year_into_chunks(2022)

    for i, year in enumerate(range(2015, 2023)):
        chunks[i][0] = chunks[i][0].replace(year=year)
        chunks[i][1] = chunks[i][1].replace(year=year)
        if i == len(range(2015, 2023)) - 1:
            chunks[i][1] = datetime(2023, 1, 1, 0, 0)

    # Resetting the dataframes
    train_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    test_data = pd.DataFrame()

    val_end_dates = [0]

    # Iterating over the chunks
    for i, chunk in enumerate(chunks):
        start_date = chunk[0]
        index_val_start = weather_data_clean.index[weather_data_clean['UTC_DATE'] == start_date].tolist()

        end_date = chunk[1]
        index_val_end = weather_data_clean.index[weather_data_clean['UTC_DATE'] == end_date].tolist()

        # Check if both start and end indexes are found
        if index_val_start and index_val_end:
            # Training data: Data between the end of the last validation chunk and the start of the current chunk
            temp_train_data = weather_data_clean.loc[val_end_dates[i]:index_val_start[0]]
            train_data = pd.concat([train_data, temp_train_data])

            # Validation data: Data within the current chunk
            temp_val_data = weather_data_clean.loc[index_val_start[0]:index_val_end[0]]
            validation_data = pd.concat([validation_data, temp_val_data])

            # Update val_end_dates for the next iteration
            # Since end index of current chunk will be the start for next training data
            val_end_dates.append(index_val_end[0] + 1)

    # Define test data as data from 2023 onwards
    test_start_index = weather_data_clean.index[weather_data['UTC_DATE'] >= datetime(2023, 1, 1)].tolist()
    if test_start_index:
        test_data = weather_data_clean.loc[test_start_index[0]:]

    # PCA and KMeans instance for precipitation
    random_state = 42
    pca = PCA(n_components=5, random_state=random_state)
    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=random_state)
    data_scaler = StandardScaler()
    exclude_columns = ['UTC_DATE', 'IMAGE_INDEX']

    # Apply standard scaler to some features
    scaled_train_df = post_feature_engineering(train_data, 'Training', data_scaler, pca, kmeans,
                                               exclude_columns, numerical_columns)
    scaled_validation_df = post_feature_engineering(validation_data, 'Validation', data_scaler, pca, kmeans,
                                                    exclude_columns, numerical_columns)
    scaled_test_df = post_feature_engineering(test_data, 'Test', data_scaler, pca, kmeans, exclude_columns,
                                              numerical_columns)

    return scaled_train_df, scaled_validation_df, scaled_test_df


class WeatherDataset(Dataset):
    def __init__(self, df, input_hours, target_hours, image_array, exclude_columns, exclude_zeros=False):
        """
        Initialize the dataset.
        :param df: DataFrame containing the time series data.
        :param input_hours: Number of hours to use as input.
        :param target_hours: Number of hours to use as target.
        :param exclude_columns: Columns to exclude from input features.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data = df.sort_values(by='UTC_DATE')
        self.dates = df['UTC_DATE'].tolist()

        # Extracting input columns
        self.input_columns = [col for col in df.columns if col not in exclude_columns]

        self.input_hours = input_hours
        self.target_hours = target_hours

        self.features = torch.tensor(df[self.input_columns].values.astype(float), dtype=torch.float).to(device)
        self.targets = torch.tensor(df['PRECIP_AMOUNT_TORONTO CITY'].to_numpy(), dtype=torch.float).to(device)

        # Preprocess data to create input-target pairs
        self.exclude_zeros = exclude_zeros
        self.indices = self.preprocess()

        self.image_array = image_array
        self.image_idx = df['IMAGE_INDEX'].to_numpy().astype(int)

        # print(self.image_idx)

    def preprocess(self):
        """
        Preprocess the time series by getting a list of valid indices.
        """
        indices = []

        for start_idx in range(0, len(self.data) - self.input_hours + 1):

            dates = self.dates[start_idx:start_idx + self.input_hours + self.target_hours]

            if dates[0] + timedelta(hours=(self.input_hours + self.target_hours - 1)) == dates[-1]:

                target = self.targets[start_idx:start_idx + self.input_hours + self.target_hours]

                if self.exclude_zeros:
                    if torch.any(target > 0):
                        indices.append(start_idx)
                else:
                    indices.append(start_idx)

        return indices

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get entries from the dataset.
        """
        start_idx = self.indices[idx]
        end_idx1 = start_idx + self.input_hours
        end_idx2 = end_idx1 + self.target_hours

        input_tensor = self.features[start_idx:end_idx1]
        image_tensor = self.image_array[self.image_idx[start_idx:end_idx1]]
        target_tensor = self.targets[end_idx1:end_idx2]
        return input_tensor, image_tensor, target_tensor


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x):
        """
        Forward pass for the CNN.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 4 x 8 x 8 = 256
        return x


class Model(nn.Module):
    def __init__(self, cnn, hidden_dim, num_layers, rnn_type='RNN'):
        super(Model, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.cnn = cnn

        self.name = f'{self.rnn_type}_{self.num_layers}layers'

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.rnn2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.rnn2 = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.fc1 = nn.Linear(66, 32)
        self.bn1 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(288, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc_out1 = nn.Linear(hidden_dim, 8)
        self.fc_out2 = nn.Linear(8, 2)

    def forward(self, time_series_data, images):
        """
        Forward pass for the GRU.
        """

        # Check for NaN in inputs
        assert not torch.isnan(time_series_data).any(), "NaNs in time_series_data"
        assert not torch.isnan(images).any(), "NaNs in images"

        batches = images.size(0)

        combined_outputs = []

        for time_step in range(time_series_data.size(1)):
            hourly_radar = images[:, time_step, :, :]
            hourly_radar = hourly_radar[:, None, :, :]
            cnn_output = self.cnn(hourly_radar)

            hourly_data = time_series_data[:, time_step, :]
            hourly_data = F.relu(self.bn1(self.fc1(hourly_data)))

            combined_data = torch.cat((hourly_data, cnn_output), dim=1)
            combined_data = F.relu(self.bn2(self.fc2(combined_data)))

            assert not torch.isnan(combined_data).any(), "NaNs in combined_data"

            combined_outputs.append(combined_data)

        combined_outputs_tensor = torch.stack(combined_outputs).reshape(batches, -1, self.hidden_dim)
        assert not torch.isnan(combined_outputs_tensor).any(), "NaNs in combined_outputs_tensor"

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batches, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batches, self.hidden_dim).to(self.device)

        # Forward pass through RNN
        if self.rnn_type == 'LSTM':
            output, _ = self.rnn(combined_outputs_tensor, (h0, c0))
        else:
            output, _ = self.rnn(combined_outputs_tensor, h0)

        # Introduce residual connection
        output = torch.add(output, combined_outputs_tensor)

        # Do this again
        # Initialize hidden and cell states
        h0_2 = torch.zeros(self.num_layers, batches, self.hidden_dim).to(self.device)
        c0_2 = torch.zeros(self.num_layers, batches, self.hidden_dim).to(self.device)

        # Forward pass through RNN
        if self.rnn_type == 'LSTM':
            output2, _ = self.rnn2(output, (h0_2, c0_2))
        else:
            output2, _ = self.rnn2(output, h0_2)

        # Introduce residual connection
        output2 = torch.add(output2, output)

        assert not torch.isnan(output2).any(), "NaNs in RNN output"

        last_time_step_output = output2[:, -1, :]

        assert not torch.isnan(last_time_step_output).any(), "NaNs in last_time_step_output"

        out = F.relu(self.fc_out1(last_time_step_output))

        out = self.fc_out2(out)
        assert not torch.isnan(out).any(), "NaNs in final output"

        return out
