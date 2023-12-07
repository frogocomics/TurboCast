import os
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cdist
from math import radians, sin, cos, sqrt, atan2


def haversine(coord1, coord2):
    """
    Haversine distance function, where coord1 and coord2 are (x, y) latitude/longitude coordinates.

    Returns the distance in kilometers.
    """

    radius = 6371.0  # Earth's radius in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Angular distance between latitudes
    dlat = radians(lat2 - lat1)

    # Angular distance between longitudes
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = radius * c
    return distance


def impute(dataframes, distances_array, feature):
    """
    Impute a missing feature using Inverse Distance Weighing (IDW). Dataframes is a list of dataframes, where each
    dataframe corresponds to a weather station. All dataframes must have the same date column. The feature is the column
    name of the feature to impute.
    """

    # Get boolean array indicating which features are missing in which dataframe
    missing = []

    for df in dataframes:
        missing.append(df[feature].isna().to_numpy())

    missing = np.column_stack(missing)

    # Impute row-by-row
    for i in range(len(dataframes[0])):

        # An array corresponding to each dataframe, where True indicates the value is missing for that dataframe,
        # and False indicates the value is available for that dataframe.
        missing_arr = missing[i]

        # If there are no True (missing) values, then can move on...
        if np.sum(missing_arr) > 0:

            # Array of missing indices
            missing_indices = np.where(missing_arr == True)[0]

            # Array of existing indices
            existing_indices = np.where(missing_arr == False)[0]

            # Impute for each missing index
            for missing_idx in missing_indices:
                # Get inverse squared distance between missing weather station and all AVAILABLE weather stations
                weights = distances_array[missing_idx, existing_indices] ** -2

                # Get existing values from all AVAILABLE weather stations
                z = np.array([[dataframes[j].iloc[i][feature]] for j in existing_indices]).flatten()

                # Find imputed value through IDW
                z_imputed = np.sum(weights * z) / np.sum(weights)

                # Add imputed value into dataframe with missing value
                dataframes[missing_idx].at[i, feature] = z_imputed


def replace_outliers_with_nan(weather_station):
    """
    Convenience function to replace precipitation outliers (>100 mm/hr) with NAs, so that
    we can impute it.
    """
    column = weather_station['PRECIP_AMOUNT']
    lower_bound = 0
    upper_bound = 100
    outliers = (column < lower_bound) | (column > upper_bound)
    print(f'{outliers.sum()} precipitation outliers removed')
    weather_station.loc[outliers, 'PRECIP_AMOUNT'] = pd.NA


def vapor_pressure_water(temp):
    """
    Get the saturated vapor pressure of water in kPa, given a temperature.
    """
    # Using Wagner equation
    T = temp + 273.15  # Convert temperature to Kelvin
    A = 7.89750
    B = 3132.2
    C = 61.2
    D = 0.25700

    log_pressure = A - (B / (T + C)) - (D * np.log10(T))
    pressure = 10 ** log_pressure

    # Convert from mm Hg to kPa
    return pressure / 760 * 101.325


if __name__ == '__main__':

    # Get list of csvs in 'download' folder
    csv_list = ['download/' + filename for filename in os.listdir('download') if filename.endswith('.csv')]

    # Get a list of all station names
    station_names = set()

    for filename in csv_list:
        df = pd.read_csv(filename)
        station_names = station_names.union(set(df['Station Name'].unique()))

    station_names = list(station_names)

    # Dict where key = station name, value = station data
    dataframes = {}

    # List of columns to keep
    selected_columns = ['UTC_DATE', 'STATION_NAME', 'x', 'y', 'PRECIP_AMOUNT',
                        'TEMP', 'DEW_POINT_TEMP', 'RELATIVE_HUMIDITY',
                        'STATION_PRESSURE', 'WIND_DIRECTION', 'WIND_SPEED']

    # Data type for each column to keep
    data_types = {
        'Station Name': str,
        'Longitude (x)': float,
        'Latitude (y)': float,
        'Precip. Amount (mm)': float,
        'Temp (°C)': float,
        'Dew Point Temp (°C)': float,
        'Rel Hum (%)': float,
        'Stn Press (kPa)': float,
        'Wind Dir (10s deg)': float,
        'Wind Spd (km/h)': float
    }

    rename = {
        'Station Name': 'STATION_NAME',
        'Longitude (x)': 'x',
        'Latitude (y)': 'y',
        'Precip. Amount (mm)': 'PRECIP_AMOUNT',
        'Temp (°C)': 'TEMP',
        'Dew Point Temp (°C)': 'DEW_POINT_TEMP',
        'Rel Hum (%)': 'RELATIVE_HUMIDITY',
        'Stn Press (kPa)': 'STATION_PRESSURE',
        'Wind Dir (10s deg)': 'WIND_DIRECTION',
        'Wind Spd (km/h)': 'WIND_SPEED',
        'Date/Time (UTC)': 'UTC_DATE'
    }

    for filename in csv_list:
        df = pd.read_csv(filename, dtype=data_types, parse_dates=['Date/Time (UTC)'])  # Parse UTC_DATE as datetime
        df.rename(columns=rename, inplace=True)
        df = df[selected_columns]

        for station_name in station_names:

            # Select entries for only one station
            sub_df = df[df['STATION_NAME'] == station_name]

            # If dict is not empty, concat to existing rows if exists already, create new entry if dict if does not
            # exist.
            if not sub_df.empty:
                if station_name in dataframes:
                    dataframes[station_name] = pd.concat([dataframes[station_name], sub_df], axis=0)
                else:
                    dataframes[station_name] = sub_df

    # First date with enough values available for imputation
    start_date = datetime(2014, 11, 5, 20)
    cutoff_date = datetime(2023, 2, 15, 13)

    for station_name, df in dataframes.items():
        df = df.drop_duplicates(subset=['UTC_DATE'])  # Drop duplicates

        # Filter DF so that UTC date is after the start date and before the cutoff date
        df = df[(df['UTC_DATE'] >= start_date) & (df['UTC_DATE'] <= cutoff_date)]

        # REINDEX:
        # Create a DateTimeIndex with hourly frequency for the desired range
        new_index = pd.date_range(start=start_date, end=cutoff_date, freq='1H')

        # Reindex the DataFrame with the new DateTimeIndex and keep missing values as NaN for 'value' column
        df = df.set_index('UTC_DATE').reindex(new_index).reset_index()
        df = df.rename(columns={'index': 'UTC_DATE'})

        dataframes[station_name] = df

    # Discard keys as keep as list of dataframes
    dataframes = list(dataframes.values())

    # Remove outliers from dataframes
    for dataframe in dataframes:
        replace_outliers_with_nan(dataframe)

    # Get list of (x, y) coordinates for each dataframe in dataframes
    coordinates = [(df['x'].iloc[df['x'].first_valid_index()], df['y'].iloc[df['y'].first_valid_index()]) for df in
                   dataframes]
    coordinates = np.array(coordinates)

    # Get 2D distance array (Haversine distance between any two coordinates)
    distances_array = cdist(coordinates, coordinates, metric=haversine)

    # Create WIND_X and WIND_Y features
    # Reasoning is that WIND_DIRECTION is an angle, and has a discontinuity at 0/360, which the neural network
    # model does not like! Thus, better to represent wind as vector in cartesian form instead of in polar form.
    for df in dataframes:
        angles_rad = np.radians(df['WIND_DIRECTION'] * 10)  # WIND_DIRECTION in 10s of degrees
        df['WIND_X'] = df['WIND_SPEED'] * np.sin(angles_rad)
        df['WIND_Y'] = df['WIND_SPEED'] * np.cos(angles_rad)

    # List of features to impute
    to_impute = ['PRECIP_AMOUNT', 'TEMP', 'DEW_POINT_TEMP', 'RELATIVE_HUMIDITY', 'STATION_PRESSURE', 'WIND_X',
                 'WIND_Y']

    for feature in to_impute:
        print(f'Imputing {feature}')
        impute(dataframes, distances_array, feature)

    for df in dataframes:
        df['WIND_SPEED'] = np.sqrt(df['WIND_X'].to_numpy() ** 2 + df['WIND_Y'].to_numpy() ** 2)

        # Calculate the angle in radians
        angle_rad = np.arctan2(df['WIND_Y'].to_numpy(), df['WIND_X'].to_numpy())
        # Convert radians to degrees
        angle_deg = np.degrees(angle_rad)
        # Adjust angles to be clockwise from north
        angles_from_north = 90 - angle_deg
        angles_from_north[angles_from_north < 0] += 360  # Adjust negative angles

        df['WIND_DIRECTION'] = angles_from_north / 10

        # Create VAPOR_PRESSURE feature
        df['VAPOR_PRESSURE'] = vapor_pressure_water(df['TEMP']) * df['RELATIVE_HUMIDITY'] / 100

        # Save processed files
        station_name = df['STATION_NAME'].iloc[df['STATION_NAME'].first_valid_index()]

        df.to_csv(f'processed/{station_name}.csv')
