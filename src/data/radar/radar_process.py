import numpy as np
import os

import pandas as pd
from scipy.interpolate import CubicSpline
from skimage.restoration import denoise_tv_chambolle
import multiprocessing
import cv2
from scipy.ndimage import median_filter
import validation_utils
import argparse
import datetime

# Define color mapping
rain_x = np.array([0, 1, 3, 5, 7, 9, 11, 13])
rain_y = np.array([0.1, 1, 4, 12, 24, 50, 100, 200])

snow_x = np.array([0, 1, 3, 5, 7, 9, 11, 13])
# Assume 1 cm/hr of snow is equivalent to 1 mm/hr rain: this is a commonly-used rule of thumb.
snow_y = np.array([0.1, 0.2, 0.5, 1, 2, 4, 7.5, 20])

# Create a cubic spline interpolation
rain_spline = CubicSpline(rain_x, rain_y)
snow_spline = CubicSpline(snow_x, snow_y)

arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5])

rain_mapping_values = np.round(rain_spline(arr), 2)
snow_mapping_values = np.round(snow_spline(arr), 2)

# These will be used to convert the integer arrays to float arrays.
rain_mapping_values = np.insert(rain_mapping_values, 0, 0)
snow_mapping_values = np.insert(snow_mapping_values, 0, 0)


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Multithreaded processing for downloaded data")

    parser.add_argument(
        "-n",
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads to use (default: 4)",
    )
    parser.add_argument(
        "--start_month",
        type=validation_utils.validate_month,
        required=True,
        help="Start month and year (YYYY-MM)",
    )
    parser.add_argument(
        "--end_month",
        type=validation_utils.validate_month,
        required=True,
        help="End month and year (YYYY-MM)",
    )

    return parser.parse_args()


def run_1(queue, temp_folder, mode):
    """
    Thread run function for step 1.
    """
    processed_images = []
    dates = []

    pid = os.getpid()

    while True:
        data = queue.get(block=True)

        if data is None:

            # Write processed_images
            np.savez_compressed(f"{temp_folder}/{mode}_{pid}.npz", np.asarray(processed_images))

            # Write dates
            with open(f"{temp_folder}/{mode}_{pid}.txt", 'w') as file:
                file.write('\n'.join(dates))
            del processed_images, dates
            break

        image, mask, date = data

        dates.append(date)

        image_processed = np.zeros((478, 478))

        # Convert int array to float array through predefined mapping
        for x in range(478):
            for y in range(478):
                # Use different (int -> float) mapping for rain vs. snow
                if mode == 'rain':
                    image_processed[x, y] = rain_mapping_values[image[x, y]]
                else:  # mode == 'snow':
                    image_processed[x, y] = snow_mapping_values[image[x, y]]

        # Scale image and mask
        size = 128
        image_processed = cv2.resize(image_processed, dsize=(size, size), interpolation=cv2.INTER_AREA)
        mask_processed = cv2.resize(mask.astype(float), dsize=(size, size), interpolation=cv2.INTER_AREA)

        # Normalize, inpaint, and then unnormalize
        orig_min = np.min(image_processed)
        orig_max = np.max(image_processed)

        image_processed = cv2.normalize(image_processed, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
        mask_processed = (mask_processed * 255).astype(np.uint8)

        inpaint_result = cv2.inpaint(image_processed, mask_processed, inpaintRadius=3, flags=cv2.INPAINT_TELEA).astype(
            float)

        inpaint_result = inpaint_result / 65535.0 * (orig_max - orig_min) + orig_min

        # Perform Total Variation denoising
        inpaint_result = denoise_tv_chambolle(inpaint_result, weight=0.01)  # Very low weight is sufficient!

        # Reduce spikes
        spike_threshold = 1.0
        filtered_data = median_filter(inpaint_result, size=8)

        # Replace spikes with a smoothed interpolation
        spike_mask = np.abs(inpaint_result - filtered_data) > spike_threshold
        inpaint_result[spike_mask] = (inpaint_result + filtered_data)[spike_mask] / 2.0

        # Make sure all values are positive
        inpaint_result[inpaint_result < 0] = 0

        # Append processed image
        processed_images.append(inpaint_result)


def process_folder(folder, mode='rain', n_threads=1):
    """
    Process a folder and return processed images and dates. This implementation is multithreaded.
    """

    # Load compressed array of images
    images = np.load(f'{folder}/images_{mode}.npz')
    images = images.f.arr_0

    # Load compressed array of masks
    masks = np.load(f'{folder}/masks_{mode}.npz')
    masks = masks.f.arr_0

    # Load dates
    with open(f'{folder}/dates_{mode}.txt', 'r') as file:
        dates = [line.strip() for line in file]

    # Multithreading
    queue = multiprocessing.Queue()
    threads = []

    for i in range(n_threads):
        thread = multiprocessing.Process(target=run_1, args=(queue, f'{folder}/temp', mode))
        thread.start()

        threads.append(thread)

    for image, mask, date in zip(images, masks, dates):
        # print(date)
        queue.put((image, mask, date))

    # Thread will terminate once None value is reached
    for i in range(n_threads):
        queue.put(None)

    for thread in threads:
        thread.join()

    # Combine results
    combined_images = []
    combined_dates = []

    for filename in os.scandir(f"{folder}/temp"):
        if filename.is_file() and filename.name.endswith('.npz'):
            # Load npz file
            images = np.load(f'{folder}/temp/{filename.name}')
            images = images.f.arr_0

            combined_images.append(images)

            # Load dates
            with open(f'{folder}/temp/{filename.name[:-4]}.txt', 'r') as file:
                dates = [line.strip() for line in file]

            combined_dates.extend(dates)

    combined_images = np.concatenate(combined_images)

    # Delete temp files
    for filename in os.scandir(f"{folder}/temp"):
        if filename.is_file():
            os.remove(filename.path)

    # Sort by dates since multithreading messes up the order of the dates
    sorted_idx = sorted(range(len(combined_dates)), key=lambda k: combined_dates[k])

    combined_images = combined_images[sorted_idx]
    combined_dates = [combined_dates[i] for i in sorted_idx]

    return combined_images, combined_dates


def multithread_process_month(year, month, mode, n_threads):
    """
    Process a month and save results to disk.
    """
    folder = f'download/{year}-{month}'

    if os.path.exists(folder):
        print('Processing ' + folder + ' [mode=' + mode + ']')

        images, dates = process_folder(folder, mode, n_threads)

        # Save images
        np.savez_compressed(f'{folder}/processed_images_{mode}.npz', images)

        # Save dates
        with open(f'{folder}/processed_dates_{mode}.txt', 'w') as file:
            file.write('\n'.join(dates))
    else:
        print("Skipped " + folder)


def calculate_midpoints(date_list):
    """
    Given a sequential list of dates, get a list of all the midpoints of all the dates. If a list of n dates is
    provided, this will return an array of n -1 midpoints.
    """

    midpoints = []

    for i in range(len(date_list) - 1):
        # Calculate the midpoint between consecutive dates
        midpoint = date_list[i] + (date_list[i + 1] - date_list[i]) / 2
        midpoints.append(midpoint)

    return midpoints


def custom_unique(arr):
    """
    Get the array of unique values and index
    of unique values from a list.
    """

    unique_values = []
    unique_indices = []
    seen = {}

    for idx, value in enumerate(arr):
        if value not in seen:
            seen[value] = idx
            unique_values.append(value)
            unique_indices.append(idx)

    return unique_values, unique_indices


def time_average_images(folder, mode='rain'):
    """
    Get a sequence of hourly averaged images, given a mode.
    """

    # Load array of images
    images = np.load(f'{folder}/processed_images_{mode}.npz')
    images = images.f.arr_0

    # Size of image (dimension x dimension)
    dimension = images.shape[1]

    # Load dates and convert to datetime objects
    with open(f'{folder}/processed_dates_{mode}.txt', 'r') as file:
        dates = [datetime.datetime.strptime(line.strip(), '%Y-%m-%d %H:%M') for line in file]

    # Get a list of unique dates and their corresponding index in the original array.
    dates, date_indices = custom_unique(dates)

    # Create a dict where key = date and value = index of date
    d = {}

    for date, date_index in zip(dates, date_indices):
        d[date] = date_index

    # Initialize
    hours = [dates[0].replace(minute=0)]  # List of all hours featured
    split_dates = [[dates[0]]]  # List of list of all dates which are in a hour corresponding to hours
    midpoints = []

    # Add dates into split_dates based on the hour of the date
    for date in dates[1:]:

        # Get hour of date
        floored_date = date.replace(minute=0)

        # If previous date is within hour
        if hours[-1] == floored_date:
            split_dates[-1].append(date)
        else:  # If date is in new hour
            split_dates.append([date])
            hours.append(floored_date)

    '''
    Add dates from the previous or following hour.
    Note: to avoid data leakage, dates from the
    next hour will be added only if right at the
    hour (minute = 0). This could be thought of
    as minute "60" of the current hour.
    '''

    # Hour array: all dates which are in that hour
    # Hour: the date representing that hour
    for i, (hour_array, hour) in enumerate(zip(split_dates, hours)):

        # First date in the hour
        first_date = hour_array[0]

        start_hour = hour  # MINUTE 0
        end_hour = start_hour + datetime.timedelta(hours=1)  # MINUTE 60
        previous_hour = start_hour - datetime.timedelta(hours=1)  # MINUTE 0 OF PREVIOUS HOUR

        if first_date.minute != 0:

            if i != 0:

                # Add last date from previous hour, if present.
                if hours[i - 1] == previous_hour:
                    proposed_date = split_dates[i - 1][-1]

                    # Check midpoint between proposed_date and first_date is after start_hour
                    midpoint = proposed_date + (first_date - proposed_date) / 2

                    if midpoint > start_hour:
                        hour_array.insert(0, proposed_date)

        # As date with minute 60 will never be added, add the first date from the
        # following hour if and only if minute == 0. If minute == 3, for example,
        # don't add, as that would constitute data leakage!

        if i != len(split_dates) - 1:

            proposed_date = split_dates[i + 1][0]

            if proposed_date == end_hour:
                hour_array.append(proposed_date)

        # Last, calculate midpoints for that hour. All midpoints will be
        # WITHIN that hour.
        midpoints.append(calculate_midpoints(hour_array))

    # Proportion of each date's "influence" within each hour
    proportions = []

    # Hour array: all dates which are in that hour
    # Hour: the date representing that hour
    # Midpoint: the midpoints in that hour
    for i, (hour_array, hour, midpoint) in enumerate(zip(split_dates, hours, midpoints)):

        # Make a copy of hour_array
        modified_hour_array = hour_array.copy()

        # First date in the hour
        first_date = hour_array[0]
        # Last date in the hour
        last_date = hour_array[-1]

        start_hour = hour  # MINUTE 0
        end_hour = start_hour + datetime.timedelta(hours=1)  # MINUTE 60

        minutes = np.zeros(len(hour_array))

        # Take cases:
        if first_date <= start_hour and last_date >= end_hour:
            modified_hour_array[0] = start_hour
            modified_hour_array[-1] = end_hour

            for j in range(len(midpoint)):
                time_diff = midpoint[j] - modified_hour_array[j]
                minutes[j] += time_diff.total_seconds() / 60

                time_diff = modified_hour_array[j + 1] - midpoint[j]
                minutes[j + 1] += time_diff.total_seconds() / 60

        elif first_date <= start_hour and last_date < end_hour:
            modified_hour_array[0] = start_hour

            for j in range(len(midpoint)):
                time_diff = midpoint[j] - modified_hour_array[j]
                minutes[j] += time_diff.total_seconds() / 60

                time_diff = modified_hour_array[j + 1] - midpoint[j]
                minutes[j + 1] += time_diff.total_seconds() / 60

            time_diff = modified_hour_array[0] - start_hour
            minutes[0] += time_diff.total_seconds() / 60

        elif first_date > start_hour and last_date >= end_hour:
            modified_hour_array[-1] = end_hour

            for j in range(len(midpoint)):
                time_diff = midpoint[j] - modified_hour_array[j]
                minutes[j] += time_diff.total_seconds() / 60

                time_diff = modified_hour_array[j + 1] - midpoint[j]
                minutes[j + 1] += time_diff.total_seconds() / 60

            time_diff = end_hour - modified_hour_array[-1]
            minutes[-1] += time_diff.total_seconds() / 60

        else:  # first_date > start_hour and last_date < end_hour:
            for j in range(len(midpoint)):
                time_diff = midpoint[j] - modified_hour_array[j]
                minutes[j] += time_diff.total_seconds() / 60

                time_diff = modified_hour_array[j + 1] - midpoint[j]
                minutes[j + 1] += time_diff.total_seconds() / 60

            time_diff = modified_hour_array[0] - start_hour
            minutes[0] += time_diff.total_seconds() / 60

            time_diff = end_hour - modified_hour_array[-1]
            minutes[-1] += time_diff.total_seconds() / 60

        proportion = minutes / 60
        proportions.append(proportion)

    processed_images = []

    # Fix: if last date's minute == 0, then discard last hour.
    if dates[-1].minute == 0:
        hours = hours[:-1]
        split_dates = split_dates[:-1]
        proportions = proportions[:-1]

    exclude_idx = []

    # proportion: weight of each date in hour
    # hour_array: dates in each hour which should be used
    for i, (proportion, hour_array) in enumerate(zip(proportions, split_dates)):

        if len(proportion) == len(hour_array):

            combined = np.zeros((dimension, dimension))

            for j, date in enumerate(hour_array):
                # Fetch image
                image = images[d[date]]

                combined += proportion[j] * image

            processed_images.append(combined)
        else:
            # Sanity check: this should NOT happen.
            print(processed_images)
            print(hour_array)

            exclude_idx.append(i)

    # Should not be needed, but remove problematic hours.
    return [hours[i] for i in range(len(hours)) if i not in exclude_idx], processed_images


def time_average_and_save(folder, mode='rain'):
    """
    Time average and save images in a folder, for a given mode.
    """
    hours, processed_images = time_average_images(folder, mode)
    # Convert datetime back to string
    hours = [d.strftime("%Y-%m-%d %H") for d in hours]

    # Save images
    np.savez_compressed(folder + '/' + f'averaged_{mode}.npz', processed_images)

    # Save dates
    with open(folder + '/' + f'{mode}_dates_averaged.txt', 'w') as file:
        file.write('\n'.join(hours))


def run_2(queue):
    """
    Thread run function for step 2.
    """
    pid = os.getpid()
    print(pid, 'running started')

    while True:
        data = queue.get(block=True)

        if data is None:
            print(pid, 'running finished')
            break

        year, month = data

        folder = f'download/{year}-{month}'
        print("Processing " + folder)

        # Get hourly averages for both rain and snow images.
        time_average_and_save(folder, 'rain')
        time_average_and_save(folder, 'snow')

        images, dates = combine_rain_snow(folder)

        np.savez_compressed(f'{folder}/combined.npz', images)

        # Write dates
        with open(f'{folder}/combined_dates.txt', 'w') as file:
            file.write('\n'.join([date.strftime('%Y-%m-%d %H') for date in dates]))


def combine_rain_snow(folder):
    """
    Combine hourly-averaged rain and snow images in a folder into a series of composite precipitation images.
    """

    # Load rain dates
    with open(f'{folder}/rain_dates_averaged.txt', 'r') as file:
        rain_dates = [datetime.datetime.strptime(line.strip(), '%Y-%m-%d %H') for line in file]

    # Load rain images
    averaged_rain = np.load(f'{folder}/averaged_rain.npz')
    averaged_rain = averaged_rain.f.arr_0

    # Load snow dates
    with open(f'{folder}/snow_dates_averaged.txt', 'r') as file:
        snow_dates = [datetime.datetime.strptime(line.strip(), '%Y-%m-%d %H') for line in file]

    # Load snow images
    averaged_snow = np.load(f'{folder}/averaged_snow.npz')
    averaged_snow = averaged_snow.f.arr_0

    # Combine and get unique snow and rain images
    combined_dates = rain_dates + snow_dates
    combined_dates, _ = custom_unique(combined_dates)
    combined_dates = sorted(combined_dates)

    combined_results = np.zeros((len(combined_dates), 128, 128))

    # Iterate over each date
    for i, combined_date in enumerate(combined_dates):
        has_rain = combined_date in rain_dates
        has_snow = combined_date in snow_dates

        # Get rain and snow index if available
        if has_rain:
            rain_index = rain_dates.index(combined_date)
        if has_snow:
            snow_index = snow_dates.index(combined_date)

        # Cases: Combine if both rain and snow are available. If only one source is available, then use that source.
        if has_rain and not has_snow:
            combined_results[i] = averaged_rain[rain_index]
        elif not has_rain and has_snow:
            combined_results[i] = averaged_snow[snow_index]
        else:  # If both has_rain and has_snow, take the maximum of the rain and snow input.
            combined_results[i] = np.maximum(averaged_rain[rain_index], averaged_snow[snow_index])

    return combined_results, combined_dates


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    n_threads = args.num_threads    # Number of threads
    start_month = args.start_month  # Start month
    end_month = args.end_month      # End month (inclusive)

    # STEP 1 ######################################################################
    #   Process all downloaded data through resizing, inpainting, and denoising   #
    ###############################################################################

    # Need to process for both rain and snow
    multithread_process_month(start_month.year, start_month.month, 'rain', n_threads)
    multithread_process_month(start_month.year, start_month.month, 'snow', n_threads)

    while start_month < end_month:
        start_month += datetime.timedelta(days=32)  # Move to the next month
        start_month = start_month.replace(day=1)  # Set day to 1 to avoid issues with varying month lengths
        multithread_process_month(start_month.year, start_month.month, 'rain', n_threads)
        multithread_process_month(start_month.year, start_month.month, 'snow', n_threads)

    # STEP 2 ######################################################################
    #   Combine all processed data into hourly average images, and then combine   #
    #   rain and snow images into a composite precipitation image.                #
    ###############################################################################

    queue = multiprocessing.Queue()
    threads = []

    for i in range(n_threads):
        thread = multiprocessing.Process(target=run_2, args=(queue,))
        thread.start()

        threads.append(thread)

    # Reset start month
    start_month = args.start_month
    queue.put((start_month.year, start_month.month))

    while start_month < end_month:
        start_month += datetime.timedelta(days=32)  # Move to the next month
        start_month = start_month.replace(day=1)  # Set day to 1 to avoid issues with varying month lengths
        queue.put((start_month.year, start_month.month))

    # Thread will terminate once None value is reached
    for i in range(n_threads):
        queue.put(None)

    for thread in threads:
        thread.join()

    # STEP 3 ######################################################
    #   Load from each folder in download and combine and save.   #
    ###############################################################

    # Create list of folders
    folders = []

    # Reset start month
    start_month = args.start_month
    folders.append(f'download/{start_month.year}-{start_month.month}')

    while start_month < end_month:
        start_month += datetime.timedelta(days=32)  # Move to the next month
        start_month = start_month.replace(day=1)  # Set day to 1 to avoid issues with varying month lengths
        folders.append(f'download/{start_month.year}-{start_month.month}')

    # Create list of dates and save
    dates = []

    for folder in folders:
        try:
            with open(f'{folder}/combined_dates.txt', 'r') as file:
                for line in file:
                    dates.append(line.strip())
        except:
            # If there are any loading issues, disregard. But this should not happen!
            pass

    # Now, dates list contains each line from the file
    image_dates = pd.DataFrame(dates, columns=['UTC_DATE'])
    image_dates['UTC_DATE'] = pd.to_datetime(image_dates['UTC_DATE'], format='%Y-%m-%d %H')
    image_dates.to_csv('processed/image_dates.csv', index=False)

    # Create array of images and save
    images = []

    for folder in folders:
        try:
            npz_file = np.load(f'{folder}/combined.npz')
            images.extend(npz_file.f.arr_0)  # Append images to the list
        except Exception as e:
            print(f"Error loading file from {folder}: {e}")

    # Convert the list of images to a numpy array
    images_array = np.array(images)
    # Save the numpy array to an .npy file
    np.save('processed/all_images.npy', images_array)
