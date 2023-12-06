import multiprocessing

import numpy as np
import base64
from PIL import Image, UnidentifiedImageError
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import datetime
import re
import pytz
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, TimeoutException
import os
import argparse
import validation_utils


# Color mapping hex color -> int
color_values = [
    '99ccff',
    '0099ff',
    '00ff66',
    '00cc00',
    '009900',
    '006600',
    'ffff33',
    'ffcc00',
    'ff9900',
    'ff6600',
    'ff0099',
    '9933cc',
    '660099'
]

# Areas without precipitation (without a color) will map to "0", which is the default value
mapping_dict = {color_value: i + 1 for i, color_value in enumerate(color_values)}


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Multithreaded web scraping for Weather Canada radar data')

    parser.add_argument(
        '-n',
        '--num_threads',
        type=int,
        default=1,
        help='Number of threads to use (default: 1)',
    )
    parser.add_argument(
        '--start_month',
        type=validation_utils.validate_month,
        required=True,
        help='Start month and year (YYYY-MM)',
    )
    parser.add_argument(
        '--end_month',
        type=validation_utils.validate_month,
        required=True,
        help='End month and year (YYYY-MM)',
    )

    parser.add_argument(
        '--weather',
        type=validation_utils.validate_weather,
        default='rain',
        help='Weather type: "rain" or "snow" (default: rain)'
    )

    return parser.parse_args()


def get_datetime(text):
    """
    Extract the date and time from a string like:

    Composite PRECIPET - Rain - 2023-01-28, 23:00 EST, 20/20

    In this case, function would return 2023-01-28 23:00
    """

    d = re.findall(r'\d{4}-\d{2}-\d{2}', text)[0]  # Regex: Match DDDD-DD-DD where D is a digit
    t = re.findall(r'\d{2}:\d{2}', text)[0]        # Regex: Match DD:DD where D is a digit
    return d + ' ' + t


def convert_to_utc(est_time_str):
    """
    Convert a string in the format YYYY-MM-DD HH:MM which is in EST to a string in the format YYYY-MM-DD which is in
    UTC.
    """
    # Define the input date and time in EST
    est_time = pytz.timezone('US/Eastern').localize(datetime.datetime.strptime(est_time_str, "%Y-%m-%d %H:%M"))

    # Convert time from EST to UTC
    utc_time = est_time.astimezone(pytz.timezone('UTC'))

    # Format the UTC time as a string
    return utc_time.strftime("%Y-%m-%d %H:%M")


def load_boolean_mask(file):
    """
    Load an image as a 2D boolean array.
    """

    return np.array(Image.open(file)) > 0


def get_day_hour_pairs(year, month, hour_increment=2):
    """
    Given a year and a month, get (day, hour) pairs with a two-hour increment.
    """

    # Calculate the number of days in the given month
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year

    num_days = (datetime.date(next_year, next_month, 1) - datetime.date(year, month, 1)).days

    # Generate (day, hour) pairs for each day in the month with the specified hour increment
    day_hour_pairs = []

    for day in range(1, num_days + 1):
        for hour in range(0, 24, hour_increment):
            day_hour_pairs.append((day, hour))

    return day_hour_pairs


def get_rain_url(year, month, day, hour):
    """
    Get the primary (not backup) url for rain data on a given year, month, day, and hour.
    """

    return f'https://climate.weather.gc.ca/radar/index_e.html?site=CASKR&' + \
        f'year={year}&month={month}&day={day}&hour={hour}&minute=0&' + \
        f'duration=2&image_type=COMP_PRECIPET_RAIN_WEATHEROFFICE'


def get_snow_url(year, month, day, hour):
    """
    Get the primary (not backup) url for snow data on a given year, month, day, and hour.
    """

    return f'https://climate.weather.gc.ca/radar/index_e.html?site=CASKR&' + \
        f'year={year}&month={month}&day={day}&hour={hour}&minute=0&' + \
        f'duration=2&image_type=COMP_PRECIPET_SNOW_WEATHEROFFICE'


def process_url(gif_url, source, ref_masks):
    """
    Process the image, which is given as a gif url.

    Example of gif url: data:image/gif;base64,R0lGODlhRALgAfYAA...

    Returns a processed image (2D integer array) and mask (2D boolean array).
    """

    # Unpack reference masks
    mask_labels, mask_rings, mask_precip, mask_precipet = ref_masks

    # Extract the base64-encoded image data
    image_data = gif_url.split(';base64,')[1]

    # Decode the base64 data and load it as an image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = image.crop((1, 1, 479, 479))  # Crop image

    precip = np.zeros((478, 478), dtype=int)

    width, height = image.size
    palette = image.getpalette()
    data = list(image.getdata())

    # If radar source is 'COMP_PRECIPET', need to create red mask
    if source == 'COMP_PRECIPET':
        mask_red = np.zeros((478, 478), dtype=bool)

    # Iterate over the image
    for y in range(height):
        for x in range(width):
            index = data[x * height + y]
            color = palette[index * 3: (index + 1) * 3]  # Extract RGB values from the palette

            # Convert RGB values to hex
            hex_color = '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

            if source == 'COMP_PRECIPET' and not mask_labels[x, y]:  # Labels can be automatically ignored
                # If pixel color is red and pixel location is on one of the rings, assume the ring is obscuring that
                # section of the image.
                if hex_color == 'ff0000' and mask_rings[x, y]:
                    mask_red[x, y] = True
                elif hex_color in mapping_dict:
                    # Otherwise, use mapping to convert color in that pixel to an integer
                    precip[x, y] = mapping_dict[hex_color]
            elif source == 'PRECIP' and not mask_precip[x, y] and hex_color in mapping_dict:
                # Use mapping to convert color in that pixel to an integer
                precip[x, y] = mapping_dict[hex_color]
            elif not mask_precipet[x, y] and hex_color in mapping_dict:  # source == 'PRECIPET'
                # Use mapping to convert color in that pixel to an integer
                precip[x, y] = mapping_dict[hex_color]

    # For 'COMP_PRECIPET' source, need to combine label mask and red mask. Otherwise, masks will always be the same.
    if source == 'COMP_PRECIPET':
        mask = np.logical_or(mask_labels, mask_red)
    elif source == 'PRECIP':
        mask = mask_precip
    else:
        mask = mask_precipet

    return precip, mask


def find_available_source(driver, url):
    """
    Find available sources. Method tries different urls and uses the radar element to check if the radar data is
    available for that source.
    """

    try:
        driver.find_element(By.ID, 'radar')
    except NoSuchElementException:
        url = url.replace('COMP_PRECIPET', 'PRECIP')
        driver.get(url)

        try:
            driver.find_element(By.ID, 'radar')
        except NoSuchElementException:
            url = url.replace('PRECIP', 'PRECIPET')
            driver.get(url)

            try:
                driver.find_element(By.ID, 'radar')
            except NoSuchElementException:
                # No source is available
                return None
            return 'PRECIPET'
        return 'PRECIP'

    return 'COMP_PRECIPET'


def get_all(driver, url, ref_masks):
    """
    Fetch all of the images, dates, and masks for a single url.
    """
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 15)

        # First, check options which are available.
        source = find_available_source(driver, url)

        if source is None:
            # In this case, no sources are available. Return empty output as a result.
            return [], [], []

        # Next image button
        next_image_button = driver.find_element(By.ID, 'nextimage')
        # Go to first image button
        first_image_button = driver.find_element(By.ID, 'firstimage')

        images = []
        masks = []
        dates = []

        # Radar image
        image = driver.find_element(By.ID, 'radar')
        gif_url = image.get_attribute('src')

        # Click and wait for update
        try:
            first_image_button.click()
            wait.until(lambda driver: image.get_attribute('src') != gif_url)
        except ElementNotInteractableException:
            # In this case, the first image button is not clickable. We can safely ignore this.
            pass

        # Get the gif url
        gif_url = driver.find_element(By.ID, 'radar').get_attribute('src')
        i, mask = process_url(gif_url, source, ref_masks)
        images.append(i)
        masks.append(mask)
        dates.append(get_datetime(driver.find_element(By.ID, 'animation-info').text))

        while next_image_button.is_enabled():
            # Click and wait for update
            next_image_button.click()
            wait.until(lambda driver: image.get_attribute('src') != gif_url)

            # Get the gif url
            gif_url = driver.find_element(By.ID, 'radar').get_attribute('src')
            i, mask = process_url(gif_url, source, ref_masks)

            images.append(i)
            masks.append(mask)
            dates.append(get_datetime(driver.find_element(By.ID, 'animation-info').text))

        return images, dates, masks
    except TimeoutException:
        # Do catch all for timeout exception--strange things can happen during web scraping.
        return [], [], []
    except UnidentifiedImageError:
        # This should not occur--but this is raised when there is an error in processing the gif url.
        return [], [], []


def download(year, month, ref_masks, mode='rain'):
    """
    Download all images, with their corresponding date/time and
    inpainting mask in a month.
    """
    day_hour_pairs = get_day_hour_pairs(year, month)

    # Use bundled Chrome installation and Chrome driver
    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = '../../../bin/chrome-win64/chrome.exe'
    driver = webdriver.Chrome(options=chrome_options, executable_path='../../../bin/chromedriver-win64/chromedriver.exe')

    all_images = []
    all_dates = []
    all_masks = []

    # Iterate over each day and hour at a 2-hour increment
    for day, hour in day_hour_pairs:

        # Get the rain/snow url
        if mode == 'rain':
            url = get_rain_url(year, month, day, hour)
        else:
            url = get_snow_url(year, month, day, hour)

        # Web scrape url and append image, date, mask results
        images, dates, masks = get_all(driver, url, ref_masks)

        all_images.extend(images)
        all_dates.extend(dates)
        all_masks.extend(masks)

    # Apply correction to dates

    # Correction 1: Some dates show first day of previous month while correct should
    # be last day of previous month.
    first_day_of_current_month = datetime.date(year, month, 1)
    last_day_of_previous_month = first_day_of_current_month - datetime.timedelta(days=1)
    s = f'{last_day_of_previous_month.year}-{last_day_of_previous_month.month}'

    for i, date_text in enumerate(all_dates):
        if date_text.startswith(s):
            date_text = f'{last_day_of_previous_month} {date_text[-5:]}'
            all_dates[i] = date_text

    # Correction 2: Fix incorrect date on December 31st.
    for i, date_text in enumerate(all_dates):
        if date_text[5:10] == '12-00':
            # Invalid! Should be 12-31
            all_dates[i] = date_text[:5] + '12-31' + date_text[10:]

    # Convert dates from EST to UTC
    # Text format: 2021-12-31 19:40
    all_dates = [convert_to_utc(d) for d in all_dates]
    all_images = np.asarray(all_images)
    all_masks = np.asarray(all_masks)

    folder = f'download/{year}-{month}'

    if not os.path.exists(folder):
        os.mkdir(folder)

    # Create temporary folder for later processing step
    tmp_folder = folder + '/temp'

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    # Define the output file path
    output_file = folder + '/' + f'dates_{mode}.txt'

    with open(output_file, 'w') as file:
        file.write('\n'.join(all_dates))

    # Save images and masks as compressed numpy array
    np.savez_compressed(folder + '/' + f'images_{mode}.npz', all_images)
    np.savez_compressed(folder + '/' + f'masks_{mode}.npz', all_masks)


def run(queue, mode, ref_masks):
    """
    Thread run function.
    """
    pid = os.getpid()
    print(pid, 'running started')

    while True:
        data = queue.get(block=True)

        if data is None:
            print(pid, 'running finished')
            break

        year, month = data

        # Radar scraping month
        print(f'Downloading radar data for {year}-{month}')
        download(year, month, ref_masks, mode=mode)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    n_threads = args.num_threads    # Number of threads
    start_month = args.start_month  # Start month
    end_month = args.end_month      # End month (inclusive)
    mode = args.weather             # Mode (rain OR snow)

    # Load masks
    mask_labels = load_boolean_mask('refmasks/mask_COMP_PRECIPET_labels.png')
    mask_rings = load_boolean_mask('refmasks/mask_COMP_PRECIPET_rings.png')
    mask_precip = load_boolean_mask('refmasks/mask_PRECIP.png')
    mask_precipet = load_boolean_mask('refmasks/mask_PRECIPET.png')

    # Multithreading
    queue = multiprocessing.Queue()
    threads = []

    for i in range(n_threads):
        thread = multiprocessing.Process(target=run,
                                         args=(queue, mode, [mask_labels, mask_rings, mask_precip, mask_precipet]))
        thread.start()

        threads.append(thread)

    # Add (year, month) pairs to queue
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
