import argparse
import datetime
import multiprocessing
import os
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

import validation_utils


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Multithreaded web scraping for Weather Canada station data')

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

    return parser.parse_args()


def download_month(driver, year, month, station_id):
    """
    Download weather station data for a given year, month, and station.

    Returns True if successfully downloaded, False otherwise.
    """

    url = (
        'https://climate.weather.gc.ca/climate_data/hourly_data_e.html?'
        'hlyRange=2002-06-04%7C2023-11-29&dlyRange=2002-06-04%7C2023-11-29'
        f'&mlyRange=2003-07-01%7C2006-12-01&StationID={station_id}'
        '&Prov=ON&urlExtension=_e.html&searchType=stnName&optLimit=yearRange&'
        'StartYear=1840&EndYear=2023&selRowPerPage=25&Line=26&searchMethod='
        'contains&txtStationName=TORONTO&timeframe=1&time=LST&time=UTC&'
        f'Year={year}&Month={month}&Day=1#'
    )
    driver.get(url)
    try:
        # Get download button and click
        next_image_button = driver.find_element(By.XPATH, '//input[@value="Download Data"]')
        next_image_button.click()
        return True  # Indicates success
    except NoSuchElementException:  # Catch error and return False to indicate error
        return False


def run(queue):
    """
    Thread run function.
    """
    pid = os.getpid()
    print(pid, 'running started')

    # Use bundled Chrome installation and Chrome driver
    chrome_options = webdriver.ChromeOptions()
    download_path = os.getcwd() + '\\download'
    chrome_options.add_experimental_option('prefs', {'download.default_directory': download_path})

    chrome_options.binary_location = '../../../bin/chrome-win64/chrome.exe'
    driver = webdriver.Chrome(options=chrome_options, executable_path='../../../bin/chromedriver-win64/chromedriver.exe')

    while True:
        data = queue.get(block=True)

        if data is None:
            # Wait for files to finish downloading before closing
            print('Waiting for files to finish downloading')
            time.sleep(10)

            print(pid, 'running finished')
            driver.quit()
            break

        year, month, station_id = data

        # Weather scraping month and station
        response = download_month(driver, year, month, station_id)

        if response:
            print(f'Downloaded {station_id}\tYear {year}\tMonth {month}')
        else:
            print(f'FAILED TO DOWNLOAD {station_id}\tYear {year}\tMonth {month}')


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()

    n_threads = args.num_threads   # Number of threads
    start_month = args.start_month # Start month
    end_month = args.end_month     # End month (inclusive)

    # Mapping of station names to station ids
    station_ids = {
        'BARRIE-ORO': 42183,
        'EGBERT CS': 27604,
        'ELORA RCS': 41983,
        'HAMILTON RBG CS': 27529,
        'OSHAWA': 48649,
        'PETERBOROUGH A': 48952,
        'PORT WELLER (AUT)': 7790,
        'ST CATHARINES A': 53000,
        'TORONTO CITY CENTRE': 48549,
        'TORONTO CITY': 31688
    }

    # Multithreading
    queue = multiprocessing.Queue()
    threads = []

    for i in range(n_threads):
        thread = multiprocessing.Process(target=run,
                                         args=(queue,))
        thread.start()

        threads.append(thread)

    original_set = set()

    # Add (year, month) pairs to queue
    for station_name, station_id in station_ids.items():
        queue.put((start_month.year, start_month.month, station_id))
        original_set.add((start_month.year, start_month.month, station_name))

    while start_month < end_month:
        start_month += datetime.timedelta(days=32)  # Move to the next month
        start_month = start_month.replace(day=1)  # Set day to 1 to avoid issues with varying month lengths

        for station_name, station_id in station_ids.items():
            queue.put((start_month.year, start_month.month, station_id))
            original_set.add((start_month.year, start_month.month, station_name))

    # Thread will terminate once None value is reached
    for i in range(n_threads):
        queue.put(None)

    for thread in threads:
        thread.join()

    # The code sometimes has downloading errors; let's go through the download folder and remove duplicates first.
    for file in os.listdir('download'):
        if file.endswith(').csv'):
            os.remove('download/' + file)

    downloaded_set = set()
    for file in os.listdir('download'):
        if file.endswith('.csv'):
            station_name = pd.read_csv('download/' + file)['Station Name'].iloc[0]
            month, year = file.split('_')[5].split('-')
            month, year = int(month), int(year)
            downloaded_set.add((year, month, station_name))

    missing_set = original_set - downloaded_set

    # If there are any months missing for a weather station, attempt to redownload those months. If the download still
    # fails, this means data for that month is simply unavailable.
    if bool(missing_set):
        print(f'{len(missing_set)} missing entries found')

        # Ensure number of threads is less than the length of missing set,
        # so that all threads have something to do!
        n_threads_reduced = min(len(missing_set), n_threads)

        queue = multiprocessing.Queue()
        threads = []

        for i in range(n_threads_reduced):
            thread = multiprocessing.Process(target=run,
                                             args=(queue,))
            thread.start()

            threads.append(thread)

        for entry in missing_set:
            queue.put(entry)

        # Thread will terminate once None value is reached
        for i in range(n_threads_reduced):
            queue.put(None)

        for thread in threads:
            thread.join()
