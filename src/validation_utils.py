import datetime
import argparse


def validate_month(month):
    """
    Validate whether the month string is in the form "YYYY-MM".
    Throws an ArgumentTypeError if this is not the case.
    """
    try:
        date = datetime.datetime.strptime(month, '%Y-%m')
        return date
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid month format. Please use YYYY-MM")


def validate_weather(weather):
    """
    Validate whether the weather is rain or snow.
    Throws an ArgumentTypeError if this is not the case.
    """
    if weather.lower() not in ['rain', 'snow']:
        raise argparse.ArgumentTypeError("Weather type must be 'rain' or 'snow'")
    return weather.lower()
