import pandas as pd
import numpy as np

weather = pd.read_csv('../source/weather.csv', index_col=0)

def weather_c(weather):
    weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])
    weather = weather.set_index('Timestamp')
    weather.index = weather.index.round(freq='15 min')
    weather = weather.sort_values(['SiteId', 'Timestamp', 'Distance'])
    weather = weather.reset_index(level=0)
    weather = weather.drop_duplicates(['Timestamp', 'SiteId'], keep='first')

    return weather

weather_crt=weather_c(weather)

weather_crt.to_csv('../source/weather_crt.csv')

