import pandas as pd
import numpy as np


def process_time(df):
    # Convert timestamp into a pandas datatime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')

    # Extract units of time from the timestamp
    df['min'] = df.index.minute
    df['hour'] = df.index.hour
    df['wday'] = df.index.dayofweek
    df['mday'] = df.index.day
    df['yday'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Create a time of day to represent hours and minutes
    df['time'] = df['hour'] + (df['min'] / 60)
    df = df.drop(columns=['hour', 'min'])


    # wday has period of 6
    df['wday_sin'] = np.sin(2 * np.pi * df['wday'] / 6)
    df['wday_cos'] = np.cos(2 * np.pi * df['wday'] / 6)

    # yday has period of 365
    df['yday_sin'] = np.sin(2 * np.pi * df['yday'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['yday'] / 365)

    # month has period of 12
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # time has period of 24
    df['time_sin'] = np.sin(2 * np.pi * df['time'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['time'] / 24)

    df['wday_1'] = df['wday'] / 6
    df['wday_1'] = df['wday'] / 6

    # yday has period of 365
    df['yday_1'] = df['yday'] / 365
    df['yday_1'] = df['yday'] / 365

    # month has period of 12
    df['month_1'] = df['month'] / 12
    df['month_1'] = df['month'] / 12

    # time has period of 24
    df['time_1'] = df['time'] / 24
    df['time_1'] = df['time'] / 24
    # turn the index into a column
    df = df.reset_index(level=0)

    return df


def add_weather(df, weather):
    # Keep track of the original length of the dataset
    original_length = len(df)

    # Convert timestamp to a pandas datetime object
    weather['Timestamp'] = pd.to_datetime(weather['Timestamp'])
    weather = weather.set_index('Timestamp')
    weather['Temperature_1']=(weather['Temperature']-weather['Temperature'].min())/(weather['Temperature'].max()-weather['Temperature'].min())
    # Round the  weather data to the nearest 15 minutes
    weather.index = weather.index.round(freq='15 min')
    weather = weather.reset_index(level=0)

    # Merge the building data with the weather data
    df = pd.merge(df, weather, how='left', on=['Timestamp', 'SiteId'])

    # Drop the duplicate temperature measurements, keeping the closest location
    df = df.sort_values(['Timestamp', 'SiteId', 'Distance'])
    df = df.drop_duplicates(['Timestamp', 'SiteId'], keep='first')

    # Checking length of new data
    new_length = len(df)

    # Check to make sure the length of the dataset has not changed
    assert original_length == new_length, 'New Length must match original length'

    return df

def add_holiday(df,meta,n):



    # Iterate through each site and find days off

    # Extract the metadata information for the site
    meta_slice = meta.loc[meta['SiteId'] == n]
    #print(meta_slice)
    # Create a new dataframe for the site

    site_meta = pd.DataFrame(columns=['SiteId', 'wday', 'off'],
                                 index=[0, 1, 2, 3, 4, 5, 6])

    site_meta['wday'] = [0, 1, 2, 3, 4, 5, 6]
    site_meta['SiteId'] = n
    if not (meta_slice.empty):
        # Record the days off
        site_meta.loc[0, 'off'] = float(meta_slice['MondayIsDayOff'])
        site_meta.loc[1, 'off'] = float(meta_slice['TuesdayIsDayOff'])
        site_meta.loc[2, 'off'] = float(meta_slice['WednesdayIsDayOff'])
        site_meta.loc[3, 'off'] = float(meta_slice['ThursdayIsDayOff'])
        site_meta.loc[4, 'off'] = float(meta_slice['FridayIsDayOff'])
        site_meta.loc[5, 'off'] = float(meta_slice['SaturdayIsDayOff'])
        site_meta.loc[6, 'off'] = float(meta_slice['SundayIsDayOff'])

        # Append the resulting dataframe to all site dataframe


        # Find the days off in the training and testing data

    else:
        site_meta.loc[0, 'off'] = np.nan
        site_meta.loc[1, 'off'] = np.nan
        site_meta.loc[2, 'off'] = np.nan
        site_meta.loc[3, 'off'] = np.nan
        site_meta.loc[4, 'off'] = np.nan
        site_meta.loc[5, 'off'] = np.nan
        site_meta.loc[6, 'off'] = np.nan

    df = df.merge(site_meta, how='left', on=['SiteId', 'wday'])
    return df

def main():
    meta = pd.read_csv('../source/metadata.csv')
    for i in range(1, 306):
        train = pd.read_csv('../train_cuting/train_cutting{num}.csv'.format(num=i), index_col=0)
        weather = pd.read_csv('../weather_cuting/weather_cutting{num}.csv'.format(num=i), index_col=0)
        train = process_time(train)
        train = add_weather(train, weather)
        train = add_holiday(train, meta, i)
        train.to_csv('../train_deal_c/train/data_add/data_add{num}.csv'.format(num=i), index=False)


if __name__ == '__main__':
    main()
