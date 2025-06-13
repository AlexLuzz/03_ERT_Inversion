import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from meteostat import Hourly, Stations
import matplotlib.dates as mdates


def load_temp_data(sensor_data_folder):
    # Get all CSV files in the folder
    xlsx_files = glob.glob(os.path.join(sensor_data_folder, '*.xlsx'))

    dfs = []
    for file in xlsx_files:
        # Read Excel, replace "#/NA#" with np.nan
        df = pd.read_excel(file, na_values=["#/NA"], header=0)
        # Try to parse Timestamp with both formats
        def parse_date(x):
            for fmt in ["%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return pd.to_datetime(x, format=fmt)
                except Exception:
                    continue
            return pd.NaT
        df['Timestamp'] = df['Timestamp'].apply(parse_date)
        # Identify suffix from filename
        if '301' in file:
            suffix = '_301'
        elif '302' in file:
            suffix = '_302'
        elif '303' in file:
            suffix = '_303'
        else:
            raise ValueError(f"Filename {file} does not contain 301, 302, or 303.")
        # Rename temperature columns
        temp_cols = [col for col in df.columns if '-60cm' in col or '-90cm' in col]
        rename_dict = {col: col + suffix for col in temp_cols}
        df = df.rename(columns=rename_dict)
        # Keep only Timestamp and temperature columns
        keep_cols = ['Timestamp'] + list(rename_dict.values())
        df = df[keep_cols]
        dfs.append(df)

    # Merge on Timestamp
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on='Timestamp', how='outer')

    # Sort by Timestamp
    df_merged = df_merged.sort_values('Timestamp').reset_index(drop=True)
    return df_merged

def plot_weather(start_date, end_date, ax=None, temp_step=2, precip_step=24, plot_precip=False):
    """
    Fetches weather data from the YUL station in Montreal and plots temperature (and optionally precipitation).

    Parameters:
    - start_date (str): Starting date for data selection (format: 'YYYY-MM-DD').
    - end_date (str): Ending date for data selection (format: 'YYYY-MM-DD').
    - ax (Axes): Matplotlib axes object (optional).
    - temp_step (int): Step for temperature data aggregation.
    - precip_step (int): Step for precipitation data aggregation.
    - plot_precip (bool): Whether to plot precipitation (default: False).

    Returns:
    - fig (Figure): The figure object containing the plots.
    - ax (Axes): The axes object for temperature (main y-axis, left).
    """
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    stations = Stations()
    station = stations.region('CA', 'QC')

    # Fetch daily weather data for the specified date range
    data = Hourly('SOK6B', start_date, end_date)
    data = data.fetch()

    # Extract precipitation and temperature data
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']

    times = data.index

    # Aggregate data based on the specified steps
    times_aggregated_temp = times[::temp_step]
    times_aggregated_precip = times[::precip_step]
    precipitation_aggregated = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]
    temperature_aggregated = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]

    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()  # Get the figure from the provided axis

    # Plot temperature data with color change below 0°C (main y-axis, left)
    temp_colors = ['deepskyblue' if temp < 0 else 'orange' for temp in temperature_aggregated]
    ax.plot(times_aggregated_temp, temperature_aggregated, color='orange', linestyle='-', linewidth=2, label='Weather Temp (°C)')
    for i in range(len(times_aggregated_temp) - 1):
        ax.plot(
            times_aggregated_temp[i:i + 2],
            temperature_aggregated[i:i + 2],
            color=temp_colors[i],
            linestyle='-',
            linewidth=2,
        )
    ax.set_ylabel('Temperature (°C)', color='orange')
    ax.tick_params(axis='y', labelcolor='orange')

    # Optionally plot precipitation on a secondary y-axis (right)
    if plot_precip:
        ax_precip = ax.twinx()
        ax_precip.bar(times_aggregated_precip, precipitation_aggregated, width=0.3, alpha=0.8, color='royalblue', label='Precipitation (mm)')
        ax_precip.set_ylabel('Precipitation (mm)', color='royalblue')
        ax_precip.tick_params(axis='y', labelcolor='royalblue')

    # Add a horizontal line at 0°C
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # Format x-axis for date and time
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right')

    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    title = "Weather Data: Temperature"
    if plot_precip:
        title += " and Precipitation"
        
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

    return fig, ax

def plot_weather_periods(start_date, end_date, ax=None, temp_step=2, precip_step=24, plot_precip=False, period_list=None):
    """
    Plots weather temperature (and optionally precipitation) only during periods where all sensors have data.
    Draws a vertical black line between each period if period_list is provided.
    """
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Fetch weather data
    data = Hourly('SOK6B', start_date, end_date).fetch()
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']
    times = data.index

    # Aggregate data
    times_aggregated_temp = times[::temp_step]
    temperature_aggregated = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]
    times_aggregated_precip = times[::precip_step]
    precipitation_aggregated = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]

    # Filter to keep only values inside any period
    if period_list is not None:
        keep_mask = pd.Series(False, index=times_aggregated_temp)
        for p_start, p_end in period_list:
            keep_mask |= (times_aggregated_temp >= p_start) & (times_aggregated_temp <= p_end)
        times_aggregated_temp = times_aggregated_temp[keep_mask]
        temperature_aggregated = pd.Series(temperature_aggregated, index=keep_mask.index)[keep_mask]

    # Create figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot filtered temperature
    ax.plot(times_aggregated_temp, temperature_aggregated, color='orange', linestyle='-', linewidth=2, label='Weather Temp (°C)')

    # Draw vertical black lines between periods
    if period_list is not None and len(period_list) > 1:
        for _, p_end in period_list[:-1]:
            ax.axvline(pd.to_datetime(p_end), color='black', linestyle='-', linewidth=2)

    ax.set_ylabel('Temperature (°C)', color='orange')
    ax.tick_params(axis='y', labelcolor='orange')

    # Optionally plot precipitation
    if plot_precip:
        ax_precip = ax.twinx()
        ax_precip.bar(times_aggregated_precip, precipitation_aggregated, width=0.3, alpha=0.8, color='royalblue', label='Precipitation (mm)')
        ax_precip.set_ylabel('Precipitation (mm)', color='royalblue')
        ax_precip.tick_params(axis='y', labelcolor='royalblue')

    # Add a horizontal line at 0°C
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    # Format x-axis for date and time
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right')

    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

    return fig, ax

def get_full_sensor_periods(temp_data):
    """
    Returns a list of (start, end) Timestamp tuples where all 6 sensor columns have data (not NaN).
    """
    # Identify the 6 sensor columns
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    # Boolean mask: True where all 6 sensors have data
    mask = temp_data[sensor_cols].notna().all(axis=1)
    periods = []
    in_period = False
    for idx, val in enumerate(mask):
        if val and not in_period:
            start = temp_data['Timestamp'].iloc[idx]
            in_period = True
        elif not val and in_period:
            end = temp_data['Timestamp'].iloc[idx-1]
            periods.append((start, end))
            in_period = False
    # Handle if the last period goes to the end
    if in_period:
        end = temp_data['Timestamp'].iloc[-1]
        periods.append((start, end))
    return periods

def plot_weather_and_sensors(temp_data, start_date, end_date, plot_precip=False):
    """
    Plots weather temperature (and optionally precipitation) and the 6 sensor temperature columns on the same time axis.
    All temperatures (weather and sensors) are plotted on the main y-axis (left).
    """
    # Plot weather data and get axes
    fig, ax = plot_weather(start_date, end_date, plot_precip=plot_precip)

    # Plot each sensor temperature column on the same axis as weather temperature
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    colors = ['red', 'green', 'blue', 'purple', 'brown', 'magenta']
    for i, col in enumerate(sorted(sensor_cols)):
        ax.plot(temp_data['Timestamp'], temp_data[col], label=col, color=colors[i % len(colors)], linewidth=1.5)

    # Add legend for all temperature curves
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Temperature")
    fig.tight_layout()
    plt.show()

def get_full_sensor_periods(temp_data):
    """
    Returns a list of (start, end) Timestamp tuples where all 6 sensor columns have data (not NaN).
    """
    # Identify the 6 sensor columns
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    # Boolean mask: True where all 6 sensors have data
    mask = temp_data[sensor_cols].notna().all(axis=1)
    periods = []
    in_period = False
    for idx, val in enumerate(mask):
        if val and not in_period:
            start = temp_data['Timestamp'].iloc[idx]
            in_period = True
        elif not val and in_period:
            end = temp_data['Timestamp'].iloc[idx-1]
            periods.append((start, end))
            in_period = False
    # Handle if the last period goes to the end
    if in_period:
        end = temp_data['Timestamp'].iloc[-1]
        periods.append((start, end))
    return periods

if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    
    temp_data = load_temp_data(Onedrive_path)

    periods = get_full_sensor_periods(temp_data)

    # Set start and end date based on your data or manually
    start_date = temp_data['Timestamp'].min()
    end_date = temp_data['Timestamp'].max()
    #plot_weather_and_sensors(temp_data, start_date, end_date) 

    plot_weather_and_sensors(temp_data, periods[2][0], periods[2][1])

    plt.show()