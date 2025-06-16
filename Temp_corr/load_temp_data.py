import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from meteostat import Hourly, Stations
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import numpy as np

def load_TDR_data(sensor_data_folder):
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

def plot_weather_sensor_periods(
    temp_data,
    temp_step=2,
    precip_step=24,
    plot_precip=False,
    period_list=None
):
    """
    Plots weather temperature (and optionally precipitation) and sensor data for each period.
    Each period is shown in its own subplot (1 row, X columns).
    temp_data: DataFrame with 'Timestamp' and sensor columns.
    """
    # Fetch weather data for the full range
    start_date = period_list[0][0]
    end_date = period_list[-1][1]
    data = Hourly('SOK6B', start_date, end_date).fetch()
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']
    times = data.index

    # Aggregate weather data
    times_aggregated_temp = times[::temp_step]
    temperature_aggregated = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]
    times_aggregated_precip = times[::precip_step]
    precipitation_aggregated = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]


    # Ensure Timestamp is datetime
    temp_data = temp_data.copy()
    temp_data['Timestamp'] = pd.to_datetime(temp_data['Timestamp'])

    # Convert period start/end to datetime
    period_list = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in period_list]

    n_periods = len(period_list)
    fig, axes = plt.subplots(1, n_periods, figsize=(5 * n_periods, 5), sharey=True)

    # For legend
    handles_labels = []

    # Sensor columns and colors
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    colors = ['red', 'green', 'blue', 'purple', 'brown', 'magenta']

    for i, (p_start, p_end) in enumerate(period_list):
        ax = axes[i]
        # Filter data for this period
        mask = (temp_data['Timestamp'] >= p_start) & (temp_data['Timestamp'] <= p_end)
        period_df = temp_data.loc[mask]

        # Filter temperature for this period
        mask_temp = (times_aggregated_temp >= p_start) & (times_aggregated_temp <= p_end)
        t_temp = times_aggregated_temp[mask_temp]
        temp = pd.Series(temperature_aggregated, index=times_aggregated_temp)[mask_temp]

        l1, = ax.plot(t_temp, temp, color='orange', linestyle='-', linewidth=2, label='Weather Temp (°C)')

        # Plot each sensor temperature column
        sensor_lines = []
        for j, col in enumerate(sorted(sensor_cols)):
            l2, = ax.plot(
                period_df['Timestamp'],
                period_df[col],
                label=col,
                color=colors[j % len(colors)],
                linewidth=1.5
            )
            sensor_lines.append(l2)

        ax.set_ylabel('Temperature (°C)')
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_title(f"Period {i+1}\n{p_start.date()} to {p_end.date()}")

        # Format x-axis
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60, ha='right')
        ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.6)

        # Collect handles/labels for legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            handles_labels.extend(zip(handles, labels))

    # Remove duplicate labels for legend
    seen = set()
    unique_handles_labels = []
    for h, l in handles_labels:
        if l not in seen:
            unique_handles_labels.append((h, l))
            seen.add(l)
    handles, labels = zip(*unique_handles_labels)

    # Place legend outside the plot
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(labels))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

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

def fit_chambers_heat_model(temp_data):

    # Heat equation model from Chambers et al.
    def temp_model(inputs, T_mean, delta_T, d, phase):
        z, t = inputs
        exponent = -z / d
        sin_thingy = (2 * np.pi * t / 365 + phase - z / d)
        return T_mean + (delta_T / 2) * np.exp(exponent) * np.sin(sin_thingy)

    depths = np.array([-0.6, -0.9, -0.6, -0.9, -0.6, -0.9])
    days = np.array(pd.to_datetime(temp_data['Timestamp']))
    temperatures = temp_data[[col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]].values
    temps = np.array(temperatures)

    # Initial guess: T_mean=10, delta_T=20, d=1, u=0
    initial_guess = [10, 20, 1, 0]

    # Curve fitting
    popt, pcov = curve_fit(temp_model, (depths, days), temps, p0=initial_guess)

    return modeled_temps, params

def trimed_temp_data(temp_data, periods):
    """
    Returns temp_data trimmed to only include rows within any of the given periods.

    Parameters:
        temp_data (pd.DataFrame): DataFrame with a DatetimeIndex or a 'Timestamp' column.
        periods (list of tuple): List of (start_datetime, end_datetime) tuples.

    Returns:
        pd.DataFrame: Trimmed temp_data.
    """
    # Ensure periods are pd.Timestamp
    periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods]

    # Ensure index is datetime
    if 'Timestamp' in temp_data.columns:
        temp_data = temp_data.copy()
        temp_data['Timestamp'] = pd.to_datetime(temp_data['Timestamp'])
        idx = temp_data['Timestamp']
    else:
        idx = temp_data.index

    # Build mask for all periods
    mask = pd.Series(False, index=temp_data.index)
    for start, end in periods:
        mask |= (idx >= start) & (idx <= end)

    return temp_data[mask]
    
if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    
    temp_data = load_TDR_data(Onedrive_path)

    periods = get_full_sensor_periods(temp_data)

    trimed_temp = trimed_temp_data(temp_data, periods)

    # Set start and end date based on your data or manually
    start_date = temp_data['Timestamp'].min()
    end_date = temp_data['Timestamp'].max()

    #plot_weather_and_sensors(temp_data, start_date, end_date) 

    #plot_weather_and_sensors(temp_data, periods[2][0], periods[2][1])

    # Plot weather and sensors for each period
    #fig, axes = plot_weather_sensor_periods(temp_data, period_list=periods)

    modeled_temps, params = fit_chambers_heat_model(trimed_temp)

    plt.show()