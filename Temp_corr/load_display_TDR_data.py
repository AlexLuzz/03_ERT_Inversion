import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from meteostat import Hourly, Stations
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import numpy as np

def load_TDR_data(sensor_data_folder):
    # Get all Excel files in the folder
    xlsx_files = glob.glob(os.path.join(sensor_data_folder, '*.xlsx'))

    dfs = []
    for file in xlsx_files:
        # Read Excel, replace "#/NA#" with np.nan
        df = pd.read_excel(file, na_values=["#/NA"], header=0)
        # Parse Timestamp with both formats
        def parse_date(x):
            for fmt in ["%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return pd.to_datetime(x, format=fmt)
                except Exception:
                    continue
            return pd.NaT
        df['date'] = df['Timestamp'].apply(parse_date)
        df = df.drop(columns=['Timestamp'])

        # Remove empty columns (all NaN or empty)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.match('^Unnamed')]

        dfs.append(df)

    # Merge on 'date'
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged, df, on='date', how='outer')

    # Sort by 'date'
    df_merged = df_merged.sort_values('date').reset_index(drop=True)

    # Move 'date' column to the first position
    cols = df_merged.columns.tolist()
    if 'date' in cols:
        cols.insert(0, cols.pop(cols.index('date')))
        df_merged = df_merged[cols]

    return df_merged

def get_full_sensor_periods(temp_data):
    """
    Returns a tuple: (periods, trimmed_data)
    - periods: list of (start, end) date tuples where all 6 sensor columns have data (not NaN).
    - trimmed_data: temp_data trimmed to only include rows within any of the given periods.
    """
    # Identify the 6 sensor columns
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    # Boolean mask: True where all 6 sensors have data
    mask = temp_data[sensor_cols].notna().all(axis=1)
    periods = []
    in_period = False
    for idx, val in enumerate(mask):
        if val and not in_period:
            start = temp_data['date'].iloc[idx]
            in_period = True
        elif not val and in_period:
            end = temp_data['date'].iloc[idx-1]
            periods.append((start, end))
            in_period = False
    # Handle if the last period goes to the end
    if in_period:
        end = temp_data['date'].iloc[-1]
        periods.append((start, end))

    # Build mask for all periods using 'date' column
    mask = pd.Series(False, index=temp_data.index)
    for start, end in periods:
        mask |= (temp_data['date'] >= start) & (temp_data['date'] <= end)
    # Trim the data to only include rows within any of the periods
    trimmed_data = temp_data[mask].reset_index(drop=True)
    
    return periods, trimmed_data

def plot_weather_and_sensor_periods(
    temp_data,
    period_list=None,
    temp_step=2,
    precip_step=24,
    plot_precip=False
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

    # Convert period start/end to datetime
    period_list = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in period_list]
    n_periods = len(period_list)
    fig, axes = plt.subplots(1, n_periods, figsize=(5 * n_periods, 5), sharey=True)

    # For legend
    handles_labels = []

    # Sensor columns and colors
    sensor_cols = [col for col in temp_data.columns if '-60cm' in col or '-90cm' in col]
    sensor_cols = sensor_cols[:2]  # Limit to first 6 sensor columns
    colors = ['red', 'green', 'blue', 'purple', 'brown', 'magenta']

    for i, (p_start, p_end) in enumerate(period_list):
        ax = axes[i]
        # Filter data for this period
        mask = (temp_data['date'] >= p_start) & (temp_data['date'] <= p_end)
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
                period_df['date'],
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

def fetch_and_aggregate_weather(start_date, end_date, temp_step=2, precip_step=24):
    """
    Fetches hourly weather data and aggregates temperature and precipitation.

    Parameters:
        start_date (str or datetime): Start date.
        end_date (str or datetime): End date.
        temp_step (int): Step size for temperature aggregation.
        precip_step (int): Step size for precipitation aggregation.

    Returns:
        dict: Aggregated times, temperatures, and precipitations.
    """
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    stations = Stations()
    station = stations.region('CA', 'QC')

    # Fetch hourly weather data for the specified date range
    data = Hourly('SOK6B', start_date, end_date)
    data = data.fetch()

    # Extract precipitation and temperature data
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']
    times = data.index

    # Aggregate data based on the specified steps
    t_agg_temp = times[::temp_step]
    temp_agg = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]
    t_agg_precip = times[::precip_step]
    precip_agg = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]

    # Create DataFrames
    temp_df = pd.DataFrame({'date': t_agg_temp, 'temperature': temp_agg})
    precip_df = pd.DataFrame({'date': t_agg_precip, 'precipitation': precip_agg})

    return temp_df, precip_df

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
    temp_df, precip_df = fetch_and_aggregate_weather(start_date, end_date, temp_step, precip_step)
    
    t_agg_temp, temp_agg = temp_df['date'], temp_df['temperature']
    
    t_agg_precip, precip_agg = precip_df['date'], precip_df['precipitation']

    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()  # Get the figure from the provided axis

    # Plot temperature data with color change below 0°C (main y-axis, left)
    temp_colors = ['deepskyblue' if temp < 0 else 'orange' for temp in temp_agg]
    ax.plot(t_agg_temp, temp_agg, color='orange', linestyle='-', linewidth=2, label='Weather Temp (°C)')
    for i in range(len(t_agg_temp) - 1):
        ax.plot(
            t_agg_temp[i:i + 2],
            temp_agg[i:i + 2],
            color=temp_colors[i],
            linestyle='-',
            linewidth=2,
        )
    ax.set_ylabel('Temperature (°C)', color='orange')
    ax.tick_params(axis='y', labelcolor='orange')

    # Optionally plot precipitation on a secondary y-axis (right)
    if plot_precip:
        ax_precip = ax.twinx()
        ax_precip.bar(t_agg_precip, precip_agg, width=0.3, alpha=0.8, color='royalblue', label='Precipitation (mm)')
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

def plot_weather_and_sensors(sensor_data, plot_precip=False):
    """
    Plots weather temperature (and optionally precipitation) and the 6 sensor temperature columns on the same time axis.
    All temperatures (weather and sensors) are plotted on the main y-axis (left).
    """
    # Set start and end date based on date column in temp_data
    start_date = sensor_data['date'].min()
    end_date = sensor_data['date'].max()

    # Plot weather data and get axes
    fig, ax = plot_weather(start_date, end_date, plot_precip=plot_precip)

    # Plot each sensor temperature column on the same axis as weather temperature
    sensor_cols = [col for col in sensor_data.columns if '-60cm' in col or '-90cm' in col]
    sensor_cols = sensor_cols[:2]
    colors = ['red', 'green', 'blue', 'purple', 'brown', 'magenta']
    for i, col in enumerate(sorted(sensor_cols)):
        ax.plot(sensor_data['date'], sensor_data[col], label=col, color=colors[i % len(colors)], linewidth=1.5)

    # Add legend for all temperature curves
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Temperature")
    fig.tight_layout()
    return fig, ax

def fit_chambers_heat_model(sens_data):
    # Estimate T_mean and delta_T from weather data over the last 365 days
    temp_df, _ = fetch_and_aggregate_weather(sens_data['date'].min(), sens_data['date'].max(), 24)
    last_year = temp_df['date'].max() - pd.Timedelta(days=365)
    temp_data_last_year = temp_df[temp_df['date'] >= last_year]
    T_mean = temp_data_last_year['temperature'].mean()
    delta_T = temp_data_last_year['temperature'].max() - temp_data_last_year['temperature'].min()

    # Set datetime index
    temp_data = sens_data.set_index('date')
    temp_data.index = pd.to_datetime(temp_data.index)

    # Resample to daily means for each sensor
    temp_60cm = temp_data['301 - Temp (°C) -60cm '].resample('D').mean()
    temp_90cm = temp_data['301 - Temp (°C) -120cm '].resample('D').mean()

    # Align on shared dates and drop missing values
    combined = pd.concat([temp_60cm, temp_90cm], axis=1, keys=['temp60', 'temp90']).dropna()

    # Use the first date as reference for time conversion
    ref_date = combined.index.min()
    t_days = (combined.index - ref_date).days.values.astype(float)

    # Build input vectors for depths and temperatures
    temps_combined = np.concatenate([combined['temp60'].values, combined['temp90'].values])
    depths_combined = np.concatenate([
        np.full_like(combined['temp60'].values, -0.6),  # -60 cm
        np.full_like(combined['temp90'].values, -0.9)   # -90 cm
    ])
    t_combined = np.concatenate([t_days, t_days])  # duplicated time for both depths

    # Chambers et al. heat model
    def temp_model(inputs, d, phase):
        z, t = inputs
        exponent = -z / d
        sin_part = (2 * np.pi * t / 365.0) + phase - (z / d)
        return T_mean + (delta_T / 2) * np.exp(exponent) * np.sin(sin_part)

    # Fit the model to combined data
    initial_guess = [-0.5, 1]  # d, phase
    popt, pcov = curve_fit(temp_model, (depths_combined, t_combined), temps_combined, p0=initial_guess)

    # Predict modeled temps for the 60cm depth using fitted parameters
    modeled_temps_60 = temp_model((np.full_like(t_days, -0.6), t_days), *popt)

    # Convert back to datetime for plotting or analysis
    modeled_dates = ref_date + pd.to_timedelta(t_days, unit='D')
    observed_temps_60 = combined['temp60']

    return modeled_temps_60, observed_temps_60, modeled_dates, popt
    
if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    
    sensor_data = load_TDR_data(Onedrive_path)

    periods, trimmed_data = get_full_sensor_periods(sensor_data)

    #fig, ax = plot_weather_and_sensors(sensor_data) 

    fig, axes = plot_weather_and_sensor_periods(sensor_data, periods)

    modeled_temps_60, observed_temps_60, days_of_year, popt = fit_chambers_heat_model(sensor_data)

    plt.figure(figsize=(10, 5))
    plt.scatter(days_of_year, modeled_temps_60, label='Modeled Temps', marker='o')
    plt.scatter(days_of_year, observed_temps_60, label='Daily Temps', marker='x')
    plt.xlabel('Days')
    plt.ylabel('Temperature')
    plt.title('Modeled vs Daily Temperatures')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.show()