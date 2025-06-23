import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import sys
import glob
import matplotlib.pyplot as plt
from meteostat import Hourly, Stations

class DebugCapture:
    """Capture debug output for later use in front page"""
    def __init__(self):
        self.captured_output = []
        
    def write(self, text):
        self.captured_output.append(text)
        
    def get_output(self):
        return ''.join(self.captured_output)
        
    def clear(self):
        self.captured_output = []

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

def createDebugFrontPage(debug_text, title="Chambers Model Analysis Report", figsize=(12, 7)):
    """Create front page figure(s) with debug information, spanning multiple pages if necessary.
    
    Parameters:
        debug_text (str): Debug output text to display.
        title (str): Main title for the front page.
        figsize (tuple): Size of the front page figure.
    
    Returns:
        list: List of matplotlib.figure.Figure objects (one or more pages).
    """
    
    # Split text into lines and clean up
    lines = debug_text.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    # Calculate how many lines fit per page
    y_start = 0.85
    y_end = 0.05
    line_height = 0.025
    lines_per_page = int((y_start - y_end) / line_height)
    
    # Split lines into pages
    pages = []
    for i in range(0, len(clean_lines), lines_per_page):
        pages.append(clean_lines[i:i + lines_per_page])
    
    figures = []
    
    for page_num, page_lines in enumerate(pages):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')  # Remove axes
        
        # Title (only on first page)
        if page_num == 0:
            ax.text(0.5, 0.95, title, fontsize=20, fontweight='bold', 
                    ha='center', va='top', transform=ax.transAxes)
            
            # Date
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ax.text(0.5, 0.90, f"Generated: {date_str}", fontsize=12, 
                    ha='center', va='top', transform=ax.transAxes)
        else:
            # Page header for continuation pages
            ax.text(0.5, 0.95, f"{title} (Page {page_num + 1})", fontsize=16, fontweight='bold', 
                    ha='center', va='top', transform=ax.transAxes)
            y_start = 0.90  # Start higher on continuation pages
        
        # Display debug text for this page
        for i, line in enumerate(page_lines):
            y_pos = y_start - i * line_height
            
            # Choose font size based on content
            if line.startswith('='):
                fontsize = 14
                fontweight = 'bold'
            elif 'Analyzing Borehole' in line:
                fontsize = 12
                fontweight = 'bold'
            elif line.startswith('  '):
                fontsize = 10
                fontweight = 'normal'
            else:
                fontsize = 11
                fontweight = 'normal'
                
            ax.text(0.05, y_pos, line, fontsize=fontsize, fontweight=fontweight,
                    ha='left', va='top', transform=ax.transAxes, 
                    fontfamily='monospace')
        
        # Add page number at bottom
        if len(pages) > 1:
            ax.text(0.5, 0.02, f"Page {page_num + 1} of {len(pages)}", fontsize=10, 
                    ha='center', va='bottom', transform=ax.transAxes, style='italic')
        
        # Add a border
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                                  edgecolor='black', linewidth=1, transform=ax.transAxes))
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def saveFiguresToPDF(figures, debug_text, pdf_filename, title="Chambers Model Analysis Report", 
                     figsize=(12, 7), verbose=False, dpi=300, metadata=None):
    """Save figures to PDF with debug front pages that can span multiple pages.
    
    Parameters:
        figures (list): List of analysis figure objects.
        debug_text (str): Debug output text for front pages.
        pdf_filename (str): Output PDF filename.
        title (str): Title for the front pages.
        figsize (tuple): Figure size.
        verbose (bool): Verbose output.
        dpi (int): PDF resolution.
        metadata (dict): PDF metadata.
    
    Returns:
        str: Path to created PDF file.
    """
    
    # Create debug front pages
    front_pages = createDebugFrontPage(debug_text, title, figsize)
    
    # Combine front pages with analysis figures
    all_figures = front_pages + figures
    
    # Set up PDF metadata
    pdf_metadata = {
        'Title': title,
        'Author': 'Python Analysis Script',
        'Subject': 'Temperature Model Fitting',
        'Creator': 'matplotlib',
        'CreationDate': datetime.now()
    }
    if metadata:
        pdf_metadata.update(metadata)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(pdf_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Created directory: {output_dir}")
    
    try:
        with PdfPages(pdf_filename, metadata=pdf_metadata) as pdf:
            # Save all figures (front pages + analysis plots)
            for i, fig in enumerate(all_figures):
                if verbose:
                    page_type = "front page" if i < len(front_pages) else "analysis figure"
                    print(f"Saving {page_type} {i + 1}/{len(all_figures)} to PDF.")
                
                try:
                    fig.set_size_inches(figsize)
                    pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
                    plt.close(fig)
                except Exception as e:
                    print(f"Warning: Failed to save figure {i + 1} - {e}")
                    continue
        
        if verbose:
            print(f"PDF saved with {len(front_pages)} front pages and {len(figures)} analysis figures")
            print(f"Total pages: {len(all_figures)}")
            print(f"PDF file: {pdf_filename}")
            print(f"File size: {os.path.getsize(pdf_filename) / 1024:.1f} KB")
        
        return pdf_filename
        
    except Exception as e:
        raise OSError(f"Failed to create PDF file '{pdf_filename}': {e}")

def fit_chambers_model_original(sens_data, borehole='301', debug=True, capture_debug=False):
    """
    Original Chambers et al. (2014) heat model using weather data for air temperature parameters
    """
    
    # Set up debug capture if requested
    debug_capture = None
    if capture_debug:
        debug_capture = DebugCapture()
        original_stdout = sys.stdout
        sys.stdout = debug_capture
    
    try:
        # Get weather data for the FULL sensor data range
        full_date_min = sens_data['date'].min()
        full_date_max = sens_data['date'].max()
        temp_df, _ = fetch_and_aggregate_weather(full_date_min, full_date_max, 24)
        
        # Calculate weather-based air temperature parameters from full dataset
        weather_daily = temp_df.set_index('date').resample('D').mean()
        T_weather_mean = weather_daily['temperature'].mean()
        #T_weather_mean = trim_mean(weather_daily['temperature'], proportiontocut=0.1)
        delta_T_weather = (weather_daily['temperature'].max() - weather_daily['temperature'].min())
        #delta_T_weather = weather_daily['temperature'].quantile(0.9) - weather_daily['temperature'].quantile(0.1) 

        if debug:
            print(f"Weather-based air temperature parameters:")
            print(f"  T_weather_mean: {T_weather_mean:.2f}°C")
            print(f"  delta_T_weather: {delta_T_weather:.2f}°C")
            print(f"  Full date range: {full_date_min} to {full_date_max}")
        
        # Set datetime index for sensor data
        temp_data = sens_data.set_index('date')
        temp_data.index = pd.to_datetime(temp_data.index)
        
        # Get sensor columns for the specified borehole
        col_60 = f'{borehole} - Temp (°C) -60cm '
        col_120 = f'{borehole} - Temp (°C) -120cm '
        
        # Check if columns exist
        if col_60 not in temp_data.columns or col_120 not in temp_data.columns:
            available_cols = [col for col in temp_data.columns if borehole in col]
            raise ValueError(f"Columns not found. Available columns for {borehole}: {available_cols}")
        
        # Resample to daily means
        temp_60cm = temp_data[col_60].resample('D').mean()
        temp_120cm = temp_data[col_120].resample('D').mean()
        
        # Align on shared dates and drop missing values
        combined = pd.concat([temp_60cm, temp_120cm], axis=1, keys=['temp60', 'temp120']).dropna()
        
        if len(combined) < 10:
            raise ValueError(f"Insufficient data after cleaning: only {len(combined)} valid days")
        
        if debug:
            print(f"Valid sensor data points: {len(combined)} days")
            print(f"Sensor date range: {combined.index.min()} to {combined.index.max()}")
            print(f"-60cm temp range: {combined['temp60'].min():.2f} to {combined['temp60'].max():.2f}°C")
            print(f"-120cm temp range: {combined['temp120'].min():.2f} to {combined['temp120'].max():.2f}°C")
        
        # Convert to days since start of year (to align with seasonal cycle)
        start_of_year = pd.Timestamp(combined.index.min().year, 1, 1)
        t_days = (combined.index - start_of_year).days.values.astype(float)
        
        # Prepare data for fitting - stack depths and temperatures
        depths = np.array([0.6, 1.2])  # depths in meters
        n_times = len(t_days)
        n_depths = len(depths)
        
        # Create arrays for all depth-time combinations
        temps_all = np.concatenate([combined['temp60'].values, combined['temp120'].values])
        depths_all = np.repeat(depths, n_times)
        times_all = np.tile(t_days, n_depths)
        
        # Original Chambers model - only fitting damping depth and phase shift
        def chambers_model_original(inputs, d, phase_shift):
            """
            Original Chambers et al. heat conduction model with fixed air temperature parameters from weather
            """
            z, t = inputs
            
            # Convert time to annual cycle (t in days since start of year)
            t_annual = 2 * np.pi * t / 365.25  # Convert to radians for the year
            
            # Damping factor (exponential decay with depth)
            damping = np.exp(-z / d)
            
            # Phase lag due to depth (delay increases with depth)
            phase_lag = -z / d
            
            # Temperature model using WEATHER-BASED air temperature parameters
            temp = T_weather_mean + delta_T_weather / 2 * damping * np.sin(t_annual + phase_shift + phase_lag)
            
            return temp

        # Phase shift estimation (start with spring = 0)
        phase_shift_est = 0
        
        initial_guess = [1, phase_shift_est]
        
        # Parameter bounds - only for the two fitted parameters
        bounds = (
            [0.1, -2*np.pi],  # Lower bounds: [d_min, phase_min]
            [5.0, 2*np.pi]    # Upper bounds: [d_max, phase_max]
        )
        
        # Fit the model - only damping depth and phase shift
        popt, pcov = curve_fit(
            chambers_model_original, 
            (depths_all, times_all), 
            temps_all, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        d_fitted, phase_fitted = popt
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        
        if debug:
            print(f"\nFitted parameters:")
            print(f"  Damping depth (d): {d_fitted:.3f} ± {param_errors[0]:.3f} m")
            print(f"  Phase shift: {phase_fitted:.3f} ± {param_errors[1]:.3f} rad ({np.degrees(phase_fitted):.1f}°)")
            print(f"Fixed parameters:")
            print(f"  Air mean temp: {T_weather_mean:.2f}°C (from weather)")
            print(f"  Air temp amplitude: {delta_T_weather:.2f}°C (from weather)")
        
        # Generate predictions for both depths - FOR FITTING DATES
        pred_60 = chambers_model_original((np.full_like(t_days, 0.6), t_days), *popt)
        pred_120 = chambers_model_original((np.full_like(t_days, 1.2), t_days), *popt)
        
        # Generate predictions for FULL DATE RANGE (for plotting)
        full_date_range = pd.date_range(full_date_min, full_date_max, freq='D')
        start_of_year_full = pd.Timestamp(full_date_range.min().year, 1, 1)
        t_days_full = (full_date_range - start_of_year_full).days.values.astype(float)
        
        pred_60_full = chambers_model_original((np.full_like(t_days_full, 0.6), t_days_full), *popt)
        pred_120_full = chambers_model_original((np.full_like(t_days_full, 1.2), t_days_full), *popt)
        
        # Also generate air temperature prediction for comparison
        pred_air_full = T_weather_mean + delta_T_weather * np.sin(2 * np.pi * t_days_full / 365.25 + phase_fitted)
        
        # Calculate R² and RMSE for each depth
        r2_60 = 1 - np.sum((combined['temp60'] - pred_60)**2) / np.sum((combined['temp60'] - combined['temp60'].mean())**2)
        r2_120 = 1 - np.sum((combined['temp120'] - pred_120)**2) / np.sum((combined['temp120'] - combined['temp120'].mean())**2)
        
        rmse_60 = np.sqrt(np.mean((combined['temp60'] - pred_60)**2))
        rmse_120 = np.sqrt(np.mean((combined['temp120'] - pred_120)**2))
        
        # Overall metrics
        r2_overall = 1 - np.sum((temps_all - chambers_model_original((depths_all, times_all), *popt))**2) / np.sum((temps_all - temps_all.mean())**2)
        rmse_overall = np.sqrt(np.mean((temps_all - chambers_model_original((depths_all, times_all), *popt))**2))
        
        if debug:
            print(f"\nModel performance:")
            print(f"  60cm depth - R²: {r2_60:.3f}, RMSE: {rmse_60:.3f}°C")
            print(f"  120cm depth - R²: {r2_120:.3f}, RMSE: {rmse_120:.3f}°C")
            print(f"  Overall - R²: {r2_overall:.3f}, RMSE: {rmse_overall:.3f}°C")
        
        # Prepare results
        results = {
            'parameters': {
                'T_weather_mean': T_weather_mean,  # Fixed from weather
                'delta_T_weather': delta_T_weather,  # Fixed from weather
                'damping_depth_m': d_fitted,  # Fitted
                'phase_shift_rad': phase_fitted,  # Fitted
                'phase_shift_deg': np.degrees(phase_fitted)
            },
            'parameter_errors': {
                'damping_depth_m': param_errors[0],
                'phase_shift_rad': param_errors[1]
                # No errors for fixed weather parameters
            },
            'parameter_source': {
                'T_weather_mean': 'weather_data',
                'delta_T_weather': 'weather_data', 
                'damping_depth_m': 'fitted',
                'phase_shift_rad': 'fitted'
            },
            'predictions': {
                'dates': combined.index,
                'temp_60cm_obs': combined['temp60'],
                'temp_60cm_pred': pred_60,
                'temp_120cm_obs': combined['temp120'],
                'temp_120cm_pred': pred_120,
                'time_days': t_days
            },
            'predictions_full': {
                'dates_full': full_date_range,
                'temp_air_pred_full': pred_air_full,
                'temp_60cm_pred_full': pred_60_full,
                'temp_120cm_pred_full': pred_120_full,
                'time_days_full': t_days_full
            },
            'diagnostics': {
                'r2_60cm': r2_60,
                'r2_120cm': r2_120,
                'r2_overall': r2_overall,
                'rmse_60cm': rmse_60,
                'rmse_120cm': rmse_120,
                'rmse_overall': rmse_overall,
                'n_observations': len(combined)
            },
            'model_function': chambers_model_original,
            'fitted_params': popt,
            'weather_params': {
                'T_weather_mean': T_weather_mean,
                'delta_T_weather': delta_T_weather
            }
        }
        
        # Add debug capture to results if available
        if capture_debug and debug_capture:
            results['debug_output'] = debug_capture.get_output()
        
        return results, temp_df
        
    except Exception as e:
        print(f"Original Chambers fitting failed: {str(e)}")
        print("Trying simpler approach...")
        return None
    finally:
        # Restore stdout if it was captured
        if capture_debug and debug_capture:
            sys.stdout = original_stdout

def fit_chambers_heat_model_improved(sens_data, borehole='301', debug=True, capture_debug=False):
    """
    Improved Chambers et al. (2014) heat model fitting with better debugging
    """
    # Set up debug capture if requested
    debug_capture = None
    if capture_debug:
        debug_capture = DebugCapture()
        original_stdout = sys.stdout
        sys.stdout = debug_capture

    try:
        # Get weather data for the FULL sensor data range (not just last year)
        full_date_min = sens_data['date'].min()
        full_date_max = sens_data['date'].max()
        temp_df, _ = fetch_and_aggregate_weather(full_date_min, full_date_max, 24)
        
        # Use last year for parameter estimation but keep full range for plotting
        last_year = temp_df['date'].max() - pd.Timedelta(days=365)
        temp_data_last_year = temp_df[temp_df['date'] >= last_year]
        T_mean = temp_data_last_year['temperature'].mean()
        delta_T = temp_data_last_year['temperature'].max() - temp_data_last_year['temperature'].min()

        if debug:
            print(f"Weather data - T_mean: {T_mean:.2f}°C, delta_T: {delta_T:.2f}°C")
            print(f"Full date range: {full_date_min} to {full_date_max}")
        
        # Set datetime index
        temp_data = sens_data.set_index('date')
        temp_data.index = pd.to_datetime(temp_data.index)
        
        # Get sensor columns for the specified borehole
        col_60 = f'{borehole} - Temp (°C) -60cm '
        col_120 = f'{borehole} - Temp (°C) -120cm '
        
        # Check if columns exist
        if col_60 not in temp_data.columns or col_120 not in temp_data.columns:
            available_cols = [col for col in temp_data.columns if borehole in col]
            raise ValueError(f"Columns not found. Available columns for {borehole}: {available_cols}")
        
        # Resample to daily means
        temp_60cm = temp_data[col_60].resample('D').mean()
        temp_120cm = temp_data[col_120].resample('D').mean()
        
        # Align on shared dates and drop missing values
        combined = pd.concat([temp_60cm, temp_120cm], axis=1, keys=['temp60', 'temp120']).dropna()
        
        if debug:
            print(f"Valid data points: {len(combined)} days")
            print(f"Date range: {combined.index.min()} to {combined.index.max()}")
            print(f"60cm temp range: {combined['temp60'].min():.2f} to {combined['temp60'].max():.2f}°C")
            print(f"120cm temp range: {combined['temp120'].min():.2f} to {combined['temp120'].max():.2f}°C")
        
        # Convert to days since start of year (to better align with seasonal cycle)
        start_of_year = pd.Timestamp(combined.index.min().year, 1, 1)
        t_days = (combined.index - start_of_year).days.values.astype(float)
        
        # Calculate observed means and ranges for each depth
        obs_mean_60 = combined['temp60'].mean()
        obs_mean_120 = combined['temp120'].mean()
        obs_range_60 = combined['temp60'].max() - combined['temp60'].min()
        obs_range_120 = combined['temp120'].max() - combined['temp120'].min()
        
        if debug:
            print(f"Observed 60cm: mean={obs_mean_60:.2f}°C, range={obs_range_60:.2f}°C")
            print(f"Observed 120cm: mean={obs_mean_120:.2f}°C, range={obs_range_120:.2f}°C")
        
        # Prepare data for fitting - stack depths and temperatures
        depths = np.array([0.6, 1.2])  # Use positive depths for easier interpretation
        n_times = len(t_days)
        n_depths = len(depths)
        
        # Create arrays for all depth-time combinations
        temps_all = np.concatenate([combined['temp60'].values, combined['temp120'].values])
        depths_all = np.repeat(depths, n_times)
        times_all = np.tile(t_days, n_depths)
        
        # Improved Chambers model with better parameterization
        def temp_model_improved(inputs, T_surf_mean, delta_T_surf, d, phase_shift):
            """
            Improved Chambers et al. heat conduction model
            
            Parameters:
            -----------
            T_surf_mean : surface mean temperature
            delta_T_surf : surface temperature amplitude
            d : damping depth (m)
            phase_shift : phase shift (radians)
            """
            z, t = inputs
            
            # Convert time to annual cycle (t in days since start of year)
            t_annual = 2 * np.pi * t / 365.25  # Convert to radians for the year
            
            # Damping factor (exponential decay with depth)
            damping = np.exp(-z / d)
            
            # Phase lag due to depth (delay increases with depth)
            phase_lag = -z / 4*d
            
            # Temperature model
            temp = T_surf_mean + delta_T_surf * damping * np.sin(t_annual + phase_shift + phase_lag)
            
            return temp
        
        # Better initial parameter estimation
        # Estimate surface parameters from shallow depth
        T_surf_mean_est = obs_mean_60  # Start with 60cm mean
        delta_T_surf_est = obs_range_60 * 2  # Assume surface has larger amplitude
        
        # Estimate damping depth from amplitude reduction
        if obs_range_60 > 0:
            damping_ratio = obs_range_120 / obs_range_60 if obs_range_120 > 0 else 0.5
            if damping_ratio > 0:
                d_est = (1.2 - 0.6) / np.log(1/damping_ratio) if damping_ratio < 1 else 1.0
            else:
                d_est = 1.0
        else:
            d_est = 1.0
        
        # Phase shift estimation (start with spring = 0)
        phase_shift_est = 0
        
        initial_guess = [T_surf_mean_est, delta_T_surf_est, d_est, phase_shift_est]
        
        if debug:
            print(f"Initial parameter estimates:")
            print(f"  T_surf_mean: {T_surf_mean_est:.2f}°C")
            print(f"  delta_T_surf: {delta_T_surf_est:.2f}°C")
            print(f"  Damping depth: {d_est:.3f} m")
            print(f"  Phase shift: {phase_shift_est:.3f} rad")
        
        # More reasonable parameter bounds
        bounds = (
            [obs_mean_60 - 10, 0.1, 0.1, -2*np.pi],  # Lower bounds
            [obs_mean_60 + 10, 50, 5.0, 2*np.pi]     # Upper bounds
        )
        
        try:
            # Fit the model
            popt, pcov = curve_fit(
                temp_model_improved, 
                (depths_all, times_all), 
                temps_all, 
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            T_surf_fitted, delta_T_fitted, d_fitted, phase_fitted = popt
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            if debug:
                print(f"\nFitted parameters:")
                print(f"  Surface mean temp: {T_surf_fitted:.3f} ± {param_errors[0]:.3f}°C")
                print(f"  Surface temp amplitude: {delta_T_fitted:.3f} ± {param_errors[1]:.3f}°C")
                print(f"  Damping depth (d): {d_fitted:.3f} ± {param_errors[2]:.3f} m")
                print(f"  Phase shift: {phase_fitted:.3f} ± {param_errors[3]:.3f} rad ({np.degrees(phase_fitted):.1f}°)")
            
            # Generate predictions for both depths - FOR FITTING DATES
            pred_60 = temp_model_improved((np.full_like(t_days, 0.6), t_days), *popt)
            pred_120 = temp_model_improved((np.full_like(t_days, 1.2), t_days), *popt)
            
            # Generate predictions for FULL DATE RANGE (for plotting)
            full_date_range = pd.date_range(full_date_min, full_date_max, freq='D')
            start_of_year_full = pd.Timestamp(full_date_range.min().year, 1, 1)
            t_days_full = (full_date_range - start_of_year_full).days.values.astype(float)
            
            pred_60_full = temp_model_improved((np.full_like(t_days_full, 0.6), t_days_full), *popt)
            pred_120_full = temp_model_improved((np.full_like(t_days_full, 1.2), t_days_full), *popt)
            
            # Calculate R² and RMSE for each depth
            r2_60 = 1 - np.sum((combined['temp60'] - pred_60)**2) / np.sum((combined['temp60'] - combined['temp60'].mean())**2)
            r2_120 = 1 - np.sum((combined['temp120'] - pred_120)**2) / np.sum((combined['temp120'] - combined['temp120'].mean())**2)
            
            rmse_60 = np.sqrt(np.mean((combined['temp60'] - pred_60)**2))
            rmse_120 = np.sqrt(np.mean((combined['temp120'] - pred_120)**2))
            
            # Overall metrics
            r2_overall = 1 - np.sum((temps_all - temp_model_improved((depths_all, times_all), *popt))**2) / np.sum((temps_all - temps_all.mean())**2)
            rmse_overall = np.sqrt(np.mean((temps_all - temp_model_improved((depths_all, times_all), *popt))**2))
            
            if debug:
                print(f"\nModel performance:")
                print(f"  60cm depth - R²: {r2_60:.3f}, RMSE: {rmse_60:.3f}°C")
                print(f"  120cm depth - R²: {r2_120:.3f}, RMSE: {rmse_120:.3f}°C")
                print(f"  Overall - R²: {r2_overall:.3f}, RMSE: {rmse_overall:.3f}°C")
            
            # Prepare results
            results = {
                'parameters': {
                    'T_surface_mean': T_surf_fitted,
                    'delta_T_surface': delta_T_fitted,
                    'damping_depth_m': d_fitted,
                    'phase_shift_rad': phase_fitted,
                    'phase_shift_deg': np.degrees(phase_fitted)
                },
                'parameter_errors': dict(zip(
                    ['T_surface_mean', 'delta_T_surface', 'damping_depth_m', 'phase_shift_rad'],
                    param_errors
                )),
                'predictions': {
                    'dates': combined.index,
                    'temp_60cm_obs': combined['temp60'],
                    'temp_60cm_pred': pred_60,
                    'temp_120cm_obs': combined['temp120'],
                    'temp_120cm_pred': pred_120,
                    'time_days': t_days
                },
                'predictions_full': {
                    'dates_full': full_date_range,
                    'temp_60cm_pred_full': pred_60_full,
                    'temp_120cm_pred_full': pred_120_full,
                    'time_days_full': t_days_full
                },
                'diagnostics': {
                    'r2_60cm': r2_60,
                    'r2_120cm': r2_120,
                    'r2_overall': r2_overall,
                    'rmse_60cm': rmse_60,
                    'rmse_120cm': rmse_120,
                    'rmse_overall': rmse_overall,
                    'n_observations': len(combined)
                },
                'model_function': temp_model_improved,
                'fitted_params': popt
            }
            # Add debug capture to results if available
            if capture_debug and debug_capture:
                results['debug_output'] = debug_capture.get_output()
            
            return results, temp_df
        except Exception as e:
            print(f"Fitting failed: {str(e)}")
            return None, None
    finally:
        # Restore stdout if it was captured
        if capture_debug and debug_capture:
            sys.stdout = original_stdout

def plot_results(results, weather_data, borehole='301', save_figure=False):
    """Plot results from original Chambers model with weather and air temperature prediction"""
    
    if results is None or 'predictions' not in results:
        print("No valid results to plot")
        return None
        
    # Prepare weather data for plotting (daily means to match other data)
    weather_daily = weather_data.set_index('date')
    weather_daily = weather_daily.resample('D').mean()

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    dates = results['predictions']['dates']
    
    # Plot 1: 60cm depth
    axes[0].plot(dates, results['predictions']['temp_60cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    axes[0].plot(results['predictions_full']['dates_full'], results['predictions_full']['temp_60cm_pred_full'], '-', 
                label='Chambers Model', linewidth=2, color='red')
    axes[0].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Observed Air Temperature', alpha=0.8)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Borehole {borehole} - 60cm depth (R² = {results["diagnostics"]["r2_60cm"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: 120cm depth
    axes[1].plot(dates, results['predictions']['temp_120cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    axes[1].plot(results['predictions_full']['dates_full'], results['predictions_full']['temp_120cm_pred_full'], '-', 
                label='Chambers Model', linewidth=2, color='red')
    axes[1].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Observed Air Temperature', alpha=0.8)
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title(f'Borehole {borehole} - 120cm depth (R² = {results["diagnostics"]["r2_120cm"]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals_60 = results['predictions']['temp_60cm_obs'] - results['predictions']['temp_60cm_pred']
    residuals_120 = results['predictions']['temp_120cm_obs'] - results['predictions']['temp_120cm_pred']
    
    axes[2].plot(dates, residuals_60, 'o-', label='60cm residuals', alpha=0.7, markersize=3)
    axes[2].plot(dates, residuals_120, 's-', label='120cm residuals', alpha=0.7, markersize=3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Residuals (°C)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Model Residuals')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add parameter information
    if 'parameters' in results:
        params = results['parameters']
        param_errors = results['parameter_errors']
        param_text = (
            f"FIXED from weather:\n"
            f"Air mean T: {params['T_weather_mean']:.2f}°C\n"
            f"Air temp amplitude: {params['delta_T_weather']:.2f}°C\n\n"
            f"FITTED parameters:\n"
            f"Damping depth: {params['damping_depth_m']:.3f} ± {param_errors['damping_depth_m']:.3f} m\n"
            f"Phase shift: {params['phase_shift_deg']:.1f}°\n\n"
            f"Overall R²: {results['diagnostics']['r2_overall']:.3f}"
        )
        
        axes[1].text(1.02, 0.5, param_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    
    if not save_figure:
        plt.show()
    
    return fig

def plot_fit_results_improved(results, weather_data, borehole='301', save_figure=False):
    """Improved plotting with weather data and extended predictions"""
    
    if results is None or 'predictions' not in results:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Get data for plotting
    dates = results['predictions']['dates']
    dates_full = results['predictions_full']['dates_full']
    
    # Prepare weather data for plotting (daily means to match other data)
    weather_daily = weather_data.set_index('date')
    weather_daily = weather_daily.resample('D').mean()
    
    # Plot 1: 60cm depth
    axes[0].plot(dates, results['predictions']['temp_60cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    axes[0].plot(dates, results['predictions']['temp_60cm_pred'], 'o', 
                label='Model (fitted)', markersize=2, color='red', alpha=0.8)
    # Plot extended model prediction
    axes[0].plot(dates_full, results['predictions_full']['temp_60cm_pred_full'], '-', 
                label='Model (full range)', linewidth=2, color='red', alpha=0.6)
    # Plot weather data
    axes[0].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Air Temperature', alpha=0.8)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Borehole {borehole} - 60cm depth (R² = {results["diagnostics"]["r2_60cm"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: 120cm depth
    axes[1].plot(dates, results['predictions']['temp_120cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    axes[1].plot(dates, results['predictions']['temp_120cm_pred'], 'o', 
                label='Model (fitted)', markersize=2, color='red', alpha=0.8)
    # Plot extended model prediction
    axes[1].plot(dates_full, results['predictions_full']['temp_120cm_pred_full'], '-', 
                label='Model (full range)', linewidth=2, color='red', alpha=0.6)
    # Plot weather data
    axes[1].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Air Temperature', alpha=0.8)
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title(f'Borehole {borehole} - 120cm depth (R² = {results["diagnostics"]["r2_120cm"]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals_60 = results['predictions']['temp_60cm_obs'] - results['predictions']['temp_60cm_pred']
    residuals_120 = results['predictions']['temp_120cm_obs'] - results['predictions']['temp_120cm_pred']
    
    axes[2].plot(dates, residuals_60, 'o-', label='60cm residuals', alpha=0.7, markersize=3)
    axes[2].plot(dates, residuals_120, 's-', label='120cm residuals', alpha=0.7, markersize=3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Residuals (°C)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Model Residuals')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add parameter information
    if 'parameters' in results:
        params = results['parameters']
        param_errors = results['parameter_errors']
        param_text = (
            f"Surface mean T: {params['T_surface_mean']:.2f} ± {param_errors['T_surface_mean']:.2f}°C\n"
            f"Surface amplitude: {params['delta_T_surface']:.2f} ± {param_errors['delta_T_surface']:.2f}°C\n"
            f"Damping depth: {params['damping_depth_m']:.3f} ± {param_errors['damping_depth_m']:.3f} m\n"
            f"Phase shift: {params['phase_shift_deg']:.1f}°\n"
            f"Overall R²: {results['diagnostics']['r2_overall']:.3f}"
        )
        
        axes[1].text(1.02, 0.5, param_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Synchronize x-axis limits for all plots
    x_min, x_max = dates_full.min(), dates_full.max()
    for ax in axes:
        ax.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    if not save_figure:
        plt.show()
    return fig

def model_equation_txt():
    equation_display = f"""
    {"="*70}
    CHAMBERS MODEL EQUATION & PARAMETERS
    {"="*70}

    Mathematical Model:
    ┌─────────────────────────────────────────────────────────┐
    │  T(z,t) = T̄ + ΔT · exp(-z/d) · sin(ωt + φ - z/d)      │
    └─────────────────────────────────────────────────────────┘

    Parameter Definitions:
    ┌──────────────────┬─────────────────────────────────────────┐
    │    Symbol        │              Description                │
    ├──────────────────┼─────────────────────────────────────────┤
    │    T(z,t)        │  Temperature at depth z and time t     │
    │    T̄             │  Mean annual air temperature           │
    │    ΔT            │  Annual air temperature amplitude      │
    │    z             │  Depth below surface (m)               │
    │    d             │  Damping depth (m) [FITTED]            │
    │    ω             │  Angular frequency = 2π/365.25         │
    │    t             │  Time (days since start of year)       │
    │    φ             │  Phase shift (radians) [FITTED]        │
    │    exp(-z/d)     │  Exponential damping with depth        │
    │    -z/d          │  Phase lag due to depth                │
    └──────────────────┴─────────────────────────────────────────┘

    Model Components:
    • Mean Temperature:     T̄ (from weather data)
    • Amplitude:             ΔT (from weather data)
    • Damping Factor:        exp(-z/d)
    • Seasonal Cycle:        sin(ωt + φ - z/d)
    • Fitted Parameters:     d (damping depth), φ (phase shift)
    \n.\n.\n

    {"="*70}
    """
    return equation_display

def model_equation_txt_improved():
    equation_display = f"""
    {"="*70}
    "IMPROVED" CHAMBERS MODEL EQUATION & PARAMETERS, more live "better working"
    {"="*70}

    Mathematical Model:
    ┌─────────────────────────────────────────────────────────┐
    │  T(z,t) = T̄ + ΔT · exp(-z/d) · sin(ωt + φ - z/d)      │
    └─────────────────────────────────────────────────────────┘

    Parameter Definitions:
    ┌──────────────────┬─────────────────────────────────────────┐
    │    Symbol        │              Description                │
    ├──────────────────┼─────────────────────────────────────────┤
    │    T(z,t)        │  Temperature at depth z and time t     │
    │    T̄             │  Mean measured sensor temperature      │
    │    ΔT            │  Annual measured sensor amplitude      │
    │    z             │  Depth below surface (m)               │
    │    d             │  Damping depth (m) [FITTED]            │
    │    ω             │  Angular frequency = 2π/365.25         │
    │    t             │  Time (days since start of year)       │
    │    φ             │  Phase shift (radians) [FITTED]        │
    │    exp(-z/d)     │  Exponential damping with depth        │
    │    -z/d          │  Phase lag due to depth                │
    └──────────────────┴─────────────────────────────────────────┘

    Model Components:
    • Mean Temperature:      T̄ (from sensor data), then fitted
    • Amplitude:             ΔT (from sensor data), then fitted
    • Damping Factor:        exp(-z/d)
    • Seasonal Cycle:        sin(ωt + φ - z/d)
    • Fitted Parameters:     d (damping depth), φ (phase shift)
    \n.\n.\n

    {"="*70}
    """
    return equation_display

def analyze_all_boreholes(sens_data, improved=False, save_pdf=False, pdf_filename=None):
    """Analyze all three boreholes with original Chambers model and optionally save to PDF"""
    
    results_all = {}
    figures = []
    all_debug_output = []
    if improved:
        all_debug_output.append(model_equation_txt_improved())
    else:
        all_debug_output.append(model_equation_txt())

    for borehole in ['301', '302', '303']:
        print(f"\n{'='*60}")
        print(f"Analyzing Borehole {borehole} - Original Chambers Model")
        print(f"{'='*60}")
        
        try:
            # Capture debug output for this borehole
            if improved:
                results, temp_df = fit_chambers_heat_model_improved(
                    sens_data, 
                    borehole=borehole, 
                    debug=True, 
                    capture_debug=save_pdf
                )
            else:
                results, temp_df = fit_chambers_model_original(
                    sens_data, 
                    borehole=borehole, 
                    debug=True, 
                    capture_debug=save_pdf
                )
            
            if results is not None:
                results_all[borehole] = results
                
                if save_pdf and 'debug_output' in results:
                    all_debug_output.append(f"Analyzing Borehole {borehole} - Original Chambers Model")
                    all_debug_output.append(results['debug_output'])
                
                if 'predictions' in results:  # Full Chambers model worked
                    if improved:
                        fig = plot_fit_results_improved(results, temp_df, borehole, save_figure=save_pdf)
                    else:
                        fig = plot_results(results, temp_df, borehole, save_figure=save_pdf)
                    if save_pdf and fig is not None:
                        figures.append(fig)
                    elif not save_pdf:
                        # Show plot immediately if not saving to PDF
                        plt.show()
                else:
                    print("Using simple sinusoid results - plotting not implemented yet")
            else:
                print(f"Failed to fit original Chambers model for borehole {borehole}")
                
        except Exception as e:
            print(f"Error analyzing borehole {borehole}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save to PDF if requested
    if save_pdf and figures:
        if pdf_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"chambers_analysis_{timestamp}.pdf"
        
        # Create combined debug output
        combined_debug = '\n'.join(all_debug_output)
        
        # Save everything to PDF using the updated function
        try:
            saved_path = saveFiguresToPDF(
                figures, 
                combined_debug,  # Pass debug text directly
                pdf_filename,
                title="Chambers Model Analysis Report",
                verbose=True,
                metadata={
                    'Title': 'Chambers Model Analysis Report',
                    'Author': 'Chambers Model Analysis Script',
                    'Subject': 'Temperature Model Fitting Results'
                }
            )
            print(f"\n{'='*60}")
            print(f"PDF REPORT SAVED: {saved_path}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error saving PDF: {e}")
            # Still show plots if PDF saving failed
            for fig in figures:
                plt.figure(fig.number)
                plt.show()
    
    return results_all

if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS

    sensor_data_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    geophy_drive_path = f'C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/99 - Mémoire -Article/'
    sens_data = load_TDR_data(sensor_data_path)

    # Run original Chambers model analysis with PDF export
    modeled_original = analyze_all_boreholes(
        sens_data, 
        save_pdf=True, 
        improved=True,
        pdf_filename=geophy_drive_path + "chambers_improved_model.pdf"
    )