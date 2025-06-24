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

def load_TDR_data(excel_file_path, maj=False):
    """
    Load data from a single Excel file with multiple sheets and merge them on Timestamp column
    Automatically saves/loads CSV cache for faster subsequent loads
    
    Parameters:
    -----------
    excel_file_path : str
        Path to the Excel file containing multiple sheets
    maj : bool
        If True, forces reload from Excel file even if CSV cache exists
        If False, uses CSV cache if available (default)
    
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with all sheets combined on 'date' column
    """
    
    # Generate CSV cache file path
    base_name = os.path.splitext(excel_file_path)[0]
    csv_file_path = f"{base_name}_merged.csv"
    
    # Check if CSV cache exists and maj parameter
    if not maj and os.path.exists(csv_file_path):
        print(f"Found CSV cache: {csv_file_path}")
        
        # Check if CSV is newer than Excel file
        excel_mtime = os.path.getmtime(excel_file_path)
        csv_mtime = os.path.getmtime(csv_file_path)
        
        if csv_mtime >= excel_mtime:
            print("CSV cache is up-to-date, loading from cache...")
            try:
                df_merged = pd.read_csv(csv_file_path, parse_dates=['date'], sep=';')
                print(f"Loaded from CSV cache:")
                print(f"  - {len(df_merged)} rows, {len(df_merged.columns)} columns")
                print(f"  - Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
                return df_merged
            except Exception as e:
                print(f"Error loading CSV cache: {e}")
                print("Falling back to Excel file processing...")
        else:
            print("Excel file is newer than CSV cache, reprocessing Excel file...")
    elif maj:
        print("Force reload requested (maj=True), processing Excel file...")
    else:
        print("No CSV cache found, processing Excel file...")
    
    # Process Excel file (original logic)
    print(f"\nProcessing Excel file: {excel_file_path}")
    
    # Read all sheet names from the Excel file
    excel_file = pd.ExcelFile(excel_file_path)
    sheet_names = excel_file.sheet_names
    
    print(f"Found {len(sheet_names)} sheets in {excel_file_path}")
    print(f"Sheet names: {sheet_names}")
    
    dfs = []
    
    for sheet_name in sheet_names:
        print(f"Processing sheet: {sheet_name}")
        
        try:
            # Read each sheet, replace "#/NA#" with np.nan
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, na_values=["#/NA"], header=0)
            
            # Check if Timestamp column exists
            if 'Timestamp' not in df.columns:
                print(f"Warning: No 'Timestamp' column found in sheet '{sheet_name}'. Skipping this sheet.")
                continue
            
            # Parse Timestamp with multiple formats
            def parse_date(x):
                if pd.isna(x):
                    return pd.NaT
                for fmt in ["%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]:
                    try:
                        return pd.to_datetime(x, format=fmt)
                    except:
                        continue
                # If no format works, try pandas default parsing
                try:
                    return pd.to_datetime(x)
                except:
                    return pd.NaT
            
            # Convert Timestamp to datetime and rename to 'date'
            df['date'] = df['Timestamp'].apply(parse_date)
            df = df.drop(columns=['Timestamp'])
            
            # Remove rows where date conversion failed
            df = df.dropna(subset=['date'])
            
            if len(df) == 0:
                print(f"Warning: No valid timestamps found in sheet '{sheet_name}'. Skipping this sheet.")
                continue
            
            # Remove empty columns (all NaN or empty)
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.str.match('^Unnamed')]
            
            print(f"  - Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)-1} data columns")
            print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing sheet '{sheet_name}': {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid sheets found with processable data")
    
    # Merge all dataframes on 'date'
    print(f"\nMerging {len(dfs)} sheets...")
    
    df_merged = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        print(f"Merging sheet {i+1}/{len(dfs)}...")
        df_merged = pd.merge(df_merged, df, on='date', how='outer')
    
    # Sort by 'date'
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')
    
    # Move 'date' column to the first position
    cols = df_merged.columns.tolist()
    if 'date' in cols:
        cols.insert(0, cols.pop(cols.index('date')))
        df_merged = df_merged[cols]
    
    print(f"\nMerging complete!")
    print(f"Final dataframe: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    print(f"Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    print(f"Columns: {df_merged.columns.tolist()}")
    
    # Save to CSV cache
    try:
        df_merged.to_csv(csv_file_path, index=False, sep=';')
        print(f"\nSaved CSV cache: {csv_file_path}")
        print(f"CSV file size: {os.path.getsize(csv_file_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Warning: Could not save CSV cache: {e}")
    
    return df_merged

def fetch_and_aggregate_weather(start_date, end_date, temp_step=1, precip_step=24):
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

def model_equation_txt(improved=False):
    txt = f"""
    {"="*70}
    CHAMBERS MODEL EQUATION & PARAMETERS
    {"="*70}

    Mathematical Model:
    ┌─────────────────────────────────────────────────────────┐
    │  T(z,t) = T̄ + ΔT · exp(-z/d) · sin(ωt + φ - z/4*d)      │
    └─────────────────────────────────────────────────────────┘

    Parameter Definitions:
    ┌──────────────────┬─────────────────────────────────────────┐
    │    Symbol        │              Description                │
    ├──────────────────┼─────────────────────────────────────────┤
    │    T(z,t)        │  Temperature at depth z and time t      │
    │    T̄            │  Mean annual air temperature           │
    │    ΔT            │  Annual air temperature amplitude       │
    │    z             │  Depth below surface (m)                │
    │    d             │  Damping depth (m)                      │
    │    ω             │  Angular frequency = 2π/365.25          │
    │    t             │  Time (days since start of year)        │
    │    φ             │  Phase shift (radians)                  │
    │    exp(-z/d)     │  Exponential damping with depth         │ 
    │    -z/4*d        │  Phase lag due to depth                 │
    └──────────────────┴─────────────────────────────────────────┘
    """
    if improved:
        txt += f"""
        Model Components:
        • Mean Temperature:      T̄
        • Amplitude:             ΔT 
        • Damping Factor:        exp(-z/d)
        • Seasonal Cycle:        sin(ωt + φ - z/d)
        • Fitted Parameters:     T̄, ΔT, d, φ
    """
    else:
        txt += f"""
        Model Components:
        • Mean Temperature:     T̄ (from weather data)
        • Amplitude:             ΔT (from weather data)
        • Damping Factor:        exp(-z/d)
        • Seasonal Cycle:        sin(ωt + φ - z/4*d)
        • Fitted Parameters:     d (damping depth), φ (phase shift)
    """
    txt += f"""
    \n.\n.\n

    {"="*70}
    """
    return txt

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
    
def fit_chambers_model(sens_data, borehole='301', model_type='original', hour_step=24, debug=True, capture_debug=False):
    """
    Chambers et al. (2014) heat model with original and improved versions
    
    Parameters:
    -----------
    sens_data : DataFrame
        Sensor data with date column and temperature measurements
    borehole : str
        Borehole identifier ('301', '302', '303')
    model_type : str
        'original' - uses weather data parameters (physically correct)
        'improved' - fits all parameters including surface temp (better fit, less legitimate)
    hour_step : int
        Temporal discretization step in hours (default=24 for daily means)
        6 = 4 times per day, 12 = twice per day, 24 = daily
    debug : bool
        Print debug information
    capture_debug : bool
        Capture debug output for PDF generation
    
    Returns:
    --------
    tuple : (results dict, weather dataframe)
    """
    
    # Set up debug capture if requested
    debug_capture = None
    if capture_debug:
        debug_capture = DebugCapture()
        original_stdout = sys.stdout
        sys.stdout = debug_capture
    
    try:
        if debug:
            print(f"Running {model_type.upper()} Chambers Model for Borehole {borehole}")
            print(f"Temporal discretization: {hour_step}-hour means ({24/hour_step:.1f} points per day)")
            if model_type == 'improved':
                print("WARNING: Improved model fits all parameters (less physically legitimate)")
        
        # =================================================================
        # STEP 1: DATA PREPARATION AND TEMPORAL DISCRETIZATION
        # =================================================================
        
        # Get full date range from sensor data
        full_date_min = sens_data['date'].min()
        full_date_max = sens_data['date'].max()
        
        # Fetch weather data with specified temporal resolution
        temp_df, _ = fetch_and_aggregate_weather(full_date_min, full_date_max, hour_step)
        
        # Calculate weather-based air temperature parameters
        weather_resampled = temp_df.set_index('date').resample(f'{hour_step}h').mean()
        T_weather_mean = weather_resampled['temperature'].mean()
        delta_T_weather = (weather_resampled['temperature'].max() - weather_resampled['temperature'].min())
        
        # Manual overrides for testing (comment out for real weather data)
        if model_type == 'original':
            T_weather_mean = weather_resampled['temperature'].mean()
            delta_T_weather = (weather_resampled['temperature'].max() - weather_resampled['temperature'].min())

        if debug:
            if model_type == 'original':
                print(f"Weather-based air temperature parameters:")
                print(f"  T_weather_mean: {T_weather_mean:.2f}°C")
                print(f"  delta_T_weather: {delta_T_weather:.2f}°C")
            print(f"  Full date range: {full_date_min} to {full_date_max}")
        
        # Prepare sensor data with temporal discretization
        temp_data = sens_data.set_index('date')
        temp_data.index = pd.to_datetime(temp_data.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Get sensor columns for the specified borehole
        col_60 = f'{borehole} - Temp (C) -60cm'
        col_120 = f'{borehole} - Temp (C) -120cm'
        
        # Check if columns exist
        if col_60 not in temp_data.columns or col_120 not in temp_data.columns:
            available_cols = [col for col in temp_data.columns if borehole in col]
            raise ValueError(f"Columns not found. Available columns for {borehole}: {available_cols}")
        
        # Resample sensor data to specified temporal resolution
        resample_rule = f'{hour_step}h'
        temp_60cm = temp_data[col_60].resample(resample_rule).mean()
        temp_120cm = temp_data[col_120].resample(resample_rule).mean()
        
        # Align sensor data on shared dates and drop missing values
        sensor_combined = pd.concat([temp_60cm, temp_120cm], axis=1, keys=['temp60', 'temp120']).dropna()
        
        if debug:
            print(f"Valid sensor data points: {len(sensor_combined)} observations")
            print(f"Sensor date range: {sensor_combined.index.min()} to {sensor_combined.index.max()}")
            print(f"-60cm temp range: {sensor_combined['temp60'].min():.2f} to {sensor_combined['temp60'].max():.2f}°C")
            print(f"-120cm temp range: {sensor_combined['temp120'].min():.2f} to {sensor_combined['temp120'].max():.2f}°C")
        
        # =================================================================
        # STEP 2: TIME INDEXING FOR MODEL FITTING
        # =================================================================
        
        # Create a comprehensive time index for the full date range
        full_date_range = pd.date_range(full_date_min, full_date_max, freq=resample_rule)
        
        # Convert to fractional days since start of year for seasonal cycle alignment
        def date_to_fractional_days(dates):
            """Convert dates to fractional days since start of year"""
            start_of_year = pd.Timestamp(dates.min().year, 1, 1)
            return (dates - start_of_year).total_seconds() / (24 * 3600)  # Convert to fractional days
        
        # Time indices for fitting (only where we have sensor data)
        t_fitting = date_to_fractional_days(sensor_combined.index).values
        
        # Time indices for full prediction range
        t_full = date_to_fractional_days(full_date_range).values
        
        if debug:
            print(f"Time indexing:")
            print(f"  Fitting data points: {len(t_fitting)} (fractional days: {t_fitting.min():.1f} to {t_fitting.max():.1f})")
            print(f"  Full prediction range: {len(t_full)} (fractional days: {t_full.min():.1f} to {t_full.max():.1f})")
        
        # =================================================================
        # STEP 3: PARAMETER ESTIMATION AND MODEL SETUP
        # =================================================================
        
        # Calculate observed parameters for improved model
        if model_type == 'improved':
            obs_mean_60 = sensor_combined['temp60'].mean()
            obs_range_60 = sensor_combined['temp60'].max() - sensor_combined['temp60'].min()
            if debug:
                print(f"Observed sensor statistics:")
                print(f"  60cm: mean={obs_mean_60:.2f}°C, range={obs_range_60:.2f}°C")
                print(f"  120cm: mean={sensor_combined['temp120'].mean():.2f}°C, range={sensor_combined['temp120'].max() - sensor_combined['temp120'].min():.2f}°C")
        
        # Prepare data for fitting - stack depths and temperatures
        depths = np.array([0.6, 1.2])  # depths in meters
        n_times = len(t_fitting)
        n_depths = len(depths)
        
        # Create arrays for all depth-time combinations (for fitting only)
        temps_all = np.concatenate([sensor_combined['temp60'].values, sensor_combined['temp120'].values])
        depths_all = np.repeat(depths, n_times)
        times_all = np.tile(t_fitting, n_depths)
        
        # =================================================================
        # STEP 4: MODEL DEFINITION AND PARAMETER BOUNDS
        # =================================================================
        
        # Define model functions based on type
        if model_type == 'original':
            def chambers_model(inputs, d, phase_shift):
                """
                Original Chambers model with fixed weather parameters
                """
                z, t = inputs
                t_annual = 2 * np.pi * t / 365.25  # Convert to annual radians
                damping = np.exp(-z / d)
                phase_lag = -z / 4 * d
                temp = T_weather_mean + delta_T_weather / 2 * damping * np.sin(t_annual + phase_shift + phase_lag)
                return temp
            
            # Original model parameters and bounds
            initial_guess = [1, 0]  # [damping_depth, phase_shift]
            bounds = ([0.1, -2*np.pi], [10, 2*np.pi])
            
        else:  # improved model
            def chambers_model(inputs, T_surf_mean, delta_T_surf, d, phase_shift):
                """
                Improved Chambers model fitting all parameters
                """
                z, t = inputs
                t_annual = 2 * np.pi * t / 365.25  # Convert to annual radians
                damping = np.exp(-z / d)
                phase_lag = -z / 4 * d  # Modified phase lag
                temp = T_surf_mean + delta_T_surf / 2 * damping * np.sin(t_annual + phase_shift + phase_lag)
                return temp
            
            # Improved model parameters and bounds
            T_surf_mean_est = obs_mean_60
            delta_T_surf_est = obs_range_60 * 2
            initial_guess = [T_surf_mean_est, delta_T_surf_est, 1.0, 0]
            bounds = ([obs_mean_60 - 10, 0.1, 0.1, -2*np.pi], 
                     [obs_mean_60 + 10, 100, 5.0, 2*np.pi])
            
            if debug:
                print(f"Initial parameter estimates:")
                print(f"  T_surf_mean: {T_surf_mean_est:.2f}°C")
                print(f"  delta_T_surf: {delta_T_surf_est:.2f}°C")
        
        # =================================================================
        # STEP 5: MODEL FITTING
        # =================================================================
        
        # Fit the model
        popt, pcov = curve_fit(
            chambers_model, 
            (depths_all, times_all), 
            temps_all, 
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        
         # =================================================================
        # STEP 6: PREDICTIONS AND RESULTS PREPARATION
        # =================================================================
        
        # Generate FULL predictions for both depths over entire time range
        pred_60_full = chambers_model((np.full_like(t_full, 0.6), t_full), *popt)
        pred_120_full = chambers_model((np.full_like(t_full, 1.2), t_full), *popt)
        
        # Generate surface temperature prediction for comparison (z=0)
        pred_surface_full = chambers_model((np.full_like(t_full, 0.0), t_full), *popt)
        
        # Extract predictions at sensor measurement times for validation
        # Find indices in full time array that correspond to sensor measurement times
        sensor_indices = []
        for sensor_time in t_fitting:
            # Find closest time in full array
            idx = np.argmin(np.abs(t_full - sensor_time))
            sensor_indices.append(idx)
        
        pred_60_at_sensors = pred_60_full[sensor_indices]
        pred_120_at_sensors = pred_120_full[sensor_indices]
        
        # Extract fitted parameters based on model type
        if model_type == 'original':
            d_fitted, phase_fitted = popt
            
            if debug:
                print(f"\nFitted parameters:")
                print(f"  Damping depth (d): {d_fitted:.3f} ± {param_errors[0]:.3f} m")
                print(f"  Phase shift: {phase_fitted:.3f} ± {param_errors[1]:.3f} rad ({np.degrees(phase_fitted):.1f}°)")
                print(f"Fixed parameters:")
                print(f"  Air mean temp: {T_weather_mean:.2f}°C (from weather)")
                print(f"  Air temp amplitude: {delta_T_weather:.2f}°C (from weather)")
            
            # Parameters dictionary for original model
            parameters = {
                'T_weather_mean': T_weather_mean,
                'delta_T_weather': delta_T_weather,
                'damping_depth_m': d_fitted,
                'phase_shift_rad': phase_fitted,
                'phase_shift_deg': np.degrees(phase_fitted)
            }
            parameter_errors_dict = {
                'damping_depth_m': param_errors[0],
                'phase_shift_rad': param_errors[1]
            }
            parameter_source = {
                'T_weather_mean': 'weather_data',
                'delta_T_weather': 'weather_data', 
                'damping_depth_m': 'fitted',
                'phase_shift_rad': 'fitted'
            }
            
        else:  # improved model
            T_surf_fitted, delta_T_fitted, d_fitted, phase_fitted = popt
            
            if debug:
                print(f"\nFitted parameters:")
                print(f"  Surface mean temp: {T_surf_fitted:.3f} ± {param_errors[0]:.3f}°C")
                print(f"  Surface temp amplitude: {delta_T_fitted:.3f} ± {param_errors[1]:.3f}°C")
                print(f"  Damping depth (d): {d_fitted:.3f} ± {param_errors[2]:.3f} m")
                print(f"  Phase shift: {phase_fitted:.3f} ± {param_errors[3]:.3f} rad ({np.degrees(phase_fitted):.1f}°)")
            
            # Parameters dictionary for improved model
            parameters = {
                'T_surface_mean': T_surf_fitted,
                'delta_T_surface': delta_T_fitted,
                'damping_depth_m': d_fitted,
                'phase_shift_rad': phase_fitted,
                'phase_shift_deg': np.degrees(phase_fitted)
            }
            parameter_errors_dict = {
                'T_surface_mean': param_errors[0],
                'delta_T_surface': param_errors[1],
                'damping_depth_m': param_errors[2],
                'phase_shift_rad': param_errors[3]
            }
            parameter_source = {
                'T_surface_mean': 'fitted',
                'delta_T_surface': 'fitted', 
                'damping_depth_m': 'fitted',
                'phase_shift_rad': 'fitted'
            }
        
        # =================================================================
        # STEP 7: MODEL PERFORMANCE METRICS
        # =================================================================
        
        # Calculate R² and RMSE for each depth (using sensor measurement times)
        r2_60 = 1 - np.sum((sensor_combined['temp60'] - pred_60_at_sensors)**2) / np.sum((sensor_combined['temp60'] - sensor_combined['temp60'].mean())**2)
        r2_120 = 1 - np.sum((sensor_combined['temp120'] - pred_120_at_sensors)**2) / np.sum((sensor_combined['temp120'] - sensor_combined['temp120'].mean())**2)
        
        rmse_60 = np.sqrt(np.mean((sensor_combined['temp60'] - pred_60_at_sensors)**2))
        rmse_120 = np.sqrt(np.mean((sensor_combined['temp120'] - pred_120_at_sensors)**2))
        
        # Overall metrics
        all_observed = np.concatenate([sensor_combined['temp60'].values, sensor_combined['temp120'].values])
        all_predicted = np.concatenate([pred_60_at_sensors, pred_120_at_sensors])
        r2_overall = 1 - np.sum((all_observed - all_predicted)**2) / np.sum((all_observed - all_observed.mean())**2)
        rmse_overall = np.sqrt(np.mean((all_observed - all_predicted)**2))
        
        if debug:
            print(f"\nModel performance:")
            print(f"  60cm depth - R²: {r2_60:.3f}, RMSE: {rmse_60:.3f}°C")
            print(f"  120cm depth - R²: {r2_120:.3f}, RMSE: {rmse_120:.3f}°C")
            print(f"  Overall - R²: {r2_overall:.3f}, RMSE: {rmse_overall:.3f}°C")
        
        # =================================================================
        # STEP 8: FINAL RESULTS ASSEMBLY
        # =================================================================
        
        # Prepare results
        results = {
            'model_type': model_type,
            'hour_step': hour_step,
            'parameters': parameters,
            'parameter_errors': parameter_errors_dict,
            'parameter_source': parameter_source,
            'predictions': {
                'dates': sensor_combined.index,
                'temp_60cm_obs': sensor_combined['temp60'],
                'temp_60cm_pred': pred_60_at_sensors,
                'temp_120cm_obs': sensor_combined['temp120'],
                'temp_120cm_pred': pred_120_at_sensors,
                'time_days': t_fitting
            },
            'predictions_full': {
                'dates_full': full_date_range,
                'temp_surface_pred_full': pred_surface_full,
                'temp_60cm_pred_full': pred_60_full,
                'temp_120cm_pred_full': pred_120_full,
                'time_days_full': t_full
            },
            'diagnostics': {
                'r2_60cm': r2_60,
                'r2_120cm': r2_120,
                'r2_overall': r2_overall,
                'rmse_60cm': rmse_60,
                'rmse_120cm': rmse_120,
                'rmse_overall': rmse_overall,
                'n_observations': len(sensor_combined),
                'temporal_resolution': f'{hour_step}H'
            },
            #'model_function': chambers_model,
            'fitted_params': popt
        }
        
        # Add model-specific parameters
        if model_type == 'original':
            results['weather_params'] = {
                'T_weather_mean': T_weather_mean,
                'delta_T_weather': delta_T_weather
            }
        
        # Add debug capture to results if available
        if capture_debug and debug_capture:
            results['debug_output'] = debug_capture.get_output()
        
        return results, temp_df
        
    except Exception as e:
        print(f"{model_type.capitalize()} Chambers fitting failed: {str(e)}")
        return None, None
    finally:
        # Restore stdout if it was captured
        if capture_debug and debug_capture:
            sys.stdout = original_stdout

# Wrapper functions for backward compatibility
def fit_chambers_model_original(sens_data, borehole='301', hour_step=24, debug=True, capture_debug=False):
    """Original Chambers model using weather data parameters"""
    return fit_chambers_model(sens_data, borehole, 'original', hour_step, debug, capture_debug)

def fit_chambers_model_improved(sens_data, borehole='301', hour_step=24, debug=True, capture_debug=False):
    """Improved Chambers model fitting all parameters"""
    return fit_chambers_model(sens_data, borehole, 'improved', hour_step, debug, capture_debug)

def plot_results(results, weather_data, borehole='301', save_figure=False, show_fitted_points=False):
    """
    Plot results from Chambers model with weather data and predictions
    
    Parameters:
    -----------
    results : dict
        Model results containing predictions and parameters
    weather_data : DataFrame
        Weather data for comparison
    borehole : str
        Borehole identifier for title
    save_figure : bool
        Whether to save figure or show it
    show_fitted_points : bool
        If True, shows both fitted points and full range predictions
        If False, shows only full range predictions (cleaner)
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for saving to PDF
    """
    
    if results is None or 'predictions' not in results:
        print("No valid results to plot")
        return None
    
    # Determine model type for adaptive plotting
    model_type = results.get('model_type', 'original')
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Get data for plotting
    dates = results['predictions']['dates']
    dates_full = results['predictions_full']['dates_full']
    
    # Prepare weather data for plotting (daily means to match other data)
    weather_daily = weather_data.set_index('date')
    weather_daily = weather_daily.resample('D').mean()
    
    # =================================================================
    # PLOT 1: 60cm depth
    # =================================================================
    
    # Observed data
    axes[0].plot(dates, results['predictions']['temp_60cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    
    # Model predictions
    if show_fitted_points:
        # Show both fitted points and full range
        axes[0].plot(dates, results['predictions']['temp_60cm_pred'], 'o', 
                    label='Model (fitted)', markersize=2, color='red', alpha=0.8)
        axes[0].plot(dates_full, results['predictions_full']['temp_60cm_pred_full'], '-', 
                    label='Model (full range)', linewidth=2, color='red', alpha=0.6)
    else:
        # Show only full range (cleaner)
        axes[0].plot(dates_full, results['predictions_full']['temp_60cm_pred_full'], '-', 
                    label='Chambers Model', linewidth=2, color='red')
    
    # Weather/surface temperature
    if 'temp_surface_pred_full' in results['predictions_full']:
        axes[0].plot(dates_full, results['predictions_full']['temp_surface_pred_full'], '--', 
                    color='green', linewidth=1.5, label='Model Surface', alpha=0.8)
    
    # Observed air temperature
    axes[0].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Observed Air Temperature', alpha=0.8)
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Borehole {borehole} - 60cm depth (R² = {results["diagnostics"]["r2_60cm"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Remove x-axis labels for top plot
    axes[0].tick_params(axis='x', labelbottom=False)
    
    # =================================================================
    # PLOT 2: 120cm depth
    # =================================================================
    
    # Observed data
    axes[1].plot(dates, results['predictions']['temp_120cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    
    # Model predictions
    if show_fitted_points:
        # Show both fitted points and full range
        axes[1].plot(dates, results['predictions']['temp_120cm_pred'], 'o', 
                    label='Model (fitted)', markersize=2, color='red', alpha=0.8)
        axes[1].plot(dates_full, results['predictions_full']['temp_120cm_pred_full'], '-', 
                    label='Model (full range)', linewidth=2, color='red', alpha=0.6)
    else:
        # Show only full range (cleaner)
        axes[1].plot(dates_full, results['predictions_full']['temp_120cm_pred_full'], '-', 
                    label='Chambers Model', linewidth=2, color='red')
    
    # Weather/surface temperature
    if 'temp_surface_pred_full' in results['predictions_full']:
        axes[1].plot(dates_full, results['predictions_full']['temp_surface_pred_full'], '--', 
                    color='green', linewidth=1.5, label='Model Surface', alpha=0.8)
    
    # Observed air temperature
    axes[1].plot(weather_daily.index, weather_daily['temperature'], '-', 
                color='orange', linewidth=1.5, label='Observed Air Temperature', alpha=0.8)
    
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title(f'Borehole {borehole} - 120cm depth (R² = {results["diagnostics"]["r2_120cm"]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Remove x-axis labels for middle plot
    axes[1].tick_params(axis='x', labelbottom=False)
    
    # =================================================================
    # PLOT 3: Residuals
    # =================================================================
    
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
    
    # =================================================================
    # PARAMETER INFORMATION BOX
    # =================================================================
    
    if 'parameters' in results:
        params = results['parameters']
        param_errors = results['parameter_errors']
        
        # Create parameter text based on model type
        if model_type == 'original':
            param_text = (
                f"FIXED from weather:\n"
                f"Air mean T: {params.get('T_weather_mean', 'N/A'):.2f}°C\n"
                f"Air temp amplitude: {params.get('delta_T_weather', 'N/A'):.2f}°C\n\n"
                f"FITTED parameters:\n"
                f"Damping depth: {params['damping_depth_m']:.3f} ± {param_errors['damping_depth_m']:.3f} m\n"
                f"Phase shift: {params['phase_shift_deg']:.1f}°\n\n"
                f"Overall R²: {results['diagnostics']['r2_overall']:.3f}\n"
                f"Model: {model_type.upper()}"
            )
        else:  # improved model
            param_text = (
                f"FITTED parameters:\n"
                f"Surface mean T: {params['T_surface_mean']:.2f} ± {param_errors['T_surface_mean']:.2f}°C\n"
                f"Surface amplitude: {params['delta_T_surface']:.2f} ± {param_errors['delta_T_surface']:.2f}°C\n"
                f"Damping depth: {params['damping_depth_m']:.3f} ± {param_errors['damping_depth_m']:.3f} m\n"
                f"Phase shift: {params['phase_shift_deg']:.1f}°\n\n"
                f"Overall R²: {results['diagnostics']['r2_overall']:.3f}\n"
                f"Model: {model_type.upper()}"
            )
        
        # Add temporal resolution info if available
        if 'hour_step' in results:
            param_text += f"\nResolution: {results['hour_step']}h"
        
        # Place parameter box
        box_color = 'lightblue' if model_type == 'original' else 'lightgreen'
        axes[1].text(1.02, 0.5, param_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
    
    # =================================================================
    # FINAL FORMATTING
    # =================================================================
    
    # Synchronize x-axis limits for all plots
    x_min, x_max = dates_full.min(), dates_full.max()
    for ax in axes:
        ax.set_xlim(x_min, x_max)
    
    # Adjust layout to accommodate parameter box
    plt.tight_layout()
    
    # Show or return figure
    if not save_figure:
        plt.show()
    
    return fig

def analyze_all_boreholes(sens_data, data_to_analyze = ['BB - 301', 'BB - 302', 'BB - 303'], hour_step=24, improved=False, save_pdf=True, pdf_filename=None):
    """Analyze all three boreholes with original Chambers model and optionally save to PDF"""
    
    results_all = {}
    figures = []
    all_debug_output = []
    all_debug_output.append(model_equation_txt(improved=improved))

    for borehole in data_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing Borehole {borehole} - Original Chambers Model")
        print(f"{'='*60}")
        
        try:
            # Capture debug output for this borehole
            if improved:
                results, temp_df = fit_chambers_model_improved(
                    sens_data, 
                    borehole=borehole, 
                    hour_step=hour_step,
                    debug=True, 
                    capture_debug=save_pdf
                )
            else:
                results, temp_df = fit_chambers_model_original(
                    sens_data, 
                    borehole=borehole, 
                    hour_step=hour_step,
                    debug=True, 
                    capture_debug=save_pdf
                )
            
            if results is not None:
                results_all[borehole] = results
                
                if save_pdf and 'debug_output' in results:
                    all_debug_output.append(f"Analyzing Borehole {borehole} - Original Chambers Model")
                    all_debug_output.append(results['debug_output'])
                
                if 'predictions' in results:  # Full Chambers model worked
                    fig = plot_results(results, temp_df, borehole, save_figure=save_pdf)
                    if save_pdf and fig is not None:
                        figures.append(fig)
                    elif not save_pdf:
                        # Show plot immediately if not saving to PDF
                        plt.show()
            else:
                print(f"Failed to fit model for borehole {borehole}")
                
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

def save_model_results(results, output_path):
    import pickle
    pickle_path = f"{output_path}_model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved model results: {pickle_path}")

if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS

    one_drive_path = f'C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/'

    sens_data = load_TDR_data(one_drive_path + 'Projet_IV_TDR_Data.xlsx', maj=False)

    pdf_save_path = one_drive_path + f'99 - Mémoire -Article/' + "BB_improved_model.pdf"

    # to process CG data
    #data_to_analyze = ['CG - 301', 'CG - 302']
    # to process WM data
    #data_to_analyze = ['WM - 301', 'WM - 302']

    # Run original Chambers model analysis with PDF export
    results = analyze_all_boreholes(
        sens_data, 
        #hour_step=6,
        #data_to_analyze=data_to_analyze,
        improved=True,
        pdf_filename=pdf_save_path 
    )

    save_model_results(results, one_drive_path + f'99 - Mémoire -Article/')