import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from load_display_TDR_data import fetch_and_aggregate_weather, load_TDR_data

def fit_chambers_model_original(sens_data, borehole='301', debug=True):
    """
    Original Chambers et al. (2014) heat model using weather data for surface temperature parameters
    """
    
    # Get weather data for the FULL sensor data range
    full_date_min = sens_data['date'].min()
    full_date_max = sens_data['date'].max()
    temp_df, _ = fetch_and_aggregate_weather(full_date_min, full_date_max, 24)
    
    # Calculate weather-based surface temperature parameters from full dataset
    weather_daily = temp_df.set_index('date').resample('D').mean()
    T_surf_weather = weather_daily['temperature'].mean()
    delta_T_surf_weather = (weather_daily['temperature'].max() - weather_daily['temperature'].min()) / 2  # Amplitude, not range
    
    if debug:
        print(f"Weather-based surface parameters:")
        print(f"  T_surface_mean: {T_surf_weather:.2f}°C")
        print(f"  T_surface_amplitude: {delta_T_surf_weather:.2f}°C")
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
        print(f"60cm temp range: {combined['temp60'].min():.2f} to {combined['temp60'].max():.2f}°C")
        print(f"120cm temp range: {combined['temp120'].min():.2f} to {combined['temp120'].max():.2f}°C")
    
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
        Original Chambers et al. heat conduction model with fixed surface parameters from weather
        
        Parameters:
        -----------
        d : damping depth (m) - ONLY parameter to fit along with phase
        phase_shift : phase shift (radians) - ONLY parameter to fit along with damping depth
        
        Fixed parameters from weather data:
        T_surf_mean : surface mean temperature (from weather)
        delta_T_surf : surface temperature amplitude (from weather)
        """
        z, t = inputs
        
        # Convert time to annual cycle (t in days since start of year)
        t_annual = 2 * np.pi * t / 365.25  # Convert to radians for the year
        
        # Damping factor (exponential decay with depth)
        damping = np.exp(-z / d)
        
        # Phase lag due to depth (delay increases with depth)
        phase_lag = -z / d
        
        # Temperature model using WEATHER-BASED surface parameters
        temp = T_surf_weather + delta_T_surf_weather * damping * np.sin(t_annual + phase_shift + phase_lag)
        
        return temp
    
    # Initial parameter estimation - only for damping depth and phase shift
    # Estimate damping depth from amplitude reduction between depths
    obs_range_60 = combined['temp60'].max() - combined['temp60'].min()
    obs_range_120 = combined['temp120'].max() - combined['temp120'].min()
    
    if obs_range_60 > 0 and obs_range_120 > 0:
        damping_ratio = obs_range_120 / obs_range_60
        if damping_ratio > 0 and damping_ratio < 1:
            d_est = (1.2 - 0.6) / np.log(1/damping_ratio)
        else:
            d_est = 1.0
    else:
        d_est = 1.0
    
    # Phase shift estimation (start with spring = 0)
    phase_shift_est = 0
    
    initial_guess = [d_est, phase_shift_est]
    
    if debug:
        print(f"Initial parameter estimates (fitting only):")
        print(f"  Damping depth: {d_est:.3f} m")
        print(f"  Phase shift: {phase_shift_est:.3f} rad")
        print(f"Fixed weather parameters:")
        print(f"  T_surf_mean: {T_surf_weather:.2f}°C (from weather)")
        print(f"  delta_T_surf: {delta_T_surf_weather:.2f}°C (from weather)")
    
    # Parameter bounds - only for the two fitted parameters
    bounds = (
        [0.1, -2*np.pi],  # Lower bounds: [d_min, phase_min]
        [5.0, 2*np.pi]    # Upper bounds: [d_max, phase_max]
    )
    
    try:
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
            print(f"  Surface mean temp: {T_surf_weather:.2f}°C (from weather)")
            print(f"  Surface amplitude: {delta_T_surf_weather:.2f}°C (from weather)")
        
        # Generate predictions for both depths - FOR FITTING DATES
        pred_60 = chambers_model_original((np.full_like(t_days, 0.6), t_days), *popt)
        pred_120 = chambers_model_original((np.full_like(t_days, 1.2), t_days), *popt)
        
        # Generate predictions for FULL DATE RANGE (for plotting)
        full_date_range = pd.date_range(full_date_min, full_date_max, freq='D')
        start_of_year_full = pd.Timestamp(full_date_range.min().year, 1, 1)
        t_days_full = (full_date_range - start_of_year_full).days.values.astype(float)
        
        pred_60_full = chambers_model_original((np.full_like(t_days_full, 0.6), t_days_full), *popt)
        pred_120_full = chambers_model_original((np.full_like(t_days_full, 1.2), t_days_full), *popt)
        
        # Also generate surface temperature prediction for comparison
        pred_surface_full = T_surf_weather + delta_T_surf_weather * np.sin(2 * np.pi * t_days_full / 365.25 + phase_fitted)
        
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
                'T_surface_mean': T_surf_weather,  # Fixed from weather
                'delta_T_surface': delta_T_surf_weather,  # Fixed from weather
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
                'T_surface_mean': 'weather_data',
                'delta_T_surface': 'weather_data', 
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
                'temp_surface_pred_full': pred_surface_full,
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
                'T_surface_mean': T_surf_weather,
                'delta_T_surface': delta_T_surf_weather
            }
        }
        
        return results, temp_df
        
    except Exception as e:
        print(f"Original Chambers fitting failed: {str(e)}")
        print("Trying simpler approach...")
        return None

def plot_original_chambers_results(results, weather_data, borehole='301'):
    """Plot results from original Chambers model with weather and surface prediction"""
    
    if results is None or 'predictions' not in results:
        print("No valid results to plot")
        return
    # Prepare weather data for plotting (daily means to match other data)
    weather_daily = weather_data.set_index('date')
    weather_daily = weather_daily.resample('D').mean()

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    dates = results['predictions']['dates']
    
    # Plot 1: 60cm depth
    axes[0].plot(dates, results['predictions']['temp_60cm_obs'], 'o', 
                label='Observed', alpha=0.7, markersize=3, color='blue')
    axes[0].plot(dates, results['predictions']['temp_60cm_pred'], '-', 
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
    axes[1].plot(dates, results['predictions']['temp_120cm_pred'], '-', 
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
            f"Surface mean T: {params['T_surface_mean']:.2f}°C\n"
            f"Surface amplitude: {params['delta_T_surface']:.2f}°C\n\n"
            f"FITTED parameters:\n"
            f"Damping depth: {params['damping_depth_m']:.3f} ± {param_errors['damping_depth_m']:.3f} m\n"
            f"Phase shift: {params['phase_shift_deg']:.1f}°\n\n"
            f"Overall R²: {results['diagnostics']['r2_overall']:.3f}"
        )
        
        axes[1].text(1.02, 0.5, param_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.show()

# Modified main analysis function for original Chambers model
def analyze_all_boreholes_original_chambers(sens_data):
    """Analyze all three boreholes with original Chambers model"""
    
    results_all = {}
    
    for borehole in ['301', '302', '303']:
        print(f"\n{'='*60}")
        print(f"Analyzing Borehole {borehole} - Original Chambers Model")
        print(f"{'='*60}")
        
        try:
            results, temp_df = fit_chambers_model_original(sens_data, borehole=borehole, debug=True)
            if results is not None:
                results_all[borehole] = results
                if 'predictions' in results:  # Full Chambers model worked
                    plot_original_chambers_results(results, temp_df, borehole)
                else:  # Simple fallback model
                    print("Using simple sinusoid results - plotting not implemented yet")
            else:
                print(f"Failed to fit original Chambers model for borehole {borehole}")
        except Exception as e:
            print(f"Error analyzing borehole {borehole}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results_all

if __name__ == '__main__':

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    
    sens_data = load_TDR_data(Onedrive_path)

    # Run original Chambers model analysis
    modeled_original = analyze_all_boreholes_original_chambers(sens_data)