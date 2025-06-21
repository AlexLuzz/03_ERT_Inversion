import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from load_display_TDR_data import fetch_and_aggregate_weather, load_TDR_data

def fit_improved_chambers_then_hybrid(sens_data, borehole='301', depth=60, debug=True):
    """
    Two-stage fitting:
    1. First fit improved Chambers model (d_chambers, phase_shift)
    2. Then fit hybrid parameters (ratio_r, w_window) using Chambers as baseline
    
    Parameters:
    -----------
    sens_data : pd.DataFrame
        DataFrame with columns: 'date', and temperature columns for each sensor
    borehole : str
        Which borehole to analyze ('301', '302', or '303')
    depth : int
        Depth in cm (60 or 120)
    debug : bool
        Whether to print debugging information
    
    Returns:
    --------
    dict with fitted parameters, modeled temperatures, and diagnostics
    """
    
    # Get weather data for the entire period
    temp_df, _ = fetch_and_aggregate_weather(sens_data['date'].min(), sens_data['date'].max(), 24)
    
    # Set datetime index for sensor data
    temp_data = sens_data.set_index('date')
    temp_data.index = pd.to_datetime(temp_data.index)
    
    # Get sensor column for the specified borehole and depth
    col_name = f'{borehole} - Temp (°C) -{depth}cm '
    
    # Check if column exists
    if col_name not in temp_data.columns:
        available_cols = [col for col in temp_data.columns if borehole in col]
        raise ValueError(f"Column not found: {col_name}. Available columns for {borehole}: {available_cols}")
    
    # Resample sensor data to daily means
    temp_sensor = temp_data[col_name].resample('D').mean()
    
    # Prepare weather data with datetime index
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df = temp_df.set_index('date')
    weather_daily = temp_df['temperature'].resample('D').mean()
    
    # Align all data on shared dates and drop missing values
    combined = pd.concat([temp_sensor, weather_daily], 
                        axis=1, keys=['temp_sensor', 'air_temp']).dropna()
    
    if len(combined) < 10:
        raise ValueError(f"Insufficient data after cleaning: only {len(combined)} valid days")
    
    # Calculate T_mean and delta_T from sensor data
    T_mean_sensor = combined['temp_sensor'].mean()
    delta_T_sensor = combined['temp_sensor'].max() - combined['temp_sensor'].min()
    
    if debug:
        print(f"Sensor-derived parameters - T_mean: {T_mean_sensor:.2f}°C, delta_T: {delta_T_sensor:.2f}°C")
        print(f"Valid data points: {len(combined)} days")
        print(f"Date range: {combined.index.min()} to {combined.index.max()}")
    
    # Convert to days since start of year for better seasonal alignment
    start_of_year = pd.Timestamp(combined.index.min().year, 1, 1)
    t_days = (combined.index - start_of_year).days.values.astype(float)
    
    # Convert depth to meters
    depth_m = depth / 100.0
    
    # ========== STAGE 1: FIT IMPROVED CHAMBERS MODEL ==========
    def chambers_temp_model(t, d_chambers, phase_shift):
        """
        Improved Chambers model using sensor-derived T_mean and delta_T
        """
        # Convert time to annual cycle
        t_annual = 2 * np.pi * t / 365.25
        
        # Damping factor (exponential decay with depth)
        damping = np.exp(-depth_m / d_chambers)
        
        # Phase lag due to depth (delay increases with depth)
        phase_lag = -depth_m / d_chambers
        
        # Temperature model
        temp_chambers = T_mean_sensor + delta_T_sensor * damping * np.sin(t_annual + phase_shift + phase_lag)
        
        return temp_chambers
    
    # Initial guesses for Chambers model
    chambers_initial_guesses = [
        [0.5, 0],
        [1.0, np.pi/4],
        [0.3, -np.pi/6],
        [1.5, np.pi/2],
        [0.8, -np.pi/4],
    ]
    
    # Bounds for Chambers parameters
    chambers_bounds = ([0.1, -2*np.pi], [3.0, 2*np.pi])
    
    best_chambers_result = None
    best_chambers_rmse = np.inf
    
    if debug:
        print(f"\n=== STAGE 1: Fitting Improved Chambers Model ===")
    
    for i, initial_guess in enumerate(chambers_initial_guesses):
        try:
            if debug:
                print(f"  Trying Chambers guess {i+1}: d={initial_guess[0]:.2f}, phase={initial_guess[1]:.2f}")
            
            popt, pcov = curve_fit(
                chambers_temp_model, 
                t_days, 
                combined['temp_sensor'].values, 
                p0=initial_guess,
                bounds=chambers_bounds,
                maxfev=10000
            )
            
            pred_temp = chambers_temp_model(t_days, *popt)
            current_rmse = np.sqrt(np.mean((combined['temp_sensor'] - pred_temp)**2))
            
            if current_rmse < best_chambers_rmse:
                best_chambers_rmse = current_rmse
                best_chambers_result = (popt, pcov)
                if debug:
                    print(f"    → New best Chambers RMSE: {current_rmse:.4f}°C")
            
        except Exception as e:
            if debug:
                print(f"    → Failed: {str(e)}")
            continue
    
    if best_chambers_result is None:
        raise ValueError("Chambers model fitting failed")
    
    chambers_popt, chambers_pcov = best_chambers_result
    d_chambers_fitted, phase_chambers_fitted = chambers_popt
    chambers_param_errors = np.sqrt(np.diag(chambers_pcov))
    
    # Calculate Chambers performance
    chambers_pred = chambers_temp_model(t_days, *chambers_popt)
    chambers_r2 = 1 - np.sum((combined['temp_sensor'] - chambers_pred)**2) / np.sum((combined['temp_sensor'] - combined['temp_sensor'].mean())**2)
    chambers_rmse = np.sqrt(np.mean((combined['temp_sensor'] - chambers_pred)**2))
    
    if debug:
        print(f"\nChambers Model Results:")
        print(f"  Damping depth: {d_chambers_fitted:.3f} ± {chambers_param_errors[0]:.3f} m")
        print(f"  Phase shift: {np.degrees(phase_chambers_fitted):.1f}°")
        print(f"  R²: {chambers_r2:.3f}")
        print(f"  RMSE: {chambers_rmse:.3f}°C")
    
    # ========== STAGE 2: FIT HYBRID MODEL ==========
    def hybrid_temp_model(t, ratio_r, w_window):
        """
        Hybrid model using pre-fitted Chambers parameters
        T_hybrid = r * sliding_air_temp + (1-r) * T_chambers
        """
        w_window = max(1, min(30, int(w_window)))
        
        # Use pre-fitted Chambers model
        temp_chambers = chambers_temp_model(t, d_chambers_fitted, phase_chambers_fitted)
        
        # Calculate sliding window air temperature
        air_temp_values = combined['air_temp'].values
        air_temp_sliding = np.zeros_like(t)
        
        for i, time_point in enumerate(t):
            time_idx = int(np.round(time_point - t[0]))
            time_idx = max(0, min(len(air_temp_values) - 1, time_idx))
            
            window_start = max(0, time_idx - w_window + 1)
            window_end = min(len(air_temp_values), time_idx + 1)
            
            if window_end > window_start:
                air_temp_sliding[i] = np.mean(air_temp_values[window_start:window_end])
            else:
                air_temp_sliding[i] = air_temp_values[time_idx]
        
        # Hybrid combination
        temp_hybrid = ratio_r * air_temp_sliding + (1 - ratio_r) * temp_chambers
        
        return temp_hybrid
    
    # Initial guesses for hybrid parameters [ratio_r, w_window]
    hybrid_initial_guesses = [
        [0.1, 3],   # Small weather influence, short window
        [0.2, 7],   # Small weather influence, medium window
        [0.3, 5],   # Medium weather influence
        [0.1, 10],  # Small weather influence, long window
        [0.15, 2],  # Very small weather influence, very short window
    ]
    
    # Bounds for hybrid parameters
    hybrid_bounds = ([0.0, 1], [0.5, 30])  # Limit ratio to max 0.5 to prevent overpowering Chambers
    
    best_hybrid_result = None
    best_hybrid_rmse = np.inf
    
    if debug:
        print(f"\n=== STAGE 2: Fitting Hybrid Parameters ===")
    
    for i, initial_guess in enumerate(hybrid_initial_guesses):
        try:
            if debug:
                print(f"  Trying hybrid guess {i+1}: r={initial_guess[0]:.2f}, w={initial_guess[1]:.0f}")
            
            popt, pcov = curve_fit(
                hybrid_temp_model, 
                t_days, 
                combined['temp_sensor'].values, 
                p0=initial_guess,
                bounds=hybrid_bounds,
                maxfev=10000
            )
            
            pred_temp = hybrid_temp_model(t_days, *popt)
            current_rmse = np.sqrt(np.mean((combined['temp_sensor'] - pred_temp)**2))
            
            if current_rmse < best_hybrid_rmse:
                best_hybrid_rmse = current_rmse
                best_hybrid_result = (popt, pcov)
                if debug:
                    print(f"    → New best hybrid RMSE: {current_rmse:.4f}°C")
            
        except Exception as e:
            if debug:
                print(f"    → Failed: {str(e)}")
            continue
    
    # Decide whether to use hybrid or stick with Chambers
    use_hybrid = False
    if best_hybrid_result is not None:
        hybrid_popt, hybrid_pcov = best_hybrid_result
        r_fitted, w_fitted = hybrid_popt
        hybrid_param_errors = np.sqrt(np.diag(hybrid_pcov))
        
        hybrid_pred = hybrid_temp_model(t_days, *hybrid_popt)
        hybrid_r2 = 1 - np.sum((combined['temp_sensor'] - hybrid_pred)**2) / np.sum((combined['temp_sensor'] - combined['temp_sensor'].mean())**2)
        hybrid_rmse = np.sqrt(np.mean((combined['temp_sensor'] - hybrid_pred)**2))
        
        # Use hybrid only if it's significantly better
        improvement_threshold = 0.001  # R² must improve by at least 0.001
        if hybrid_r2 > chambers_r2 + improvement_threshold:
            use_hybrid = True
            if debug:
                print(f"\nHybrid Model Results:")
                print(f"  Ratio (r): {r_fitted:.3f} ± {hybrid_param_errors[0]:.3f}")
                print(f"  Window (w): {w_fitted:.1f} ± {hybrid_param_errors[1]:.1f} days")
                print(f"  R²: {hybrid_r2:.3f} (improvement: +{hybrid_r2-chambers_r2:.3f})")
                print(f"  RMSE: {hybrid_rmse:.3f}°C")
                print(f"  → Using HYBRID model")
        else:
            if debug:
                print(f"\nHybrid model did not improve significantly")
                print(f"  Chambers R²: {chambers_r2:.3f}")
                print(f"  Hybrid R²: {hybrid_r2:.3f} (improvement: +{hybrid_r2-chambers_r2:.3f})")
                print(f"  → Using CHAMBERS model")
    else:
        if debug:
            print(f"\nHybrid fitting failed → Using CHAMBERS model")
    
    # ========== GENERATE FULL YEAR PREDICTIONS ==========
    # Create full year time array
    year = combined.index.min().year
    full_year_start = pd.Timestamp(year, 1, 1)
    full_year_end = pd.Timestamp(year, 12, 31)
    full_year_dates = pd.date_range(full_year_start, full_year_end, freq='D')
    full_year_t_days = (full_year_dates - full_year_start).days.values.astype(float)
    
    # Get weather data for full year
    weather_full_year = weather_daily.reindex(full_year_dates)
    weather_full_year = weather_full_year.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # Generate full year Chambers prediction
    chambers_full_year = chambers_temp_model(full_year_t_days, d_chambers_fitted, phase_chambers_fitted)
    
    # Generate full year hybrid prediction if using hybrid
    if use_hybrid:
        # Calculate sliding window for full year
        weather_full_year_values = weather_full_year.values
        w_fitted_int = max(1, int(w_fitted))
        sliding_air_temp_full_year = np.zeros_like(full_year_t_days, dtype=float)
        
        for i, time_point in enumerate(full_year_t_days):
            time_idx = int(np.round(time_point))
            window_start = max(0, time_idx - w_fitted_int + 1)
            window_end = min(len(weather_full_year_values), time_idx + 1)
            
            if window_end > window_start:
                sliding_air_temp_full_year[i] = np.mean(weather_full_year_values[window_start:window_end])
            else:
                sliding_air_temp_full_year[i] = weather_full_year_values[min(time_idx, len(weather_full_year_values)-1)]
        
        hybrid_full_year = r_fitted * sliding_air_temp_full_year + (1 - r_fitted) * chambers_full_year
        final_pred = hybrid_pred
        final_r2 = hybrid_r2
        final_rmse = hybrid_rmse
    else:
        hybrid_full_year = None
        sliding_air_temp_full_year = None
        r_fitted = 0.0
        w_fitted = 1.0
        hybrid_param_errors = [0.0, 0.0]
        final_pred = chambers_pred
        final_r2 = chambers_r2
        final_rmse = chambers_rmse
    
    # Prepare results
    results = {
        'parameters': {
            'damping_depth_m': d_chambers_fitted,
            'phase_shift_rad': phase_chambers_fitted,
            'phase_shift_deg': np.degrees(phase_chambers_fitted),
            'ratio_r': r_fitted,
            'window_days': w_fitted,
            'T_mean_sensor': T_mean_sensor,
            'delta_T_sensor': delta_T_sensor,
            'depth_cm': depth,
            'depth_m': depth_m,
            'borehole': borehole,
            'use_hybrid': use_hybrid
        },
        'parameter_errors': {
            'damping_depth_m': chambers_param_errors[0],
            'phase_shift_rad': chambers_param_errors[1],
            'ratio_r': hybrid_param_errors[0] if use_hybrid else 0.0,
            'window_days': hybrid_param_errors[1] if use_hybrid else 0.0
        },
        'predictions': {
            # Measurement period
            'dates': combined.index,
            'temp_obs': combined['temp_sensor'],
            'temp_pred_final': final_pred,
            'temp_pred_chambers': chambers_pred,
            'temp_pred_hybrid': hybrid_pred if use_hybrid else chambers_pred,
            'air_temp': combined['air_temp'],
            'residuals_final': combined['temp_sensor'] - final_pred,
            'residuals_chambers': combined['temp_sensor'] - chambers_pred,
            
            # Full year
            'full_year_dates': full_year_dates,
            'full_year_chambers': chambers_full_year,
            'full_year_hybrid': hybrid_full_year if use_hybrid else chambers_full_year,
            'full_year_weather': weather_full_year
        },
        'diagnostics': {
            'chambers_r2': chambers_r2,
            'chambers_rmse': chambers_rmse,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'hybrid_r2': hybrid_r2 if use_hybrid else chambers_r2,
            'hybrid_rmse': hybrid_rmse if use_hybrid else chambers_rmse,
            'n_observations': len(combined)
        }
    }
    
    return results

def plot_two_stage_results(results):
    """Plot comprehensive results showing both Chambers and Hybrid models"""
    
    if results is None:
        print("No results to plot")
        return
    
    params = results['parameters']
    borehole = params['borehole']
    depth = params['depth_cm']
    use_hybrid = params['use_hybrid']
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Full year with all components
    # Weather data
    axes[0].plot(results['predictions']['full_year_dates'], results['predictions']['full_year_weather'], 
                '-', label='Weather Data', linewidth=1, color='orange', alpha=0.8)
    
    # Chambers model for full year
    axes[0].plot(results['predictions']['full_year_dates'], results['predictions']['full_year_chambers'], 
                '--', label='Chambers Model', linewidth=2, color='green')
    
    # Hybrid model for full year (if different from Chambers)
    if use_hybrid:
        axes[0].plot(results['predictions']['full_year_dates'], results['predictions']['full_year_hybrid'], 
                    '-', label='Hybrid Model', linewidth=2, color='red')
    
    # Sensor observations
    axes[0].plot(results['predictions']['dates'], results['predictions']['temp_obs'], 'o', 
                label='Sensor Data', markersize=4, color='blue', alpha=0.8)
    
    # Highlight measurement period
    meas_start = results['predictions']['dates'].min()
    meas_end = results['predictions']['dates'].max()
    axes[0].axvspan(meas_start, meas_end, alpha=0.15, color='gray', label='Measurement Period')
    
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Borehole {borehole} - {depth}cm: Full Year Overview\n'
                     f'{"Hybrid" if use_hybrid else "Chambers"} Model Selected '
                     f'(R² = {results["diagnostics"]["final_r2"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Measurement period detail
    axes[1].plot(results['predictions']['dates'], results['predictions']['temp_obs'], 'o', 
                label='Observed', markersize=4, color='blue')
    axes[1].plot(results['predictions']['dates'], results['predictions']['temp_pred_chambers'], '--', 
                label=f'Chambers (R²={results["diagnostics"]["chambers_r2"]:.3f})', linewidth=2, color='green')
    
    if use_hybrid:
        axes[1].plot(results['predictions']['dates'], results['predictions']['temp_pred_hybrid'], '-', 
                    label=f'Hybrid (R²={results["diagnostics"]["hybrid_r2"]:.3f})', linewidth=2, color='red')
    
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Measurement Period: Model Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    axes[2].plot(results['predictions']['dates'], results['predictions']['residuals_chambers'], 'o-', 
                label=f'Chambers Residuals (RMSE={results["diagnostics"]["chambers_rmse"]:.3f}°C)', 
                alpha=0.7, markersize=3, color='green')
    
    if use_hybrid:
        axes[2].plot(results['predictions']['dates'], results['predictions']['residuals_final'], 's-', 
                    label=f'Hybrid Residuals (RMSE={results["diagnostics"]["hybrid_rmse"]:.3f}°C)', 
                    alpha=0.7, markersize=3, color='red')
    
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Residuals (°C)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Model Residuals')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add parameter text box
    param_text = (
        f"CHAMBERS PARAMETERS:\n"
        f"Damping depth: {params['damping_depth_m']:.3f} ± {results['parameter_errors']['damping_depth_m']:.3f} m\n"
        f"Phase shift: {params['phase_shift_deg']:.1f}°\n"
        f"R²: {results['diagnostics']['chambers_r2']:.3f}\n"
        f"RMSE: {results['diagnostics']['chambers_rmse']:.3f}°C\n\n"
    )
    
    if use_hybrid:
        param_text += (
            f"HYBRID PARAMETERS:\n"
            f"Ratio (r): {params['ratio_r']:.3f} ± {results['parameter_errors']['ratio_r']:.3f}\n"
            f"Window (w): {params['window_days']:.1f} ± {results['parameter_errors']['window_days']:.1f} days\n"
            f"R²: {results['diagnostics']['hybrid_r2']:.3f}\n"
            f"RMSE: {results['diagnostics']['hybrid_rmse']:.3f}°C\n"
            f"Improvement: +{results['diagnostics']['hybrid_r2'] - results['diagnostics']['chambers_r2']:.3f}\n\n"
        )
    else:
        param_text += "HYBRID: Not used (no significant improvement)\n\n"
    
    param_text += (
        f"SENSOR DATA:\n"
        f"T_mean: {params['T_mean_sensor']:.2f}°C\n"
        f"ΔT: {params['delta_T_sensor']:.2f}°C\n"
        f"N obs: {results['diagnostics']['n_observations']}"
    )
    
    # Place text box
    fig.text(0.02, 0.5, param_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)
    plt.show()

def analyze_all_sensors_two_stage(sens_data):
    """Analyze all sensors with two-stage fitting approach"""
    
    results_all = {}
    
    for borehole in ['301', '302', '303']:
        results_all[borehole] = {}
        
        for depth in [60, 120]:
            sensor_id = f"{borehole}-{depth}cm"
            print(f"\n{'='*70}")
            print(f"Analyzing Sensor {sensor_id} - Two-Stage Fitting")
            print(f"{'='*70}")
            
            try:
                results = fit_improved_chambers_then_hybrid(
                    sens_data, borehole=borehole, depth=depth, debug=True
                )
                
                if results is not None:
                    results_all[borehole][depth] = results
                    plot_two_stage_results(results)
                    
                    # Print summary
                    params = results['parameters']
                    diagnostics = results['diagnostics']
                    use_hybrid = params['use_hybrid']
                    
                    print(f"\n✓ SUCCESS - Sensor {sensor_id}:")
                    print(f"  Model used: {'HYBRID' if use_hybrid else 'CHAMBERS'}")
                    print(f"  Final R²: {diagnostics['final_r2']:.3f}")
                    print(f"  Final RMSE: {diagnostics['final_rmse']:.3f}°C")
                    print(f"  Damping depth: {params['damping_depth_m']:.3f} m")
                    print(f"  Phase shift: {params['phase_shift_deg']:.1f}°")
                    if use_hybrid:
                        print(f"  Ratio (r): {params['ratio_r']:.3f}")
                        print(f"  Window: {params['window_days']:.1f} days")
                        print(f"  Improvement: +{diagnostics['hybrid_r2'] - diagnostics['chambers_r2']:.3f} R²")
                else:
                    print(f"✗ FAILED - Sensor {sensor_id}")
                    
            except Exception as e:
                print(f"✗ ERROR - Sensor {sensor_id}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    return results_all

# Example usage
if __name__ == '__main__':
    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_home

    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/General - Projet IV 2023 - GTO365/01-projet_IV-Mtl_Laval/03-Berlier-Bergman/05-donnees-terrains/'
    
    sens_data = load_TDR_data(Onedrive_path)

    # Analyze all sensors with two-stage approach
    results_all = analyze_all_sensors_two_stage(sens_data)