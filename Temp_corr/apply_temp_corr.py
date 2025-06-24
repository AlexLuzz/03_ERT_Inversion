import numpy as np
import pandas as pd

def load_model_results(file_path):
    """
    Load saved model results
    
    Parameters:
    -----------
    file_path : str
        Path to saved model file (.pkl)
    
    Returns:
    --------
    dict : Loaded model results
    """
    
    if file_path.endswith('.pkl'):
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
def apply_chambers_model(model_results, new_dates, depths):
    """
    Apply saved Chambers model to new dates and depths
    
    Parameters:
    -----------
    model_results : dict
        Loaded model results from save_model_results
    new_dates : array-like
        New dates to predict (pandas datetime)
    depths : array-like
        Depths to predict (in meters, positive values)
    
    Returns:
    --------
    dict : Predictions for new dates and depths
    """
    
    # Extract fitted parameters
    fitted_params = model_results['fitted_params']
    model_type = model_results['model_type']
    
    # Convert dates to fractional days since start of year
    def date_to_fractional_days(dates):
        dates = pd.to_datetime(dates)
        start_of_year = pd.Timestamp(dates.min().year, 1, 1)
        return (dates - start_of_year).total_seconds() / (24 * 3600)
    
    t_new = date_to_fractional_days(new_dates)
    
    # Recreate model function based on type
    if model_type == 'original':
        T_weather_mean = model_results['parameters']['T_weather_mean']
        delta_T_weather = model_results['parameters']['delta_T_weather']
        
        def chambers_model(inputs, d, phase_shift):
            z, t = inputs
            t_annual = 2 * np.pi * t / 365.25
            damping = np.exp(-z / d)
            phase_lag = -z / d
            temp = T_weather_mean + delta_T_weather / 2 * damping * np.sin(t_annual + phase_shift + phase_lag)
            return temp
    
    else:  # improved model
        def chambers_model(inputs, T_surf_mean, delta_T_surf, d, phase_shift):
            z, t = inputs
            t_annual = 2 * np.pi * t / 365.25
            damping = np.exp(-z / d)
            phase_lag = -z / (4 * d)
            temp = T_surf_mean + delta_T_surf * damping * np.sin(t_annual + phase_shift + phase_lag)
            return temp
    
    # Generate predictions
    predictions = {}
    for depth in depths:
        depth_key = f"depth_{depth:.0f}cm"
        predictions[depth_key] = chambers_model((np.full_like(t_new, depth), t_new), *fitted_params)
    
    return {
        'dates': new_dates,
        'predictions': predictions,
        'depths': depths,
        'model_info': {
            'type': model_type,
            'parameters': model_results['parameters']
        }
    }
    
if __name__ == "__main__":
    a = 1