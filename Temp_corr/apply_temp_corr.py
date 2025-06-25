import numpy as np
import pandas as pd
import pygimli as pg
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

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
        
def import_inversion_results(base_name):
    """
    Import PyGIMLi inversion results from multiple files.
    
    Parameters:
    -----------
    base_name : str
        Base name for the files (e.g., 'test' for test.bms, test.result, etc.)
    
    Returns:
    --------
    dict containing:
        - mesh: PyGIMLi mesh
        - results: Inverted resistivity values (2D array: time x cells)
        - rhoa: Apparent resistivity data
        - errors: Measurement errors
        - data_container: PyGIMLi DataContainer
        - times: Survey datetimes
    """
    
    results = {}
    
    try:
        # Import mesh
        print(f"Loading mesh from {base_name}.bms...")
        results['mesh'] = pg.load(f"{base_name}.bms")
        
        # Import inverted resistivity results
        print(f"Loading inversion results from {base_name}.result...")
        results['results'] = np.load(f"{base_name}.result.npy") if f"{base_name}.result".endswith('.npy') else np.loadtxt(f"{base_name}.result")
        
        # Import apparent resistivity
        print(f"Loading apparent resistivity from {base_name}.rhoa...")
        results['rhoa'] = np.load(f"{base_name}.rhoa.npy") if f"{base_name}.rhoa".endswith('.npy') else np.loadtxt(f"{base_name}.rhoa")
        
        # Import errors
        print(f"Loading errors from {base_name}.err...")
        results['errors'] = np.load(f"{base_name}.err.npy") if f"{base_name}.err".endswith('.npy') else np.loadtxt(f"{base_name}.err")
        
        # Import data container
        print(f"Loading data container from {base_name}.shm...")
        results['data_container'] = pg.load(f"{base_name}.shm")
        
        # Import survey times
        print(f"Loading survey times from {base_name}.times...")
        with open(f"{base_name}.times", 'r') as f:
            times_data = f.read().strip().split('\n')
            results['times'] = [datetime.fromisoformat(t.strip()) for t in times_data if t.strip()]
        
        print(f"Successfully loaded all files for {base_name}")
        print(f"Mesh: {results['mesh'].cellCount()} cells, {results['mesh'].nodeCount()} nodes")
        print(f"Results shape: {np.array(results['results']).shape}")
        print(f"Number of time steps: {len(results['times'])}")
        
        return results
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return None

def plot_first_inversion(results, base_name="", plot_mesh=False):
    """
    Plot the mesh and first inversion result.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from import_inversion_results()
    base_name : str
        Base name for plot titles
    """
    
    if results is None:
        print("No results to plot!")
        return
    
    mesh = results['mesh']
    resistivity_data = np.array(results['results'])
    
    # Handle different data shapes
    if resistivity_data.ndim == 1:
        # Single time step
        first_result = resistivity_data
        time_label = "Single time step"
    else:
        # Multiple time steps - take first
        first_result = resistivity_data[0, :] if resistivity_data.shape[0] < resistivity_data.shape[1] else resistivity_data[:, 0]
        time_label = f"Time: {results['times'][0].strftime('%Y-%m-%d %H:%M')}" if results['times'] else "First time step"
    

    # Plot 1: Mesh
    if plot_mesh:
        ax1, _ = pg.show(mesh)
        ax1.set_title(f'{base_name} - Mesh Structure\n{mesh.cellCount()} cells, {mesh.nodeCount()} nodes')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Elevation (m)')

    # Plot 2: First inversion result
    ax2, _ = pg.show(mesh, first_result, logScale=True, cMap='Spectral_r', colorBar=True, label='Resistivity (Ωm)')
    ax2.set_title(f'{base_name} - Resistivity Distribution\n{time_label}')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Elevation (m)')

    # Print some statistics
    print(f"\nResistivity statistics for first time step:")
    print(f"Min: {np.min(first_result):.2f} Ωm")
    print(f"Max: {np.max(first_result):.2f} Ωm")
    print(f"Mean: {np.mean(first_result):.2f} Ωm")
    print(f"Median: {np.median(first_result):.2f} Ωm")

def apply_chambers_to_mesh(results, model_results, datetime_array):
    """
    Apply Chambers model temperatures to mesh cells based on their X position and depth.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from import_inversion_results()
    model_results : dict
        Your Chambers model results object
    datetime_array : array of datetime object date 
        Dates in datetime format
    
    Returns:
    --------
    temperatures : array
        Temperature values for each mesh cell
    """
    
    if results is None:
        print("No mesh results available!")
        return None
    
    mesh = results['mesh']
    
    # Get mesh cell centers
    cell_centers = mesh.cellCenters()
    cell_x = cell_centers[:, 0]  # X-coordinate
    cell_z = cell_centers[:, 1]  # Z-coordinate (depth/elevation)
    
    # If your mesh uses negative Y for depth, adjust accordingly:
    # cell_z = -cell_centers[:, 1]  # Uncomment if needed
    
    # Create position pairs for chambers model
    positions = list(zip(cell_x, cell_z))
    
    temp_array = []

    for date in datetime_array:
        # Apply chambers model to all mesh positions
        chamber_data = apply_chambers_model(model_results, date, positions)
        
        # Extract temperatures for each mesh cell
        temperatures = np.zeros(len(positions))
        for i, (x, z) in enumerate(positions):
            position_key = f"x_{x:.2f}_z_{z:.2f}"
            if position_key in chamber_data['predictions']:
                temperatures[i] = chamber_data['predictions'][position_key][0]  # First (and only) time step
        temp_array.append(temperatures)
    print(f"Applied Chambers model temperatures to {len(temperatures)} mesh cells")
    print(f"Temperature range: {np.min(temperatures):.2f}°C to {np.max(temperatures):.2f}°C")
    
    return temp_array

def apply_chambers_model(model_results, new_dates, z_positions, x_positions=None):
    """
    Apply saved Chambers model to new dates and depths with spatial interpolation
    
    Parameters:
    -----------
    model_results : dict
        Loaded model results from save_model_results
    new_dates : array-like
        New dates to predict (pandas datetime format or string)
    z_positions : array-like
        Depths to predict (in meters, positive values)
    x_positions : array-like, optional
        X positions to predict. If None, uses z_positions as (x,z) pairs
    
    Returns:
    --------
    dict : Predictions for new dates and depths
    """
    import pandas as pd
    
    # Convert dates to fractional days since start of year
    def date_to_fractional_days(dates):
        if isinstance(dates, str):
            dates = [dates]
        dates = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')
        if hasattr(dates, '__len__') and len(dates) > 0:
            start_of_year = pd.Timestamp(dates.iloc[0].year if hasattr(dates, 'iloc') else dates[0].year, 1, 1)
        else:
            start_of_year = pd.Timestamp(dates.year, 1, 1)
        return (dates - start_of_year).total_seconds() / (24 * 3600)
    
    t_new = date_to_fractional_days(new_dates)
    if np.isscalar(t_new):
        t_new = np.array([t_new])

    # Define X positions of boreholes 
    x_301 = -2.0
    x_302 = -0.5
    x_303 = 1.0
    
    borehole_x_positions = {
        'BB - 301': x_301,
        'BB - 302': x_302, 
        'BB - 303': x_303
    }

    # Extract parameters from model results for each borehole
    borehole_params = {}
    for bh in ['BB - 301', 'BB - 302', 'BB - 303']:
        if bh in model_results:
            borehole_params[bh] = {
                'T_surface_mean': model_results[bh]['parameters']['T_surface_mean'],
                'delta_T_surface': model_results[bh]['parameters']['delta_T_surface'],
                'damping_depth_m': model_results[bh]['parameters']['damping_depth_m'],
                'phase_shift_rad': model_results[bh]['parameters']['phase_shift_rad'],
                'x_position': borehole_x_positions[bh]
            }

    def chambers_model(inputs, params):
        """Calculate temperature for given depth, time and parameters"""
        z, t = inputs
        T_mean = params['T_surface_mean']
        delta_T = params['delta_T_surface']
        d = params['damping_depth_m']
        phase_shift = params['phase_shift_rad']
        
        t_annual = 2 * np.pi * t / 365.25
        damping = np.exp(-abs(z) / d)
        phase_lag = -abs(z) / 4*d
        temp = T_mean + delta_T / 2 * damping * np.sin(t_annual + phase_shift + phase_lag)
        return temp
    
    # Handle input positions
    if x_positions is None:
        # Assume z_positions contains (x, z) pairs
        if hasattr(z_positions[0], '__len__'):
            x_coords = [pos[0] for pos in z_positions]
            z_coords = [pos[1] for pos in z_positions]
        else:
            # If only z provided, use middle borehole position
            x_coords = [x_302] * len(z_positions)
            z_coords = z_positions
    else:
        # Separate x and z coordinates provided
        x_coords = x_positions
        z_coords = z_positions
    
    # Generate predictions
    predictions = {}
    
    # Get sorted borehole data for interpolation
    available_boreholes = list(borehole_params.keys())
    bh_x_vals = [borehole_params[bh]['x_position'] for bh in available_boreholes]
    sort_idx = np.argsort(bh_x_vals)
    sorted_boreholes = [available_boreholes[i] for i in sort_idx]
    sorted_x_vals = [bh_x_vals[i] for i in sort_idx]
    
    for i, (x, z) in enumerate(zip(x_coords, z_coords)):
        position_key = f"x_{x:.2f}_z_{z:.2f}"
        
        # Calculate temperature for each borehole
        borehole_temps = {}
        for bh in available_boreholes:
            params = borehole_params[bh]
            temp = chambers_model((z, t_new), params)
            borehole_temps[bh] = temp
        
        # Interpolate based on X position
        if x <= sorted_x_vals[0]:
            # Use leftmost borehole
            interpolated_temp = borehole_temps[sorted_boreholes[0]]
        elif x >= sorted_x_vals[-1]:
            # Use rightmost borehole
            interpolated_temp = borehole_temps[sorted_boreholes[-1]]
        else:
            # Interpolate between boreholes
            for j in range(len(sorted_x_vals) - 1):
                if sorted_x_vals[j] <= x <= sorted_x_vals[j + 1]:
                    # Linear interpolation between boreholes j and j+1
                    x1, x2 = sorted_x_vals[j], sorted_x_vals[j + 1]
                    weight = (x - x1) / (x2 - x1)
                    temp1 = borehole_temps[sorted_boreholes[j]]
                    temp2 = borehole_temps[sorted_boreholes[j + 1]]
                    interpolated_temp = (1 - weight) * temp1 + weight * temp2
                    break
        
        predictions[position_key] = interpolated_temp
    
    return {
        'dates': new_dates,
        'predictions': predictions,
        'z_positions': z_coords,
        'x_positions': x_coords,
        'borehole_params': borehole_params
    }

def plot_chambers_on_mesh(results, temperatures, base_name, target_date):
    """
    Plot Chambers model temperatures on the mesh.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from import_inversion_results()
    temperatures : array
        Temperature values from apply_chambers_to_mesh()
    base_name : str
        Base name for plot titles
    target_date : str
        Date string for plot title
    """
    
    mesh = results['mesh']
        
    # Plot temperatures on mesh
    ax, _ = pg.show(mesh, temperatures, colorBar=True, cMap='coolwarm')
    ax.set_title(f'{base_name} - Chambers Model Temperatures\nDate: {target_date}')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    
    plt.tight_layout()
    plt.show()

def get_cmin_cmax(models, percentiles=(2, 98)):
    """
    Get the minimum and maximum values from a list of models.

    :param models: list - List of models (numpy arrays, lists, or pygimli.RVector objects).
    :param percentiles: tuple - Percentiles to use for cMin and cMax.
    :return: tuple - (cMin, cMax) rounded to 3 significant figures.
    """
    # Ensure models are valid arrays and handle both 1D and 2D arrays
    valid_models = []
    for model in models:
        if isinstance(model, (np.ndarray, list)):
            valid_models.append(np.array(model).flatten())
        elif isinstance(model, pg.RVector):
            valid_models.append(np.array(model).flatten())

    if not valid_models:
        raise ValueError("No valid models found. Ensure that the models list contains valid arrays or pygimli.RVector objects.")

    # Concatenate all valid model values
    all_values = np.concatenate(valid_models)

    # Calculate cMin and cMax based on the specified percentiles
    cMin = np.round(np.percentile(all_values, percentiles[0]), 3)
    cMax = np.round(np.percentile(all_values, percentiles[1]), 3)

    return cMin, cMax

def plot_model_ratios(models, grid, survey_dates, ref_survey=0, cMap='coolwarm', percentiles=(3, 97), cM=None):
    """
    Plot the ratio of all time-lapse models relative to a reference survey.

    Parameters:
        models (list): List of time-lapse resistivity models.
        grid (pg.Mesh): The grid used for the inversion.
        survey_dates (list): List of survey dates corresponding to the models.
        ref_survey (int): Index of the reference survey model for ratio calculation.
        cMap (str): Colormap for the ratio plots.
        percentiles (tuple): Percentiles to use for cMin and cMax.

    Returns:
        list: Figures for time-lapse resistivity ratios.
    """
    figs_ratio = []
    plt.ioff()
    # Get the reference model
    ref_tl_model = np.array(models[ref_survey])

    # Compute ratio
    ratio_models = [np.array(model) / ref_tl_model - 1 for model in models]
    cMin, cMax = get_cmin_cmax(ratio_models, percentiles=percentiles)
    if cM == None:  
        cM = max(abs(cMin), abs(cMax))

    for i in range(len(models)):
        if i == ref_survey:
            continue  # Skip ratio with itself

        # Plot time-lapse resistivity ratio
        fig, ax = plt.subplots(figsize=(8, 6))
        pg.show(grid, ratio_models[i], ax=ax, cMap=cMap, cMin=-cM, cMax=cM, block=False)
        ax.set_title(f"Ratio: Survey {survey_dates[i]} / Survey {survey_dates[ref_survey]}")
        plt.close('all')
        figs_ratio.append(fig)

    return figs_ratio

def saveFiguresToPDF(figures, pdf_filename, front_page=None, figsize=(12, 7), verbose=True):
    """Save a list of figures to a multi-page PDF.

    Parameters:
        figures (list): List of Matplotlib figure objects to be saved.
        pdf_filename (str): The name of the output PDF file.
        figsize (tuple): Size of the figures. Default is (12, 7).
        verbose (bool): If True, prints additional information. Default is False.
        front_page (matplotlib.figure.Figure, optional): A front page figure to be added as the first page.
    """
    with PdfPages(pdf_filename) as pdf:
        if front_page is not None:
            if verbose:
                print("Adding front page to PDF.")
            front_page.set_size_inches(figsize)
            pdf.savefig(front_page, bbox_inches='tight')
            plt.close(front_page)
        
        for i, fig in enumerate(figures):
            if verbose:
                print(f"Saving figure {i + 1}/{len(figures)} to PDF.")
            fig.set_size_inches(figsize)  # Set the figure size
            pdf.savefig(fig, bbox_inches='tight')  # Save the current figure to the PDF
            plt.close(fig)  # Close the figure to free memory
    if verbose:
        print(f"All figures saved to {pdf_filename}.")

if __name__ == "__main__":

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS
    
    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/'

    base_name = "ERT_Data/Results/HRE"  # Change this to your actual base name
    results = import_inversion_results(Onedrive_path + base_name)
    
    #plot_first_inversion(results, base_name)

    # Example usage of apply_chambers_model
    model_results = load_model_results(Onedrive_path + '99 - Mémoire -Article/BB_24h_model.pkl')

    mesh_temps = apply_chambers_to_mesh(results, model_results, results['times'])

    #plot_chambers_on_mesh(results, mesh_temps[4], "XXX", results['times'][4])

    c = 0.02
    Tref = 10
    # Apply temperature correction to results
    corr_results = results['results'] * (1 + c * (np.array(mesh_temps) - Tref))

    corrected_figs = plot_model_ratios(corr_results, results['mesh'], results['times'])
    saveFiguresToPDF(corrected_figs, Onedrive_path + 'ERT_Data/Results/HRE_corrected_ratios.pdf')
