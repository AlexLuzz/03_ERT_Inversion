import pygimli as pg
from pygimli.physics import ert
from Tools.tools import *
import numpy as np
import matplotlib.pyplot as plt

def saveVariables(invIP, folder=None):
    """Save specified variables in the specified folder."""
    print(f'Saving variables to: {folder}')
    np.savetxt(f"{folder}/chargeability.vector", invIP.model)
    np.savetxt(f"{folder}/data_chargeability.vector", invIP.dataVals)
    np.savetxt(f"{folder}/dataErrs_chargeability.vector", invIP.dataErrs)
    np.savetxt(f"{folder}/data_response.vector", invIP.response)

def perform_inversion(inversion_type, data_file, mesh_file, filename, **kwargs):
    """
    Perform inversion based on the specified type.

    Parameters:
        inversion_type (str): Type of inversion ('INV', 'TL', 'IP').
        data_file (str): Path to the data file.
        mesh_file (str): Path to the mesh file.
        filename (str): Base filename for saving results.
        survey_range (tuple, optional): Range of surveys to include (start, end).
        **kwargs: Additional inversion parameters, including 't' to specify surveys to process.

    Returns:
        models: Inversion models.
    """
    # Load the mesh
    mesh = pg.load(mesh_file)
    
    plt.ioff()

    # Define inversion parameters
    inversion_params = {
        'lam': 20,              # Regularization parameter
        'maxIter': 10,          # Maximum number of iterations
        'verbose': True,        # Whether to display detailed information
        'robustData': False,    # Whether to use robust data (set to False by default)
        'robustModel': False,   # Whether to use robust model (set to False by default)
        'blockyModel': False,   # Whether to use blocky model (set to False by default)
        'startModel': None,     # Starting model (None by default)
        'referenceModel': None  # Reference model (None by default)
    }
    
    # Merge user-defined parameters into the inversion parameters
    inversion_params.update(kwargs)

    # Generate parameters table
    table, changed_params = generate_parameters_table(**inversion_params)
    front_page = create_front_page(table)

    # Filter data based on the 't' parameter
    t = kwargs.get('t', None)

    if inversion_type == 'INV':
        inv = ert.TimelapseERT(data_file, mesh=mesh)
        inv.invert(**inversion_params)
        inv.generateTimelinePDF(filename=f"{filename}_timeline_{inversion_type}.pdf")
    elif inversion_type == 'TL':
        inv = ert.TimelapseERT(data_file, mesh=mesh)
        inv.fullInversion(**inversion_params)
        #inv.generateTimelinePDF(filename=f"{filename}_timeline_{inversion_type}.pdf")
    elif inversion_type == 'IP':
        models_rhoa = []
        models_IP = []
        timestamps = []
        # Load the .rhoa and .ip files
        rhoa_data = np.loadtxt(data_file.replace('.shm', '.rhoa')).T
        ip_data = np.loadtxt(data_file.replace('.shm', '.ip')).T
        inv_time = ert.TimelapseERT(data_file)
        
        # Use the 't' parameter to specify which surveys to process
        if t is not None:
            indices = t
        else:
            indices = range(rhoa_data.shape[0]-1)

        for i in indices:
            # Create a new DataContainer for each survey
            data_container = ert.load(data_file)
            
            # Swap the appropriate line of data
            data_container['rhoa'] = rhoa_data[i]
            data_container['ip'] = ip_data[i]*1000
            
            inv = ert.ERTIPManager(data_container, fd=False)
            inv.invert(mesh=mesh, **inversion_params)
            inv.invertIP(**inversion_params)
            # Format the timestamp to be compatible with Windows file naming conventions
            timestamp = inv_time.times[i].strftime("%m-%d_%H")
            timestamps.append(timestamp)
            inv.saveResult(folder=f"{filename}IP/{timestamp}")
            saveVariables(inv.invIP, folder=f"{filename}IP/{timestamp}/ERTIPManager")
            models_rhoa.append(inv.model)
            models_IP.append(inv.modelIP)

        filename = filename+timestamps[0]+"_"+timestamps[-1]
        figs_rhoa= plot_models(models_rhoa, mesh, timestamps)
        saveFiguresToPDF(figs_rhoa, f"{filename}_models_R_IP.pdf", front_page)
        figs_ip= plot_models(models_IP, mesh, timestamps, percentiles=(2, 98))
        saveFiguresToPDF(figs_ip, f"{filename}_model_IP.pdf", front_page)
        figs_ratio_rhoa = plot_model_ratios(models_rhoa, mesh, timestamps)
        saveFiguresToPDF(figs_ratio_rhoa, f"{filename}_ratios_R_IP.pdf", front_page)
        figs_ratio_ip = plot_model_ratios(models_IP, mesh, timestamps, percentiles=(2, 98))
        saveFiguresToPDF(figs_ratio_ip, f"{filename}_ratios_IP.pdf", front_page)
    else:
        raise ValueError(f"Unknown inversion type: {inversion_type}")

    if inversion_type != 'IP':
        inv.saveResults(basename=f"{filename}_Data_{inversion_type}")
        figs = plot_models(inv.models, mesh, inv.times, percentiles=(2, 95))
        saveFiguresToPDF(figs, f"{filename}_models_{inversion_type}.pdf", front_page)
        figs_ratios = plot_model_ratios(inv.models, mesh, inv.times, ref_survey=2, cM=0.05, percentiles=(2, 97))
        saveFiguresToPDF(figs_ratios, f"{filename}_ratios_{inversion_type}.pdf", front_page)

grid = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/Grids/01_grid_sq_0.15sp2.0.bms'

folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/"
#file = folder + "REDOUX_02-15_10h_03-01_04h.shm"
#file = folder + "RE_11-21_00h_11-25_22h.shm"
file = folder + "TLERT_Data/" + "HRE_sans_18_37_58.shm"
#file = folder + "HIV_01-19_13h_03-01_04h.shm"

filename = folder + "Results/" + file.split("/")[-1].split(".")[0]+ "sans_18_37_58"

# Survey range processd "list(range(1, 3))" = [1, 2]
#t = list(range(1, 106,5))
#t = 1
#t = [10, 15, 20, 25, 30, 35, 40, 45]

# Call the function for classic inversion
# models = perform_inversion(inversion_type='INV', data_file=file, mesh_file=grid, filename=filename, t=t)

# Call the function for IP inversion (time-domain inversion)
#models = perform_inversion(inversion_type='IP', data_file=file, mesh_file=grid, filename=folder, t=t, robustData=True)

# Call the function for time-lapse inversion
models = perform_inversion(inversion_type='TL', data_file=file, mesh_file=grid, filename=filename)

