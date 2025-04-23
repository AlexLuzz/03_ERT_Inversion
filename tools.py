import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from matplotlib.backends.backend_pdf import PdfPages
from protocol import ElecPos
import pandas as pd
from datetime import datetime
from pygimli.physics import ert
import os

def create_front_page(parameters: str, title: str = "Inversion Parameters"):
    """
    Create a front page figure with the given parameters.

    :param parameters: str - The parameters to be printed on the front page.
    :param title: str - The title of the front page.
    """
    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.axis('off')  # Turn off the axis

    # Add title
    plt.text(0.5, 0.9, title, fontsize=20, ha='center')

    # Add parameters text
    plt.text(0.5, 0.5, parameters, fontsize=12, ha='center', va='center', wrap=True)
    plt.close('all')

    return fig

def generate_parameters_table(lam=10, zWeight=1, robustData=False, robustModel=False, blockyModel=False, startModel=None, tikhonov=1.0, **kwargs):
    """
    Generate a table of inversion parameters and their values.

    :param lam: Regularization parameter for inversion.
    :param zWeight: Weighting parameter for the z direction.
    :param robustData: Use robust data weighting.
    :param robustModel: Use robust model weighting.
    :param blockyModel: Use blocky model constraint.
    :param startModel: Initial model for the inversion.
    :param tikhonov: Weighting parameter for Tikhonov regularization in the time direction.
    :return: str - A string representation of the parameters table.
    """
    parameters = {
        "lam": lam,
        "zWeight": zWeight,
        "robustData": robustData,
        "robustModel": robustModel,
        "blockyModel": blockyModel,
        "startModel": startModel,
        "tikhonov": tikhonov
    }

    table = "Parameter | Default Value | Current Value\n"
    table += "--------- | ------------- | -------------\n"
    default_values = {
        "lam": 20,
        "zWeight": 1,
        "robustData": False,
        "robustModel": False,
        "blockyModel": False,
        "startModel": None,
        "tikhonov": 1.0
    }

    changed_params_list = []

    for param, value in parameters.items():
        default_value = default_values[param]
        table += f"{param} | {default_value} | {value}\n"
        if value != default_value:
            short_name = {
                "lam": "lm",
                "zWeight": "zW",
                "robustData": "rD",
                "robustModel": "rM",
                "blockyModel": "bM",
                "startModel": "sM",
                "tikhonov": "tk"
            }[param]
            changed_params_list.append(f"{short_name}{value}")

    changed_params = ", ".join(changed_params_list)

    return table, changed_params

def get_cmin_cmax(models, percentiles=(5, 95)):
    """
    Get the minimum and maximum values from a list of models.

    :param models: list - List of models (numpy arrays or similar).
    :return: tuple - (cMin, cMax) rounded to 3 significant figures.
    """
    all_values = np.concatenate([model.flatten() for model in models])
    #cMin = np.round(np.min(all_values), 3)
    #cMax = np.round(np.max(all_values), 3)
    cMin = np.round(np.percentile(all_values, percentiles[0]), 3)
    cMax = np.round(np.percentile(all_values, percentiles[1]), 3)
    return cMin, cMax

def plot_models(models, grid, survey_dates=None, cMap='Spectral_r', percentiles=(3, 97)):
    """
    Plot models.

    Parameters:
        models (list): List of models.
        grid (pg.Mesh): The grid used for the inversion.
        survey_dates (list, optional): List of survey dates corresponding to the models.
        cMap (str): Colormap for visualization.

    Returns:
        list: Figures for time-lapse resistivity models.
    """
    figs = []
    cMin, cMax = get_cmin_cmax(models, percentiles=percentiles)
    plt.ioff()
    for i, model in enumerate(models):
        # Plot time-lapse resistivity model
        fig, ax_tl = plt.subplots(figsize=(8, 6))
        pg.show(grid, model, ax=ax_tl, cMin=cMin, cMax=cMax, cMap=cMap, logScale=True, block=False)
        title_tl = f"Resistivity from {survey_dates[i]}"
        ax_tl.set_title(title_tl)
        plt.close('all')
        figs.append(fig)

    return figs

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

def saveFiguresToPDF(figures, pdf_filename, front_page=None, figsize=(12, 7), verbose=False):
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

def add_meas_weight(df, n_elec_bh=8):
    """
    Add a 'meas' column to the DataFrame based on measurement logic.

    :param df: Input DataFrame containing columns A, B, and M.
    :param n_elec_bh: Number of electrodes for borehole measurements (default is 8).
    :return: The DataFrame with the added 'meas' column.
    """
    df['meas'] = 'Unknown'  # Default value
    df['weight'] = 1.0  # Default value

    # Calculate the 'meas' based on the provided logic
    for index, row in df.iterrows():
        # Convert 'A', 'B', and 'M' to integers
        A = int(row['A'])
        B = int(row['B'])
        M = int(row['M'])
        B_A = B - A
        M_A = M - A
        if B_A == n_elec_bh:
            if M_A == 1:
                df.at[index, 'meas'] = 'KR0'
                df.at[index, 'weight'] = 1
            elif M_A == 2:
                df.at[index, 'meas'] = 'KR1'
                df.at[index, 'weight'] = 0.7
            elif M_A == 3:
                df.at[index, 'meas'] = 'KR2' 
                df.at[index, 'weight'] = 0.5
            elif M_A == 4:
                df.at[index, 'meas'] = 'KR3' 
                df.at[index, 'weight'] = 0.3
        elif B_A == 2 * n_elec_bh:
            if M_A == 1:
                df.at[index, 'meas'] = 'KJ0' 
                df.at[index, 'weight'] = 1
            elif M_A == 2:
                df.at[index, 'meas'] = 'KJ1'
                df.at[index, 'weight'] = 0.7
            elif M_A == 3:
                df.at[index, 'meas'] = 'KJ2'
                df.at[index, 'weight'] = 0.5 
            elif M_A == 4:
                df.at[index, 'meas'] = 'KJ3' 
                df.at[index, 'weight'] = 0.3
    return df

def compute_chargeability(data, delta_t = 0.1, n_windows=10):
    """
    Compute chargeability (in milliseconds) for an IP ERT survey.

    Parameters:
        data (DataFrame): A pandas DataFrame containing the raw data with:
            - Voltage(V): Primary voltage (V_p)
            - T(N):01 to T(N):10: Decay values (V_i) for each time window.
            - App.Ch.(ms): Apparent chargeabilities in milliseconds (to cross-check).

    Returns:
        DataFrame: Original DataFrame with computed chargeability added.
    """
    # Define time window duration (in seconds)
    delta_t = delta_t  # 100 ms windows by default

    # Dynamically identify decay columns based on n_windows
    decay_columns = []
    for i in range(1, n_windows + 1):
        decay_prefix = f"T(N):{i:02d}"  # Format as T(N):01, T(N):02, ...
        decay_column = [col for col in data.columns if decay_prefix in col]
        if decay_column:
            decay_columns.append(decay_column[0])
    # Check if decay columns are present
    if not decay_columns:
        print("No decay columns found. Skipping chargeability computation.")
        data["Computed_Chargeability"] = np.nan
        return data
    
    # Ensure all decay columns and Voltage(V) are numeric
    data[decay_columns] = data[decay_columns].apply(pd.to_numeric, errors='coerce')
    data["Voltage(V)"] = pd.to_numeric(data["Voltage(V)"], errors='coerce')

    # Compute chargeability for each row
    data["Computed_Chargeability"] = data.apply(
        lambda row: delta_t * np.nansum(row[decay_columns] / row["Voltage(V)"])
        if not pd.isna(row["Voltage(V)"]) else np.nan,
        axis=1
    )

    return data

def load_amp_files(file_paths, n_elec_bh=8, delete_columns=None, clear_electrodes=None, detect_by_first_measurement=True):
    """
    Load multiple ABEM Multi-Purpose Format (.AMP) files into a single pandas DataFrame.
    
    :param file_paths: List of paths to .AMP files.
    :param delete_columns: List of columns to delete from the DataFrame.
    :param clear_electrodes: List of electrode values to clear from the DataFrame (defaults to including '32768').
    :return: A concatenated pandas DataFrame containing the data from all files with an additional 'Survey' column.
    """
    try:
        # Set default columns to delete if none provided
        if delete_columns is None:
            delete_columns = ['A(y)', 'A(z)', 'B(y)', 'B(z)', 'M(y)', 'M(z)', 'N(y)', 'N(z)']
        
        # Ensure '32767' is always included in electrodes to clear
        if clear_electrodes is None:
            clear_electrodes = ['32767']
        else:
            # Convert all to string format and ensure '32768' is included
            clear_electrodes = list(set(map(str, clear_electrodes)) | {'32767'})

        all_dfs = []  # List to store DataFrames from each file
        survey_counter = 1  # Initialize the survey counter

        for file_path in file_paths:
            # Read the file and extract the relevant data
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Initialize variables
            start_time = None
            header = None
            data_start_index = None

            # Loop to find the start time and header line index
            for i, line in enumerate(lines):
                if start_time is None and 'Date & Time:' in line:
                    start_time_str = line.split(':', 1)[1].strip()
                    start_time = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S").replace(second=0, microsecond=0)

                if header is None and 'No.' in line:
                    header = line.strip().split()
                    data_start_index = i + 1  # Set the data start index

            # Ensure both start time and header were found
            if start_time is None:
                raise ValueError(f"Error: Start time not found in the metadata for file {file_path}.")
            if header is None or data_start_index is None:
                raise ValueError(f"Error: Data header not found in the file {file_path}.")

            # Read the data section after the header line
            data = [line.strip().split() for line in lines[data_start_index:] if line.strip()]

            # Create a DataFrame with the header
            df = pd.DataFrame(data, columns=header)

            # Rename columns to remove '(x)' and keep only the base name
            df.columns = df.columns.str.replace(r'\(x\)', '', regex=True)

            # Remove specified columns if they exist in the DataFrame
            df.drop(columns=[col for col in delete_columns if col in df.columns], inplace=True)

            for electrode_col in ['A', 'B', 'M', 'N']:
                if electrode_col in df.columns:
                    # Convert column values to strings for comparison and filter rows
                    df = df[~df[electrode_col].astype(str).isin(clear_electrodes)]

            # Reset the index to make it continuous
            df = df.reset_index(drop=True)

            # Convert 'Time' column to numeric for processing
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

            # Identify survey breaks and label surveys
            df['Survey'] = survey_counter  # Initialize with current survey number

            if detect_by_first_measurement:
                # Use the first measurement to detect new surveys
                first_measurement = df.iloc[0][['A', 'B', 'M', 'N']].astype(str).values

                for i in range(1, len(df)):          
                    # Compare the current measurement with the first measurement
                    current_measurement = df.iloc[i][['A', 'B', 'M', 'N']].astype(str).values
                    
                    # Detect new survey when encountering the first measurement again
                    if np.array_equal(current_measurement, first_measurement):
                        survey_counter += 1  # Increment survey number
                    
                    # Correctly assign the survey number to the DataFrame
                    df.at[i, 'Survey'] = survey_counter
                                     
            else:
                # Use time difference to detect new surveys
                for i in range(1, len(df)):
                    if pd.isna(df.at[i, 'Time']) or pd.isna(df.at[i - 1, 'Time']):
                        continue  # Skip if either current or previous 'Time' is NaN
                    if df.at[i, 'Time'] - df.at[i - 1, 'Time'] > 35:
                        survey_counter += 1  # Increment survey number for new survey
                    df.at[i, 'Survey'] = survey_counter
                    
            # Create 'SurveyDate' column based on the start of each survey
            df['SurveyDate'] = None  # Initialize column with None

            current_survey_start_time = start_time  # Set initial survey start time

            # Iterate through the DataFrame to assign the survey date
            for i in range(len(df)):
                if i == 0 or df.iloc[i]['Survey'] != df.iloc[i - 1]['Survey']:
                    # Update the current survey start time for new surveys
                    current_survey_start_time = start_time + pd.to_timedelta(df.iloc[i]['Time'], unit='s')
                
                # Assign the survey date to each row
                df.at[i, 'SurveyDate'] = current_survey_start_time.strftime('%Y-%m-%d %H:%M')

            df = add_meas_weight(df, n_elec_bh)

            # Append the DataFrame to the list
            all_dfs.append(df)

        # Concatenate all DataFrames into one large DataFrame
        return pd.concat(all_dfs, ignore_index=True)

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

def create_crosshole_data_df(df, square_size=0.2, space=3):
    """
    Create crosshole ERT data from AMP survey data.

    Parameters:
        amp_data (pd.DataFrame): A DataFrame containing AMP measurement data.

    Returns:
        data (pg.DataContainerERT): DataContainer for CrossholeERT.
        times (np.ndarray): Array of measurement times.
        DATA (np.ndarray): Array of apparent resistivities.
        ERR (np.ndarray): Array of measurement errors.
    """
    # Initialize lists to hold positions and measurements
    elec = ElecPos(n_columns = 8, n_lines = 8)
    elec.positions()
    elec.createGrid(show=False, square_size=square_size, space=space)
    grid = elec.grid

    DATA = []  # List to store data containers
    survey_dates = []  # List to store survey dates

    # Get unique surveys in amp_data
    unique_surveys = df['SurveyDate'].unique()

    for survey_date in unique_surveys:
        survey = df[df['SurveyDate'] == survey_date]  # Filter rows for the current survey
        measurements = []

        # Loop through each row in the survey DataFrame
        for _, row in survey.iterrows():
            # Collect measurement data: [a, b, m, n, rho_a, k, err]
            # Check for the specific "1.#QNAN0" error string and replace it with np.nan
            error_value = row['Error(%)']
            if error_value == "1.#QNAN0":
                error_value = 0.01
            else:
                error_value = float(error_value)

            # Check and compute chargeability (ip) and SI unit chargeability (ma)
            computed_chargeability = row.get("Computed_Chargeability", np.nan)
            if not pd.isna(computed_chargeability):
                ip = float(computed_chargeability)
            else:
                ip = np.nan

            measurements.append([
                float(row['A']),  # A electrode x
                float(row['B']),  # B electrode x
                float(row['M']),  # M electrode x
                float(row['N']),  # N electrode x
                float(row['Res.(ohm)']),  # res
                float(row['rhoa']) if float(row['rhoa']) != 0 else np.nan,  # rho_a
                error_value,  # err
                ip,  # Computed chargeability
            ])

        # Convert measurements to a numpy array
        measurements = np.array(measurements)
        
        # Prepare the data container
        data = ert.createData(elecs=elec.pos, schemeName='uk')

        # Assign data to the container
        data['a'] = measurements[:, 0].astype(float) - 1
        data['b'] = measurements[:, 1].astype(float) - 1
        data['m'] = measurements[:, 2].astype(float) - 1
        data['n'] = measurements[:, 3].astype(float) - 1
        data['k'] = ert.createGeometricFactors(data)
        data['r'] = measurements[:, 4].astype(float)
        data['rhoa'] = measurements[:, 5].astype(float)
        data['err'] = measurements[:, 6].astype(float)
        data['ip'] = measurements[:, 7].astype(float)  # Computed chargeability
        data['valid'] = True

        # Add the data container to the list
        DATA.append(data)

        # Append the SurveyDate for this survey (taking the first row's 'SurveyDate' as the date for the entire survey)
        survey_dates.append(survey['SurveyDate'].iloc[0])

    return DATA, survey_dates, grid

def save_data(data_containers, times, filename=None, masknan=False):
    """
    Save data from a list of DataContainer instances to files.

    Parameters:
        data_containers: list of DataContainer
            List of DataContainer instances to save.
        times: list of datetime
            List of time steps corresponding to the data containers.
        filename: str, optional
            Base filename for the saved files. Defaults to using the instance's name.
        masknan: bool, optional
            Whether to mask NaN values in the saved arrays.
    """
    # Construct filename if not provided, adding an index to differentiate files
    base_filename = filename or "data"
    if base_filename.endswith(".shm"):
        base_filename = base_filename[:-4]

    # Delete existing files with the base_filename for the first DataContainer
    for ext in [".shm", ".rhoa", ".err", ".ip", ".times"]:
        try:
            os.remove(base_filename + ext)
        except FileNotFoundError:
            pass

    # Prepare data for saving
    all_rhoa = []
    all_err = []
    all_ip = []

    data_containers[1].save(base_filename + ".shm")

    # Loop over each DataContainer in the list
    for i, data_container in enumerate(data_containers):
        # Collect data for saving
        rhoa = data_container['rhoa']
        err = data_container['err']
        ip = data_container['ip'] if data_container.haveData('ip') else np.full_like(rhoa, np.nan)

        all_rhoa.append(rhoa)
        all_err.append(err)
        all_ip.append(ip)

    # Find the median length among all arrays
    lengths = [len(rhoa) for rhoa in all_rhoa]
    percentile_length = int(np.percentile(lengths, 90))

    # Cut all arrays to the 90th percentile length
    all_rhoa = [rhoa[:percentile_length] for rhoa in all_rhoa]
    all_err = [err[:percentile_length] for err in all_err]
    all_ip = [ip[:percentile_length] for ip in all_ip]

    # Find the maximum length among the truncated arrays
    max_length = max(len(rhoa) for rhoa in all_rhoa)

    # Create arrays filled with NaN of the longest shape
    all_rhoa_filled = np.full((len(all_rhoa), max_length), np.nan)
    all_err_filled = np.full((len(all_err), max_length), np.nan)
    all_ip_filled = np.full((len(all_ip), max_length), np.nan)

    # Fill the arrays with the actual data
    for i in range(len(all_rhoa)):
        all_rhoa_filled[i, :len(all_rhoa[i])] = all_rhoa[i]
        all_err_filled[i, :len(all_err[i])] = all_err[i]
        all_ip_filled[i, :len(all_ip[i])] = all_ip[i]

    # Transpose the collected data to write each line for the same measurement across all DataContainers
    all_rhoa_filled = all_rhoa_filled.T
    all_err_filled = all_err_filled.T
    all_ip_filled = all_ip_filled.T

    # Write all data from the same measurement across all DataContainers separated by a tabulation into the respective files
    with open(base_filename + ".rhoa", "a") as f_rhoa, \
         open(base_filename + ".err", "a") as f_err, \
         open(base_filename + ".ip", "a") as f_ip:
        for j in range(all_rhoa_filled.shape[0]):
            f_rhoa.write("\t".join(f"{val:6.4f}" for val in all_rhoa_filled[j]) + "\n")
            f_err.write("\t".join(f"{val:6.4f}" for val in all_err_filled[j]) + "\n")
            f_ip.write("\t".join(f"{val:6.4f}" for val in all_ip_filled[j]) + "\n")

    print(f"Data from all DataContainers saved successfully to files with base name: {base_filename}")

    # Save time steps passed to the function
    with open(base_filename + ".times", "a", encoding="utf-8") as fid:
        for d in times:
            fid.write(d.isoformat() + "\n")
