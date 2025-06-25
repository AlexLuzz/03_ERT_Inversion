import pygimli as pg
from pygimli.physics import ert
from Tools.tools import *
import pandas as pd
import pygimli.meshtools as mt

def protocol_ALERT(n_columns=8,
                   n_lines=8,
                   max_v=4,
                   skip=True):
    """
    :param pos: positions [x, y] of the electrodes
    :param n_columns: number of column of the electrodes array
    :param n_lines: number of lines of the electrodes array
    :param max_v: maximum of potential measurement made in ALERT protocol
    :return:
    """
    protocol = np.empty((4, 0), dtype=int)
    for i in range(n_columns - 1):
        for j in range(n_lines - 1):
            for k in range(min(n_lines - j - 1, max_v)):
                # Calculate the values for the current column
                A = j + i * n_lines
                B = j + n_lines + i * n_lines
                M = A + k + 1
                N = B + k + 1
                mesure = np.array(([A, B, M, N]))[:, np.newaxis]
                protocol = np.hstack((protocol, mesure))

    if skip:
        for i in range(n_columns - 2):
            for j in range(n_lines - 1):
                for k in range(min(n_lines - j - 1, max_v)):
                    # Calculate the values for the current column
                    A = j + i * n_lines
                    B = j + 2*n_lines + i * n_lines
                    M = A + k + 1
                    N = B + k + 1
                    mesure = np.array(([A, B, M, N]))[:, np.newaxis]
                    protocol = np.hstack((protocol, mesure))

    return protocol

def positions(n_columns=8, n_lines=8, depth_start=-0.2, h_spacing=0.85, v_spacing=0.2):
        pos = np.zeros((n_columns * n_lines, 2))
        for col in range(n_columns):
            for line in range(n_lines):
                x = round(h_spacing * (n_columns - 1) / 2 - h_spacing * col, 3)
                y = round(depth_start - line * v_spacing, 3)
                pos[col * n_lines + line] = [x, y]
        return pos  # Return the calculated positions

def BB_mesh():
    # Create data and mesh files
    elec_pos = positions()
    scheme = ert.createData(elecs=elec_pos, schemeName='uk')
    # Define the coordinates for the square mesh
    x_min, x_max = scheme.sensors()[:, 0].min() - 0.1, scheme.sensors()[:, 0].max() + 0.1
    y_min, y_max = scheme.sensors()[:, 1].min() - 0.1, scheme.sensors()[:, 1].max() + 0.1

    # Create a grid
    world = mt.createWorld(start=[x_min, y_min], end=[x_max, y_max], layers=[0], marker=0)

    # add additional nodes around sensor locations
    for p in scheme.sensors():
        world.createNode(p)
        world.createNode(p-0.01)

    # Create a mesh with the polygon zone
    mesh = mt.createMesh(world, quality=34, area=0.1, areaMax=0.3)

    return mesh

if __name__ == "__main__":

    user_ETS = 'AQ96560'
    user_home = 'alexi'
    user = user_ETS
    
    Onedrive_path = f'C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/'

    df = pd.read_csv(Onedrive_path + 'fused_SAS4000_OhmPi_20250616.csv', sep=';')

    inv_results = f"{Onedrive_path}Results/"
    
    mesh = BB_mesh()

    # Print df length before cleaning
    print(f"Initial data length: {len(df)}")
    # Remove rows with specific values in columns A, B, M, N
    cleaned_df = df[~df[['A', 'B', 'M', 'N']].isin([18, 34]).any(axis=1)]
    print(f"Data length after removing electrodes 18 and 34: {len(cleaned_df)}")
    cleaned_df = cleaned_df[cleaned_df['Voltage(mV)'] > 50]
    print(f"Data length after removing Voltage(mV) <= 50: {len(cleaned_df)}")
    cleaned_df = cleaned_df[cleaned_df['I(mA)'] > 2]
    print(f"Data length after removing I(mA) <= 2: {len(cleaned_df)}")
    cleaned_df['Error(%)'] = pd.to_numeric(cleaned_df['Error(%)'], errors='coerce')
    cleaned_df = cleaned_df[(cleaned_df['Error(%)'] < 10) | (df['Error(%)'].isna())]
    print(f"Data length after removing Error(%) >= 10: {len(cleaned_df)}")

    # Flood test
    df = df[df['SurveyDate'] > '2024-11-18 00:00:00']
    df = df[df['SurveyDate'] < '2024-11-28 00:00:00']

    unique_surveys = df["Survey"].unique()
    selected_surveys = unique_surveys[::3]
    df = df[df["Survey"].isin(selected_surveys)]

    DATA, survey_dates, _  = create_crosshole_data_df(df)

    # Convert survey_dates from strings to datetime objects
    survey_dates_datatime = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in survey_dates]

    inv = ert.TimelapseERT(DATA=DATA, times=survey_dates_datatime, mesh=mesh)
    inv.saveData(inv_results, masknan=True)

    inversion_params = {
            'lam': 10,              # Regularization parameter
            'maxIter': 15,          # Maximum number of iterations
            'verbose': True,        # Whether to display detailed information
            'robustData': False,    # Whether to use robust data (set to False by default)
            'robustModel': False,   # Whether to use robust model (set to False by default)
            'blockyModel': False,   # Whether to use blocky model (set to False by default)
            'startModel': None,     # Starting model (None by default)
            'referenceModel': None  # Reference model (None by default)
        }

    table, _ = generate_parameters_table(inversion_params)
    front_page = create_front_page(table)

    inv.fullInversion(**inversion_params)

    inv.saveResults(basename=inv_results + 'HRE')
    figs = plot_models(inv.models, inv.mgr.paraDomain, inv.times, percentiles=(2, 95))
    saveFiguresToPDF(figs, inv_results + "HRE_raw_models.pdf", front_page)
    figs_ratios = plot_model_ratios(inv.models, inv.mgr.paraDomain, inv.times, ref_survey=2, cM=0.05, percentiles=(2, 97))
    saveFiguresToPDF(figs_ratios, inv_results + "HRE_raw_ratios.pdf", front_page)


