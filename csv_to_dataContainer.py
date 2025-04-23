import pandas as pd
import numpy as np
from pygimli.physics import ert

def create_dataContainers_from_csv(amp_data, file = 'D:/01-Coding/01_BB_ERT/04_DataContainers/'):
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
    def electrodes_positions(n_columns = 8, n_lines=8, h_spacing=0.85, v_spacing=0.2, depth_start=-0.3):
        pos = np.zeros((n_columns * n_lines, 2))
        for col in range(n_columns):
            for line in range(n_lines):
                x = round(h_spacing * (n_columns - 1) / 2 - h_spacing * col, 3)
                y = round(depth_start - line * v_spacing, 3)
                pos[col * n_lines + line] = [x, y]
        positions_created = True
        return pos  # Return the calculated positions
    
    # Initialize lists to hold positions and measurements
    elec_pos = electrodes_positions()
    
    DATA = []  # List to store data containers
    survey_dates = []  # List to store survey dates

    # Get unique surveys in amp_data
    unique_surveys = amp_data['Survey'].unique()

    for survey_number in unique_surveys:
        survey = amp_data[amp_data['Survey'] == survey_number]  # Filter rows for the current survey
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
            chargeability = row.get("chargeability", np.nan)
            if not pd.isna(chargeability):
                ip = float(chargeability)
            else:
                ip = 0

            measurements.append([
                float(row['A']),  # A electrode x
                float(row['B']),  # B electrode x
                float(row['M']),  # M electrode x
                float(row['N']),  # N electrode x
                float(row['Res.(ohm)']),  # rho_a
                error_value,  # err
                ip,  # Computed chargeability
            ])

        # Convert measurements to a numpy array
        measurements = np.array(measurements)
        
        # Prepare the data container
        data = ert.createData(elecs=elec_pos, schemeName='uk')

        # Assign data to the container
        data['a'] = measurements[:, 0].astype(float) - 1
        data['b'] = measurements[:, 1].astype(float) - 1
        data['m'] = measurements[:, 2].astype(float) - 1
        data['n'] = measurements[:, 3].astype(float) - 1
        data['k'] = ert.createGeometricFactors(data)
        data['rhoa'] = measurements[:, 4].astype(float) * data['k']
        data['err'] = measurements[:, 5].astype(float)
        data['ip'] = measurements[:, 6].astype(float)  # Computed chargeability

        date_survey = pd.to_datetime(survey['SurveyDate'].iloc[0])
        
        data.save(f"{file}data_{date_survey.strftime('%d_%m_%Y_%H_%M')}"+'.ohm')

        # Append the SurveyDate for this survey (taking the first row's 'SurveyDate' as the date for the entire survey)
        survey_dates.append(survey['SurveyDate'].iloc[0])

    return survey_dates

file = 'D:/01-Coding/01_BB_ERT/11_Evaluate_IP_Quality/'

df = pd.read_csv(file + 'FLOOD_12_18_unfiltered.csv', sep=';')

create_dataContainers_from_csv(df)