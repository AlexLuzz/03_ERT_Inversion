from datetime import datetime
import pandas as pd
from pygimli.physics import ert
from tools import create_crosshole_data_df

df = pd.read_csv('C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/fused_SAS4000_OhmPi.csv', sep=';')

# Flood test
df = df[df['SurveyDate'] > '2024-11-12 12:00:00']
df = df[df['SurveyDate'] < '2024-11-18 12:00:00']
df = df[df['I(mA)'] > 10]
df = df[~df[['B', 'A', 'M', 'N']].isin([18]).any(axis=1)]
df = df[~df[['B', 'A', 'M', 'N']].isin([58]).any(axis=1)]
df = df[~df[['B', 'A', 'M', 'N']].isin([58]).any(axis=1)]

# Raining Even
#df = df[df['SurveyDate'] > '2024-11-21 00:00:00']
#df = df[df['SurveyDate'] < '2024-11-26 00:00:00']

# Période de redoux
#df = df[df['SurveyDate'] > '2025-02-15 00:00:00']
#df = df[df['meas'] != 'Unknown']

DATA, survey_dates, _  = create_crosshole_data_df(df)

# Convert survey_dates from strings to datetime objects
survey_dates_datatime = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in survey_dates]

file = 'C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/TLERT_Data/'
date_1 = survey_dates_datatime[0].strftime('%m-%d_%Hh')
date_2 = survey_dates_datatime[-1].strftime('%m-%d_%Hh')

#filename = f"{file}REDOUX_{date_1}_{date_2}"
filename = f"{file}FLOOD_sans_18_37_58"
inv = ert.TimelapseERT(DATA=DATA, times=survey_dates_datatime)
inv.saveData(filename, masknan=True)

#save_data(DATA, survey_dates_datatime, filename)


