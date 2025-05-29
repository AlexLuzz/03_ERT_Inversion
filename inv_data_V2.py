import pygimli as pg
from pygimli.physics import ert
from Tools.tools import *
import pandas as pd


folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/"

mesh_file = folder + "Grids/" + "BB_grid_coarse.bms"

df = pd.read_csv('C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/fused_SAS4000_OhmPi_20250526.csv', sep=';')

# Flood test
df = df[df['SurveyDate'] > '2024-11-21 00:00:00']
df = df[df['SurveyDate'] < '2024-11-24 00:00:00']
df = df[df['I(mA)'] > 10]
df = df[~df[['B', 'A', 'M', 'N']].isin([18]).any(axis=1)]
df = df[~df[['B', 'A', 'M', 'N']].isin([34]).any(axis=1)]

unique_surveys = df["Survey"].unique()
selected_surveys = unique_surveys[::4]
df = df[df["Survey"].isin(selected_surveys)]

DATA, survey_dates, _  = create_crosshole_data_df(df)

# Convert survey_dates from strings to datetime objects
survey_dates_datatime = [datetime.strptime(date, '%Y-%m-%d %H:%M') for date in survey_dates]

mesh = pg.load(mesh_file)
inv = ert.TimelapseERT(DATA=DATA, times=survey_dates_datatime, mesh=mesh)
inv.saveData(f"{folder}/Clustering/data", masknan=True)

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

inv.saveResults(basename=f"{folder}/Clustering/test")
figs = plot_models(inv.models, inv.mgr.paraDomain, inv.times, percentiles=(2, 95))
saveFiguresToPDF(figs, f"{folder}/Clustering/models_test_18_34_V2.pdf", front_page)
figs_ratios = plot_model_ratios(inv.models, inv.mgr.paraDomain, inv.times, ref_survey=2, cM=0.05, percentiles=(2, 97))
saveFiguresToPDF(figs_ratios, f"{folder}/Clustering/ratios_test_18_34_V2.pdf", front_page)


