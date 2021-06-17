# %%

# For Data Import
import glob
import os
from sys import argv
from dask.config import get
from ipywidgets.widgets.widget_controller import Axis

# For Data Wrangling and Visualizing
import numpy as np
import pandas as pd
import seaborn as sns

# For Reading FITS Data From NASA
from astropy.io import fits
import matplotlib.pyplot as plt

# For Processing the huge amount of Light Curve Data
import multiprocessing as mp
import dask
from dask.distributed import Client
import dask.dataframe as dd
from tqdm.notebook import tqdm
# %%

names_candidate_data = ['loc_rowid', 'EPIC', '2MASS', 'EPIC.Name', 'pl_name', 'k2c_refdisp', 'k2c_reflink', 'Label', 'Campaign']
columns_candidate_data = ['EPIC', 'Label', 'Campaign', 'EPIC.Name']
labels_k2 = pd.read_csv("C:/Users/PC/Documents/School/Data_Science/Repo_Depot/Kepler_Data/Data_Dump/K2-Candidates.txt", header=0, names=names_candidate_data, usecols=columns_candidate_data)

labels_k2["K2_False_Positive"] = np.where(labels_k2["Label"].str.contains("False Positive", case=False), 1, 0)
labels_k2["K2_Confirmed"] = np.where(labels_k2["Label"].str.contains("confirmed", case=False), 1, 0)
labels_k2["K2_Candidate"] = np.where(labels_k2["Label"].str.contains("candidate", case=False), 1, 0)

labels_k2 = labels_k2.drop(["Label"], axis=1)

labels_k2 = labels_k2[labels_k2["Campaign"].notna()]

labels_k2["EPIC"] = labels_k2["EPIC"].str.extract("(\d+)").astype(int)
labels_k2["Campaign"] = labels_k2["Campaign"].astype(int)

labels_k2["Event"] = labels_k2["EPIC.Name"].str[-2:].astype(int)

labels_k2.drop_duplicates(inplace=True)
labels_k2.set_index(['EPIC', 'Campaign'], drop=False, inplace=True)

labels_k2['disp'] = np.where(labels_k2.K2_False_Positive==0, 1, 0)



#%%
light_curves = pd.concat([pd.read_fwf(file, header=None, names=[0, 1, "Link"]) for file in glob.glob("C:/Users/PC/Documents/School/Data_Science/Repo_Depot/Kepler_Data/Data_Dump/wgets/Light_Curves/*")], ignore_index=True).drop([0,1], axis=1)

light_curves["Campaign"] = light_curves["Link"].str.extract("(?:s/c)(\d{1,2})").astype(int)
light_curves["EPIC"] = light_curves["Link"].str.extract("(?:ktwo)(\d+)").astype(int)
light_curves["Cadence"] = light_curves["Link"].str[-8:-5]

light_curves = light_curves.loc[light_curves.Cadence == "llc"]

light_curves = light_curves.set_index(["EPIC", "Campaign"])

names_target_data = ["EPIC", "2MASS", "Campaign", "Obj_Type", "rastr",
                     "decstr","k2_propid","Distance","k2_disterr1","k2_disterr2",
                     "k2_teff","k2_tefferr1","k2_tefferr2","Stellar_Radius","k2_raderr1",
                     "k2_raderr2","Stellar_Mass","k2_masserr1","k2_masserr2","k2_kepmag",
                     "k2_kepmagerr","k2_kepmagflag","k2_vjmag","k2_vjmagerr","k2_kmag","k2_kmagerr"]
columns_target_data = ["EPIC", "2MASS", "Campaign", "Obj_Type"]
target_data = pd.read_csv("C:/Users/PC/Documents/School/Data_Science/Repo_Depot/Kepler_Data/Data_Dump/K2-Targets.txt", header=0,names=names_target_data, usecols=columns_target_data)

target_data=target_data[~target_data["Campaign"].isin(["E", "11", 11])]
target_data["Campaign"] = target_data["Campaign"].astype(int)
target_data=target_data[~target_data["EPIC"].isin([";"])]

target_data["EPIC"]=target_data["EPIC"].astype(int)

target_data=target_data.set_index(["EPIC", "Campaign"])

target_light_curves = target_data.merge(light_curves, left_index=True, right_index=True)

target_light_curves.drop(['2MASS', 'Obj_Type', 'Cadence'], axis=1, inplace=True)

target_light_curves['disp'] = np.where(target_light_curves.index.isin(labels_k2.loc[labels_k2.disp==1].index.unique()), 1, 0)

#%%
campaign_positive_hits = {key: 100*(target_light_curves.xs(key, level=1).disp.value_counts()[1]/target_light_curves.xs(key, level=1).disp.value_counts()[0]) for key in set(target_light_curves.loc[target_light_curves['disp']==1].index.get_level_values(1))}

besthits = sorted(campaign_positive_hits, key=campaign_positive_hits.get, reverse=True)

input_subset = target_light_curves.loc[pd.IndexSlice[:, besthits[:4]], :]
#%%
trash = [target_data, target_light_curves, columns_target_data, names_target_data, \
    light_curves, labels_k2, names_candidate_data, columns_candidate_data]
del target_data, target_light_curves, columns_target_data, names_target_data, \
    light_curves, labels_k2, names_candidate_data, columns_candidate_data, \
        trash

#%%
def get_flux_data (link):
    global pbar
    pbar.update(1)
    with fits.open(link, mode="readonly") as hdul:
        curve = hdul[1].data['PDCSAP_FLUX']
        mask = np.isnan(curve)
        curve[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), curve[~mask])
        return curve
#%%
def map_to_flux (df):
    return df['Link'].map(get_flux_data)

# %%
with tqdm(total=len(input_subset)) as pbar:
    result = input_subset['Link'].map(get_flux_data)

2#%%
input_subset
# %%
import gc
gc.collect()

# %%
result
# %%
input_subset = input_subset.merge(pd.DataFrame(result.tolist(), index=result.index), left_index=True, right_index=True)
trash = [result,]
del result, trash
# %%
input_subset.shape
# %%
input_subset.fillna(0)
# %%
input_subset.to_csv('inpput_subset.csv')
# %%
