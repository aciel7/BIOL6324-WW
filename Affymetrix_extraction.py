# SET THE FOLLOWING TO THE TOP LEVEL DIRECTORY OF THE REPO
repopath = "/home/william/Desktop/bioinfo/"

# %%
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import gc
# %% script to dump all chp files to tsv plaintext
os.chdir(repopath)
os.system(command="export PATH="+repopath+"aptools/bin:$PATH; ./extract_chp.sh")

# %% get a list of the extracted CHP TSV files and put in list
os.chdir(repopath+"chp_extracted")
chpfilepath_list = os.listdir()
chpfilepath_list.sort()
# %% split all of the chp files into a dict with the key being the cancer type for easier looping


def get_filepath_from_accession(accession, file_list):
    return next((filepath for filepath in file_list if accession in filepath))


chpfilepath_dict = defaultdict(list)
sample_info_df = pd.read_csv(repopath+"auxiliary_files/samples.csv")
for title, accession in zip(sample_info_df["Title"], sample_info_df["Accession"]):
    cancer_type = title.split(",")[1].split("-")[0].strip()
    chpfilepath_dict[cancer_type].append(get_filepath_from_accession(accession, chpfilepath_list))
# %% load in the probeset tsv data into a dataframe
os.chdir(repopath)
probeset_df = pd.read_csv("./auxiliary_files/GPL570_probeset.tsv", sep="\t")
control_probes = probeset_df[probeset_df["SPOT_ID"] == '--Control']["ID"]

# %% normalizes intensity values

# os.chdir(repopath+"chp_extracted/")
# for key in chpfilepath_dict.keys():
#     for filepath in chpfilepath_dict[key]:
#         df = pd.read_csv(filepath, sep="\t")
#         df.index = df["Probe Set Name"]
#         ctrl_vals = df.loc[control_probes]
#         break
#     break
# %% Loads in the data and targets for the classifier from the raw dataframes

signal_list = []
apm_list = []  # absent, present, missing list
target_list = []
os.chdir(repopath+"chp_extracted/")

# I hate pandas dataframes; imports all of the files
for key in tqdm(chpfilepath_dict.keys()):
    for filepath in tqdm(chpfilepath_dict[key], leave=False):
        df = pd.read_csv(filepath, sep="\t")
        df.index = df["Probe Set Name"]
        df.drop(df.loc["Center X":]["Probe Set Name"], inplace=True)
        ctrl_vals = df.loc[control_probes]
        signal_list.append(df["Signal"].to_numpy().astype(float))
        apm_list.append(df["Detection"])
        target_list.append(key)
        # print(df["Probe Set Name"])
        
signal_array = np.array(signal_list)
apm_array = np.array(apm_list)

del apm_list
del signal_list
gc.collect()
# %% Mutual information
probe_mi_list = []
for i in tqdm(range(apm_array.shape[1])):
    probe_mi_list.append(metrics.mutual_info_score(apm_array[:, i], target_list))
# %%
ranked_probe_importance_indices = np.argsort(probe_mi_list, )[::-1]
ranked_probe_importances = np.sort(probe_mi_list, )[::-1]
most_important_probes = df["Probe Set Name"].iloc[ranked_probe_importance_indices[:100]].to_numpy()
ranked_probe_importances = ranked_probe_importances[:100]
for probe, importance in zip(most_important_probes, ranked_probe_importances):
    psdf = probeset_df.loc[probeset_df["ID"] == probe]
    print("Importance:  " + str(importance))
    print(psdf.at["Gene Title"])
    print()
# %% organizes the data into training and test
# TODO: add shuffling, cross validation
import random

# random.shuffle(apm_list)
le = LabelEncoder()
le.fit(target_list)
enc = OrdinalEncoder()
enc.fit(apm_list)
targs = le.transform(target_list)
data = enc.transform(apm_list, ).astype(np.int8)

# %%
X_train, X_test, y_train, y_test = train_test_split(data, targs, test_size=.2, shuffle=True)
# %% XGBOOST 
bst = XGBClassifier(n_estimators=100, max_depth=20, learning_rate=.1, objective='binary:logistic', n_jobs=21)
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
y_pred = le.inverse_transform(preds)
y_true = le.inverse_transform(y_test)
# %%
metrics.accuracy_score(y_true, y_pred)
# %% Plot confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")
# metrics.RocCurveDisplay.from_predictions(y_test, preds, plot_chance_level=True,  )