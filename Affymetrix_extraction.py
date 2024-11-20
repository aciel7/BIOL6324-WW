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
from tabulate import tabulate
import copy

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
del title
del accession
gc.collect()
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
# %% Loads in the data and targets for the classifier from the raw dataframes and sticks them in two dicts;
# one for the targets and one for the sample information (whether transcript was absent, present etc. )

signal_list = []
apm_list = []  # absent, present, missing list
target_list = []
os.chdir(repopath+"chp_extracted/")
apm_dict = defaultdict(list)
target_dict = defaultdict(list)

# I hate pandas dataframes; imports all of the files
for key in tqdm(chpfilepath_dict.keys()):
    for filepath in tqdm(chpfilepath_dict[key], leave=False):
        df = pd.read_csv(filepath, sep="\t")
        df.index = df["Probe Set Name"]
        df.drop(df.loc["Center X":]["Probe Set Name"], inplace=True)
        # ctrl_vals = df.loc[control_probes]
        # signal_list.append(df["Signal"].to_numpy().astype(float))
        # apm_list.append(df["Detection"])
        # target_list.append(key)
        # signal_dict[key].append(df["Signal"].to_numpy().astype(float))
        apm_dict[key].append(df["Detection"])
        target_dict[key].append(key)
    apm_dict[key] = np.array(apm_dict[key])
    target_dict[key] = np.array(target_dict[key])
# signal_array = np.array(signal_list)
# apm_array = np.array(apm_list)

# %% this class makes managing the data much easier


class data_class:
    def __init__(self, apm_dict, target_dict):
        self.apm = apm_dict
        self.target = target_dict        
    def __getitem__(self, key):
        sub_data = np.append(self.apm[key], self.apm["control"], axis = 0)
        sub_targets = np.append(self.target[key], self.target["control"])
        return sub_data, sub_targets
    def keys(self):
        return self.apm.keys()


data = data_class(apm_dict, target_dict)


del target_dict
del apm_dict
gc.collect()



# %% Get Mutual information for each cancer/control combination
def get_mi_data(data, targs):
    probe_mi_list = []
    for i in tqdm(range(data.shape[1]), leave=False):
        probe_mi_list.append(metrics.mutual_info_score(data[:, i], targs))
    return probe_mi_list


probe_mi_dict = {}
for key in tqdm(data.keys()):
    if key != "control":
        sub_apm, sub_targets = data[key]
        probe_mi_dict[key] = get_mi_data(sub_apm, sub_targets)
        break
del sub_apm
del sub_targets
gc.collect()
# %% Print interesting things about mutual information


num_most_important = 10

for key in probe_mi_dict.keys():
    ranked_probe_importance_indices = np.argsort(probe_mi_dict[key])[::-1]
    ranked_probe_importances = np.sort(probe_mi_dict[key])[::-1]
    most_important_probes = df["Probe Set Name"].iloc[ranked_probe_importance_indices[:2000]].to_numpy()
    ranked_probe_importances = ranked_probe_importances[:2000]
    
    i = 0
    gene_symbols = []
    entrez_ids = []
    importances = []
    table_header = ["importance ranking", "Gene Symbol", "ENTREZ gene ID"]
    for probe, importance in zip(most_important_probes, ranked_probe_importances):
        psdf = probeset_df.loc[probeset_df["ID"] == probe]
        if type(psdf["Gene Symbol"].to_numpy()[0]) != float:
            importances.append(importance)
            gene_symbols.append(psdf["Gene Symbol"].to_numpy()[0])
            entrez_ids.append(psdf["ENTREZ_GENE_ID"].to_numpy()[0])
            i += 1
        if i == num_most_important:
            break
    print(key)
    print(tabulate(np.array([importances, gene_symbols, entrez_ids]).T, headers=table_header))
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