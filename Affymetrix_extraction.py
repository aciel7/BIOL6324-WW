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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
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
os.chdir(repopath)
mips = np.load("mips.npy", allow_pickle=True)


os.chdir(repopath+"chp_extracted/")
apm_dict = defaultdict(list)
target_dict = defaultdict(list)
signal_dict = defaultdict(list)

# I hate pandas dataframes; imports all of the files
for key in tqdm(chpfilepath_dict.keys()):
    for filepath in tqdm(chpfilepath_dict[key], leave=False):
        df = pd.read_csv(filepath, sep="\t")
        df.index = df["Probe Set Name"]
        df.drop(df.loc["Center X":]["Probe Set Name"], inplace=True)
        sig = df["Signal"].loc[mips].to_numpy().astype(np.float32)
        sig = sig/np.max(sig) # Normalize the signal
        signal_dict[key].append(sig)
        apm_dict[key].append(df["Detection"].loc[mips])
        target_dict[key].append(key)
    apm_dict[key] = np.array(apm_dict[key])
    target_dict[key] = np.array(target_dict[key])


keys = list(apm_dict.keys())
for key in keys:
    if len(apm_dict[key]) < 30:
        apm_dict.pop(key, None)
        target_dict.pop(key, None)


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
# data_sig = data_class(signal_dict, target_dict)

# del signal_dict
del target_dict
del apm_dict
gc.collect()

    
# %% Get Mutual information for each cancer/control combination
def get_mi_data(data, targs):
    probe_mi_list = []
    for i in tqdm(range(data.shape[1]), leave=False):
        probe_mi_list.append(metrics.adjusted_mutual_info_score(data[:, i], targs))
    return probe_mi_list

probe_mi_dict = {}
for key in tqdm(data.keys()):
    if key != "control":
        sub_data, sub_targets = data[key]
        probe_mi_dict[key] = get_mi_data(sub_data, sub_targets)
del sub_data
del sub_targets
gc.collect()
# %% Print interesting things about mutual information
os.chdir(repopath)
mips = np.empty((0))
num_most_important = 10
for key in probe_mi_dict.keys():
    ranked_probe_importance_indices = np.argsort(probe_mi_dict[key])[::-1]
    ranked_probe_importances = np.sort(probe_mi_dict[key])[::-1]
    most_important_probes = df["Probe Set Name"].iloc[ranked_probe_importance_indices[:300]].to_numpy()
    ranked_probe_importances = ranked_probe_importances[:300]
    mips = np.append(mips, most_important_probes)
    i = 0
    gene_symbols = []
    entrez_ids = []
    importances = []
    table_header = ["Importance", "Gene Symbol", "ENTREZ ID"]
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
    
mips = np.unique(mips)
np.save("auxiliary_files/mips.npy", mips)
np.save("mips.npy", mips)

# %%
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
os.chdir(repopath)
def run_classifier(sub_data, sub_targets, key):
    le = LabelEncoder()
    le.fit(sub_targets)
    enc = OrdinalEncoder()
    enc.fit(sub_data)
    targs = le.transform(sub_targets)
    data = enc.transform(sub_data,).astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(data, targs, test_size=.2, shuffle=True)
    print(np.bincount(y_test))
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train )
    
    bst = XGBClassifier(n_estimators=400, max_depth=50, learning_rate=.005, objective='binary:logistic', n_jobs=21,tree_method="hist", device="cuda" )
    # bst.fit(X_train, y_train, sample_weight = sample_weights)
    bst.fit(X_train, y_train,)
    preds = bst.predict(X_test)
    y_pred = le.inverse_transform(preds)
    y_true = le.inverse_transform(y_test)
    print(key)
    # print(metrics.f1_score(y_true, y_pred))
    # print(metrics.precision_score(y_true, y_pred))
    # print(metrics.recall_score(y_true, y_pred))

    print(metrics.balanced_accuracy_score(y_true, y_pred, adjusted=False))
    
    print()
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")
    plt.title(key)
    plt.savefig("figures/"+key.replace(" ", "_")+"_CM.png")
    # metrics.RocCurveDisplay.from_estimator(bst, X_test, y_test, drop_intermediate=False, plot_chance_level=False, sample_weight=sample_weights)
    
    metrics.RocCurveDisplay.from_estimator(bst, X_test, y_test, drop_intermediate=False, plot_chance_level=True, )
    plt.title(key)
    plt.savefig("figures/"+key+"ROC.png")

    
def run_mlpc(sub_data, sub_targets):
    le = LabelEncoder()
    le.fit(sub_targets)
    enc = OrdinalEncoder()
    enc.fit(sub_data)
    targs = le.transform(sub_targets)
    data = enc.transform(sub_data,).astype(np.int8)
    X_train, X_test, y_train, y_test = train_test_split(data, targs, test_size=.2, shuffle=True)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    
    mlpc = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(120, 10), random_state=1)
    mlpc.fit(X_train, y_train)
    preds = mlpc.predict(X_test)
    y_pred = le.inverse_transform(preds)
    y_true = le.inverse_transform(y_test)
    metrics.accuracy_score(y_true, y_pred)
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")



def run_tf(sub_data, sub_targets):
    le = LabelEncoder()
    le.fit(sub_targets)
    enc = OrdinalEncoder()
    enc.fit(sub_data)
    targs = le.transform(sub_targets)
    data = enc.transform(sub_data,).astype(np.int8)
    x_train, x_test, y_train, y_test = train_test_split(data, targs, test_size=.2, shuffle=True)
    
    ros = RandomOverSampler(random_state=42)
    x_train, y_train = ros.fit_resample(x_train, y_train)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    neg, pos = np.bincount(targs)
    total = neg + pos
    initial_bias = np.log([pos/neg])
    output_bias = tf.keras.initializers.Constant(initial_bias)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid',),      

      # keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),      
    ])
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    METRICS = [
      keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      keras.metrics.MeanSquaredError(name='Brier score'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]
    model.compile(metrics=METRICS, optimizer="adam", loss = loss_fn)
    model.fit(x_train, y_train, epochs=20)
    model.evaluate(x_test,  y_test, verbose=1)
    preds = model.predict(x_test)
    y_pred = le.inverse_transform(np.round(preds).astype(int).reshape(-1))
    y_true = le.inverse_transform(y_test.reshape(-1))
    metrics.accuracy_score(y_true, y_pred)
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, xticks_rotation="vertical")
    return np.round(preds).astype(int).reshape(-1)

for key in tqdm(data.keys()):
    if key != "control":
        sub_data, sub_targets = data[key]
        run_classifier(sub_data, sub_targets, key)
        # run_mlpc(sub_data, sub_targets)
        # preds = run_tf(sub_data, sub_targets)
        

