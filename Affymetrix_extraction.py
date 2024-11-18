#SET THE FOLLOWING TO THE TOP LEVEL DIRECTORY OF THE REPO
repopath = "/home/william/Desktop/bioinfo/"

# %%
from Bio.Affy import CelFile
import os
import pandas as pd
from collections import defaultdict
class afmx_sample:
    def __init__(self, path, cancer_type):
        self.cancer_type = cancer_type
        with open(path, "rb") as file:    
            celdata = CelFile.read(file)
            

# %% get a list of all the raw files paths and 
# split them into CEL and mas5 files



os.chdir(repopath + "raw_data")
file_list = os.listdir()
celfile_list = []
mas5file_list =[]
for file in file_list:
    if "CEL" in file:
        celfile_list.append(file)
    elif "mas5" in file:
        mas5file_list.append(file)
# %% script to dump all chp files to plaintext
os.chdir("")




# %%
def get_filepath_from_accession(accession, file_list):
    return next((filepath for filepath in file_list if accession in filepath))

celfile_dict = defaultdict(list)
sample_info_df = pd.read_csv("../sample.csv")
for title, accession in zip(sample_info_df["Title"], sample_info_df["Accession"]):
    cancer_type = title.split(",")[1].split("-")[0].strip()
    celfile_dict[cancer_type].append(get_filepath_from_accession(accession, celfile_list))

# %% Open all celfiles and 
celdata_list = []
num_excpts = 0
for path in celfile_list:
    with open(path, "rb") as file: 
        try:
            print(path)
            celdata = CelFile.read(file)
        except:
            print("exception " + path)
            num_excpts +=1
            break
# %%
sample_list = []

for cancer_type in celfile_dict.keys():
    for path in celfile_dict[cancer_type]:
        pass
    pass