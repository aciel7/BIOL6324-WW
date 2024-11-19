#SET THE FOLLOWING TO THE TOP LEVEL DIRECTORY OF THE REPO
repopath = "/home/william/Desktop/bioinfo/"

# %%
import os
import pandas as pd
from collections import defaultdict
            
#%% script to dump all chp files to tsv plaintext
os.chdir(repopath)
os.system(command="export PATH="+repopath+"aptools/bin:$PATH; ./extract_chp.sh")

# %% get a list of the extracted CHP TSV files and put in list
os.chdir(repopath+"chp_extracted")
chpfile_list = os.listdir()

# %% split all of the chp files into a dict with the key being the cancer type for easier looping

def get_filepath_from_accession(accession, file_list):
    return next((filepath for filepath in file_list if accession in filepath))

chpfile_dict = defaultdict(list)
sample_info_df = pd.read_csv(repopath+"auxiliary_files/samples.csv")
for title, accession in zip(sample_info_df["Title"], sample_info_df["Accession"]):
    cancer_type = title.split(",")[1].split("-")[0].strip()
    chpfile_dict[cancer_type].append(get_filepath_from_accession(accession, chpfile_list))
# %% load in the probeset tsv data into a dataframe
os.chdir(repopath)
probeset_df = pd.read_csv("./auxiliary_files/GPL570_probeset.tsv", sep="\t")

# %% 
