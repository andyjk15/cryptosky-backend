import csv, os
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('data_collector/tweets.csv')

data.rename(columns={'created_at' : 'classes'}, inplace=True)

for i, row in tqdm(data.iterrows()):
    data.loc[i, "classes"] = "null"

data.to_csv('data_collector/spam_ham_unclean.csv', sep='\t', encoding='utf-8', index=False)

with open('data_collector/spam_ham_unclean.csv','r') as in_file, open('data_collector/spam_ham_alt.csv','w') as out_file:
    seen = set() # set for fast O(1) amortized lookup
    for line in tqdm(in_file):
        if line in seen: continue # skip duplicate
            
        seen.add(line)
        out_file.write(line)

os.remove('data_collector/spam_ham_unclean.csv')