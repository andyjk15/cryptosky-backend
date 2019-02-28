import csv

csv_list = []

with open('data_collector/tweets.csv','r') as in_file, open('data_collector/dupes_removed.csv','w') as out_file:
    seen = set() # set for fast O(1) amortized lookup
    for line in in_file:
        if line in seen: continue # skip duplicate
            
        seen.add(line)
        out_file.write(line)