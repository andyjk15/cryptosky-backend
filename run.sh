#!/bin/bash

#Script to start data collection processes

#Check if required files are in the correct dirs
file_paths=("data_collector/prices/price_collector.py" "data_collector/twitter/tweet_collector.py")

for path in ${file_paths[*]}
do
    if [ ! -e $path ] 
    then
        echo "$path - Not Found"
        echo "Ensure that the scripts are located in those directories"
    else
        echo "BOOP"
        nohup python $path &
        echo "Starting $path"
    fi
done