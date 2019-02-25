#!/bin/bash
# To run, cd into the directory (make sure the file you want to split is named file.csv) type command `sh csv-splitter.sh`
# You can convert any txt file into a csv file by changing the extension to .csv
# Split your csv (or txt) file at a specific line number by chaning the 10000 in line 17 to any value you'd like.

split -b 100m package.txt snippet


# Look for a file called movies.txt (8gb file, unable to open w/o splitting)
# FILE=$(ls -1 | grep movies.txt)
# NAME=${FILE%%.csv}

# # Save the first line in the file as your header
# head -1 $FILE > header.csv
# # Save the rest of file (line 2 and on) as your data file
# tail -n +2 $FILE > data.csv

# # Split data at every X number of lines
# split -l 10000 data.csv

# # Iterate over each split file
# for a in x??
#     do
#         # Add the header to each new split file
#         cat header.csv $a > $NAME.$a.csv
#     done

# rm data.csv x??
