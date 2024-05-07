import os
dataset_path = "./data/Data/genres_original"
for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    print(i, dirpath,dirnames,filenames)