import os
from tqdm import tqdm

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

all_files = getListOfFiles('subject4')

check = ['s1','s2','s3','s4']
for Path in tqdm(all_files):
    try:
        session = Path.split('\\')[-1].split('_')[2]
    except Exception as e:
        continue
    if session in check:
        if 'train' not in Path:
            newPath = Path.replace("valid","train")
            os.replace(Path, newPath)

    else:
        if 'valid' not in Path:
            newPath = Path.replace("train","valid")
            os.replace(Path, newPath)

