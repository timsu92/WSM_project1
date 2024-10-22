import os


def folderTxtIterator(folderPath: str):
    for dirpath, _, fileNames in os.walk(folderPath):
        for fName in fileNames:
            if fName.endswith(".txt"):
                with open(os.path.join(dirpath, fName)) as f:
                    yield os.path.join(dirpath, fName), f.read()