import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of sound


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length = 512, n_fft = 2048):
    # data dictionairy
    data = {
        "mappings" : [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    #sub-dirrectories
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            cathegory = dirpath.split("/")[-1] # -> [dataset, down]
            data["mappings"].append(cathegory)

            #loop through all filenames to extract MFCC's
            for f in filenames:
                file_path = os.path.join(dirpath,f) #filepath
                signal, sr = librosa.load(file_path) #loading audio file
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    MFCCs = librosa.feature.mfcc(signal,n_mfcc=13, hop_length = 512, n_fft = 2048)

                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)

    with open(json_path ,"w") as fp:
        json.dump(data, fp, indent = 4)

if __name__ == "__main__"
    prepare_dataset(DATASET_PATH,JSON_PATH)