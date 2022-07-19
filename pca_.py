import json

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition


DATA_PATH = "D:/Speaker recognition da consegnare/JSON/data_90_HMMTEST.json"
NUM_SPEAKER = 90
NUM_AUDIO = NUM_SPEAKER * 90
N_COMPONENTS = 8
SIZE_SPECTROGRAM = 226

def pca(DATA_PATH):

    data = load_data(DATA_PATH)

    data['MFCCs_delta1_delta2'] = np.array(data['MFCCs_delta1_delta2'])
    print("Shape MFCCs", data['MFCCs_delta1_delta2'].shape)
    nsamples, nx, ny = data['MFCCs_delta1_delta2'].shape
    data['MFCCs_delta1_delta2'] = data['MFCCs_delta1_delta2'].reshape((nsamples * nx, ny))

    matrix = data['MFCCs_delta1_delta2']
    scalare = StandardScaler()
    matrix = pd.DataFrame(scalare.fit_transform(matrix))
    print("*** ", matrix)
    pca = sklearn.decomposition.PCA().fit(matrix)
    # print('****pca', pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.plot(pca.explained_variance_ratio_, 'o-')
    plt.xlabel('number of components', )
    plt.ylabel('cumulative explained variance')
    plt.show()

    ##Riduzione della dimensionalità lineare mediante Singular Value Decomposition dei dati per proiettarli in uno spazio dimensionale inferiore.
    # I dati di input vengono centrati ma non ridimensionati (questo spiega l'operazione di fit_transform precedente).

    pca = sklearn.decomposition.PCA(n_components=N_COMPONENTS)
    extraction_pca = pd.DataFrame(pca.fit_transform(matrix))
    print("****** Extraction_pca ", extraction_pca)

    #data["MFCCs"] = extraction_pca
    data["MFCCs_delta1_delta2"] = extraction_pca.to_numpy()
    data["MFCCs_delta1_delta2"] = data["MFCCs_delta1_delta2"].reshape(NUM_AUDIO, SIZE_SPECTROGRAM, N_COMPONENTS)
    print("** SHAPE ", data['MFCCs_delta1_delta2'].shape)
    data["MFCCs_delta1_delta2"] = data["MFCCs_delta1_delta2"].tolist()

    # # save data in json file
    # with open('JSON/pca_hmm.json', "w") as fp:
    #     json.dump(data, fp, indent=4)

    return


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    return data


if __name__ == "__main__":
    pca(DATA_PATH)