import json
import librosa
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np

JSON_PATH = "D:\hmm_test_10.json"


def main():
    data_dnn = {
        "labels": [],
        "MFCCs_delta1_delta2": [],
    }


    data = load_data(JSON_PATH)
    matrix = np.array(data['MFCCs_delta1_delta2'])
    y_words = data['labels']

    # min_max_scaler = preprocessing.MinMaxScaler()
    #
    # scaled_x = min_max_scaler.fit_transform(matrix)

    #input = librosa.util.normalize(matrix)

    scaler = MinMaxScaler3D()
    X = scaler.fit_transform(matrix)


    data_dnn['MFCCs_delta1_delta2'] = X.tolist()
    data_dnn['labels'] = y_words


    print("DATA DNN ", data_dnn['MFCCs_delta1_delta2'])

    print("write data file")
    # save data in json file
    with open('JSON/hmm_test_10_normalize.json', "w") as fp:
        json.dump(data_dnn, fp, indent=4)



    return

def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    return data

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

if __name__ == "__main__":
    main()