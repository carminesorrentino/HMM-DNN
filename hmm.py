import json
from hmmlearn import hmm
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


DATASET_PATH = "C:/Users/alfre/Desktop/VoxCeleb Demo 10"
JSON_PATH = "JSON/data_5_40.json"
SAMPLES_TO_CONSIDER = 22050
NUM_SPEAKER = 5
NUM_AUDIO = NUM_SPEAKER * 40
SIZE_SPECTROGRAM = 226 #[226 = log_mel_spectrogram ; 44 = log_spectrogram]

def hmm():
    data = load_data(JSON_PATH)

    hmm_models = []
    array_viterbi = []
    start = 0
    end = 40

    for speaker in range(0, NUM_SPEAKER):
        sub_audio = data["MFCCs_delta1_delta2"][start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        #sub_audio = MFCC_np[start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        print("*Sub Audio", sub_audio[0], len(sub_audio))
        label = data["labels"][start]  # recupera la label (id_speaker) per i 40 audio considerati

        X = np.array([])  # inizializza array dati
        y_words = []  # inizializza array labels

        for audio in sub_audio:
            print("AUDIO ", audio)
            if len(X) == 0:
                #X = sub_audio
                X = audio
            else:
                #X = np.append(X, sub_audio, axis=0)
                X = np.append(X, audio, axis=0)
                print("else")

        y_words.append(label)

        # print('X.shape =', X.shape)
        print("X len ", len(X))
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))

        log, viterbi = hmm_trainer.model.decode(X, algorithm='viterbi')
        print("*HMM TRAINER ", viterbi)
        hmm_trainer = None
        array_viterbi.append(viterbi)

        start = start + 40
        end = end + 40

    print("viterbi array", array_viterbi, len(array_viterbi))
    # for item in array_viterbi:
    #     print("item ", item, len(item))

    array_viterbi = np.array(array_viterbi, dtype=np.float64)


    print("Shape before ", len(data['MFCCs_delta1_delta2'][0]), len(data['MFCCs_delta1_delta2'][0][0]))

    big_array = []
    for arr in array_viterbi:
        big_array = np.concatenate([big_array, arr])
        #print("********* BIG ARRAY LEN" , len(big_array))

    #print("len big_array", len(big_array))

    index = 0  # indice per l'audio (array esterno)
    count = 0 #indice per gli array di ciascun audio (array interno)
    for i in range (0, len(big_array)): # per ciascun array presente in array_viterbi
        # count = 0 #indice per gli array di ciascun audio (array interno)
        # print(array_viterbi[i])
        # for j, el in enumerate (array_viterbi[i]):
            #print("i, j, count , index ",i, j, count, index)
        if(count < SIZE_SPECTROGRAM):
            data['MFCCs_delta1_delta2'][index][count].append(big_array[i])
            #data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
            count = count + 1
        else:
            index = index + 1
            count = 0
            data['MFCCs_delta1_delta2'][index][count].append(big_array[i])
            #data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
            count = count + 1
        try:
            print("MFCCs_delta1_delta2[{}][{}] = {} - len = {}".format(index, count, data['MFCCs_delta1_delta2'][index][count], len(data['MFCCs_delta1_delta2'][index][count])))
        except:
            continue
    print("Shape after ", len(data['MFCCs_delta1_delta2'][0]), len(data['MFCCs_delta1_delta2'][0][0] ))

    # save data in json file
    with open('D:/JSON/hmm.json', "w") as fp:
        json.dump(data, fp, indent=4)

    return


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    return data

class HMMTrainer(object):

   def __init__(self, model_name='GaussianHMM', n_components=2, cov_type='diag', n_iter=1000):
       self.model_name = model_name
       self.n_components = n_components
       self.cov_type = cov_type
       self.n_iter = n_iter
       self.models = []

       if self.model_name == 'GaussianHMM':
           self.model = hmm.GaussianHMM(n_components=self.n_components,
                                        covariance_type=self.cov_type, n_iter=self.n_iter)
       else:
           raise TypeError('Invalid model type')

   def train(self, X):
       np.seterr(all='ignore')
       self.models.append(self.model.fit(X))


   def get_score(self, input_data):
       return self.model.score(input_data)


if __name__ == "__main__":
    hmm()