import librosa
import os
import json
import noisereduce as nr
from hmmlearn import hmm
import sklearn.utils
import tensorflow as tf
#DATASET_PATH = "dataset"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats


DATASET_PATH = "C:/Users/alfre/Desktop/VoxCeleb Demo 10"
#JSON_PATH = "JSON/data_90_HMMTEST.json"
JSON_PATH = "JSON/data_5_40.json"
SAMPLES_TO_CONSIDER = 22050
NUM_SPEAKER = 90
AUDIO = 90
NUM_AUDIO = NUM_SPEAKER * AUDIO
SIZE_SPECTROGRAM = 226 #[226 = log_mel_spectrogram ; 44 = log_spectrogram]

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):



    # Split train e test set
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    data_dnn = {
        "mapping": [],
        "labels": [],
        "MFCCs_delta1_delta2": [],
        # "Delta1": [],
        # "Delta2": [],
        #"files": []
    }

    data = load_data(JSON_PATH)

    hmm_models = []
    array_viterbi = []
    start = 0
    end = AUDIO
    y_words = []  # inizializza array labels
    #### Training
    for speaker in range(0, NUM_SPEAKER):
        print("speaker", speaker)
        sub_audio = data["MFCCs_delta1_delta2"][start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        # sub_audio = MFCC_np[start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        #print("*Sub Audio", sub_audio[0], len(sub_audio))
        label = data["labels"][start]  # recupera la label (id_speaker) per i 40 audio considerati

        X = np.array([])  # inizializza array dati

        hmm_trainer = HMMTrainer()
        for audio in sub_audio:
            hmm_trainer.train(audio)

            y_words.append(label)

        hmm_models.append((hmm_trainer, label))



        # print('X.shape =', X.shape)
        #print("X len ", len(X))


        start = start + AUDIO
        end = end + AUDIO

    ## passare tutti gli audio a ciascun modello acustico HMM
    ### dobbiamo ottenere un array di vettori per ogni speaker
    y = []
    scores = []
    log_likelihood_all = []


    for audio in data["MFCCs_delta1_delta2"]:
        audio = np.array(audio)
        # audio = audio.T

        score_audio = []
        log_likelihood_audio = []

        # for mfcc in audio:
        for mfcc in range (0, 13):
            #print("calcola mfcc ", mfcc)
            start = mfcc
            end = mfcc + 1

            score_speaker = []
            log_likelihood_speaker = []

            for hmm, labels in hmm_models:
                #print("len mfcc ", len(mfcc))
                s = hmm.get_score(audio[:,start:end])
                score_speaker.append(s)

                log_likelihood, states = hmm.model.decode(audio[:,start:end], algorithm='viterbi')
                log_likelihood_speaker.append(log_likelihood)

                # print("score_speaker ", score_speaker, len(score_speaker))
                # print("log_likelihood_speaker ", log_likelihood_speaker, len(log_likelihood_speaker))

            score_audio.append(score_speaker)
            log_likelihood_audio.append(log_likelihood_speaker)


        scores.append(score_audio)
        log_likelihood_all.append(log_likelihood_audio)

    # print("Scores ", scores, len(scores))
    # print("log_likelihood_all ", log_likelihood_all, len(log_likelihood_all))



    input = []
    for i in range(0, len(scores)): #audio
        audio = []

        for j in range (0, 13): # mfcc
            array = []

            for z in range (0, NUM_SPEAKER): #numero di speaker
                # print("i / j ", i , j)
                res = log_likelihood_all[i][j][z] - scores[i][j][z]
                # print("res ", res)
                array.append(res)

            audio.append(array)

        input.append(audio)

    print("input ", input, len(input))
    print("labels ", y_words, len(y_words))

    data_dnn['MFCCs_delta1_delta2'] = input
    data_dnn['labels'] = y_words

    print("write data file")
    #save data in json file
    with open('JSON/hmm_test_10.json', "w") as fp:
        json.dump(data_dnn, fp, indent=4)

    return

def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
           row.append(matrix[i][j])
        matrix_T.append(row)

    return matrix_T

def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    return data

class HMMTrainer(object):
   # def __init__(self, model_name='GaussianHMM', n_components=6):
   #   self.model_name = model_name
   #   self.n_components = n_components
   #
   #   self.models = []
   #   if self.model_name == 'GaussianHMM':
   #      self.model=hmm.GaussianHMM(n_components=6)
   #   else:
   #      print("Please choose GaussianHMM")

   def __init__(self, model_name='GaussianHMM', n_components=3, cov_type='diag', n_iter=1000):
       self.model_name = model_name
       self.n_components = n_components
       self.cov_type = cov_type
       self.n_iter = n_iter
       self.models = []

       if self.model_name == 'GaussianHMM':
           self.model = hmm.GMMHMM(n_components=self.n_components, algorithm='viterbi',  n_mix=64,
                                   startprob_prior=1.0,  transmat_prior=1.0, params='',
                                        covariance_type=self.cov_type, n_iter=self.n_iter)
       else:
           raise TypeError('Invalid model type')

   def train(self, X):
       np.seterr(all='ignore')
       self.models.append(self.model.fit(X))


   def get_score(self, input_data):
       return self.model.score(input_data)



def getAudio():

    df = pd.read_csv("C:/Users/alfre/PycharmProjects/speaker_recognition/csv/resized_dataset_path.csv")
    #print("df ", df[0, 1].values)
    labels = df.loc[:, 'labels'].values
    path = df.loc[:, 'path'].values

    data = {
        "labels" : labels,
        "path" : path
    }
    print("DATA ", data)
    return data




def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation)
                        #,callbacks=[earlystop_callback]
                        )
    return history

#
# def plot_history(history):
#     """Plots accuracy/loss for training/validation set as a function of the epochs
#     :param history: Training history of model
#     :return:
#     """
#
#     fig, axs = plt.subplots(2)
#
#     # create accuracy subplot
#     axs[0].plot(history.history["accuracy"], label="accuracy")
#     axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
#     axs[0].set_ylabel("Accuracy")
#     axs[0].legend(loc="lower right")
#     axs[0].set_title("Accuracy evaluation")
#
#     # create loss subplot
#     axs[1].plot(history.history["loss"], label="loss")
#     axs[1].plot(history.history['val_loss'], label="val_loss")
#     axs[1].set_xlabel("Epoch")
#     axs[1].set_ylabel("Loss")
#     axs[1].legend(loc="upper right")
#     axs[1].set_title("Loss evaluation")
#
#     plt.show()
#     plt.savefig("Acc E"+EPOCHS.__str__()+ "-BS"+BATCH_SIZE.__str__())
#     plt.close()
#
#     return

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)



