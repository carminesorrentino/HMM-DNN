import json
from hmmlearn import hmm
import tensorflow as tf
import numpy as np
import pandas as pd
from utils.hmm_lib import HMMTrainer


JSON_PATH = "D:/Speaker recognition da consegnare/JSON/data_90_HMM_delta.json"
SAMPLES_TO_CONSIDER = 22050
NUM_SPEAKER = 90
AUDIO = 90
NUM_AUDIO = NUM_SPEAKER * AUDIO
SIZE_SPECTROGRAM = 226
NUM_FEATURES = 39

def preprocess_dataset(json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    data_dnn = {
        "mapping": [],
        "labels": [],
        "MFCCs_delta1_delta2": []
    }

    data = load_data(JSON_PATH)

    hmm_models = []
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


    for index, audio in enumerate (data["MFCCs_delta1_delta2"][0: NUM_AUDIO]):
        print("Audio ", index, "/", NUM_AUDIO)
        audio = np.array(audio)
        # audio = audio.T

        score_audio = []
        log_likelihood_audio = []

        # for mfcc in audio:s
        for mfcc in range (0, NUM_FEATURES):
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
        print("compute new vector ", i)
        audio = []

        for j in range (0, NUM_FEATURES): # mfcc
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
    with open('D:/Speaker recognition da consegnare/JSON/hmm_deltas.json', "w") as fp:
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



if __name__ == "__main__":
    preprocess_dataset(JSON_PATH)



