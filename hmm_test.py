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
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 5
#LEARNING_RATE = 0.0001
LEARNING_RATE = 0.0001
NUM_LABELS = 300

DATASET_PATH = "C:/Users/alfre/Desktop/VoxCeleb Demo 10"
JSON_PATH = "JSON/data_5_40.json"
SAMPLES_TO_CONSIDER = 22050
NUM_SPEAKER = 5
NUM_AUDIO = NUM_SPEAKER * 40
SIZE_SPECTROGRAM = 226 #[226 = log_mel_spectrogram ; 44 = log_spectrogram]

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    data = load_data(JSON_PATH)
    #print("data ", data['MFCCs_delta1_delta2']) #contiene solo MFCCs
    # X = data['MFCCs_delta1_delta2']
    # y = data['labels']

    hmm_models = []
    array_viterbi = []
    start = 0
    end = 40

    # Split train e test set
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    data = load_data(JSON_PATH)

    hmm_models = []
    array_viterbi = []
    start = 0
    end = 40
    y_words = []  # inizializza array labels
    #### Training
    for speaker in range(0, NUM_SPEAKER):
        sub_audio = data["MFCCs_delta1_delta2"][start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        # sub_audio = MFCC_np[start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
        print("*Sub Audio", sub_audio[0], len(sub_audio))
        label = data["labels"][start]  # recupera la label (id_speaker) per i 40 audio considerati

        X = np.array([])  # inizializza array dati

        hmm_trainer = HMMTrainer()
        for audio in sub_audio:
            hmm_trainer.train(audio)
            # hmm_models.append((hmm_trainer, label))
            # print("score", hmm_trainer.get_score(audio))
            # print("AUDIO ", audio)
            # if len(X) == 0:
            #     # X = sub_audio
            #     X = audio
            # else:
            #     # X = np.append(X, sub_audio, axis=0)
            #     X = np.append(X, audio, axis=0)
            #     print("else")
            y_words.append(label)

        hmm_models.append((hmm_trainer, label))



        # print('X.shape =', X.shape)
        print("X len ", len(X))
        # hmm_trainer = HMMTrainer()
        # hmm_trainer.train(X)
        # hmm_models.append((hmm_trainer, label))
        #
        # print("score", hmm_trainer.get_score(X))


        # log, viterbi = hmm_trainer.model.decode(X, algorithm='viterbi')
        # print("*HMM TRAINER ", viterbi)
        # hmm_trainer = None
        # array_viterbi.append(viterbi)

        start = start + 40
        end = end + 40

    ## passare tutti gli audio a ciascun modello acustico HMM
    ### dobbiamo ottenere un array di vettori per ogni speaker
    y = []
    scores = []
    log_likelihood_all = []
    for audio in data["MFCCs_delta1_delta2"]:
    #for hmm, labels in hmm_models:
        score_speaker = []
        log_likelihood_speaker = []
        #for audio in data["MFCCs_delta1_delta2"]:
        for hmm, labels in hmm_models:
            s = hmm.get_score(audio)
            score_speaker.append(s)

            log_likelihood, states = hmm.model.decode(audio, algorithm='viterbi')
            log_likelihood_speaker.append(log_likelihood)

        scores.append(score_speaker)
        log_likelihood_all.append(log_likelihood_speaker)

        #y.append(labels)
        # print("Score ", labels, score_speaker, len(score_speaker))
        # print("log_likelihood_speaker ", labels, log_likelihood_speaker, len(log_likelihood_speaker))


    print("Scores ", scores, len(scores))
    print("log_likelihood_all ", log_likelihood_all, len(log_likelihood_all))

    #log, viterbi = hmm_trainer.model.decode(X, algorithm='viterbi')

    input = []
    for i in range(0, len(scores)): #audio
        array = []
        for j in range (0, 5): # speaker
            # print("i / j ", i , j)
            res = log_likelihood_all[i][j] - scores[i][j]
            # print("res ", res)
            array.append(res)

        input.append([array])

    print("input ", input, len(input))
    print("labels ", y_words, len(y_words))

    ############## Rete DNN

    X = np.array(input)
    y = np.array(y_words)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

    print("train " , y_train, len(y_train))
    print("val " , y_validation, len(y_validation))
    print("test " , y_test, len(y_test))
    print("X shape", X_train.shape)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    model = tf.keras.models.Sequential()

    # # 1st conv layer
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    #
    # # 2nd conv layer
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # tf.keras.layers.Dropout(0.3)

    # 3rd conv layer
    # model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))
    # tf.keras.layers.Dropout(0.3)

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    #tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=0.0001)

    # compile model
    model.compile(optimizer=optimiser,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # Evaluating the model on the training and testing set
    # training_loss, training_acc = model.evaluate(X_train, y_train, verbose=0)
    # print("\nTraining loss: {}, training accuracy: {}".format(training_loss, 100*training_acc))

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100 * test_acc))
    # print("viterbi array", array_viterbi, len(array_viterbi))
    # # for item in array_viterbi:
    # #     print("item ", item, len(item))
    #
    # array_viterbi = np.array(array_viterbi, dtype=np.float64)
    #
    # print("Shape before ", len(data['MFCCs_delta1_delta2'][0]), len(data['MFCCs_delta1_delta2'][0][0]))

###############################
    # big_array = []
    # for arr in array_viterbi:
    #     big_array = np.concatenate([big_array, arr])
    #     print("********* BIG ARRAY LEN", len(big_array))
    #
    # print("len big_array", len(big_array))
    #
    # index = 0  # indice per l'audio (array esterno)
    # count = 0  # indice per gli array di ciascun audio (array interno)
    # for i in range(0, len(big_array)):  # per ciascun array presente in array_viterbi
    #     # count = 0 #indice per gli array di ciascun audio (array interno)
    #     # print(array_viterbi[i])
    #     # for j, el in enumerate (array_viterbi[i]):
    #     # print("i, j, count , index ",i, j, count, index)
    #     if (count < SIZE_SPECTROGRAM):
    #         data['MFCCs_delta1_delta2'][index][count].append(big_array[i])
    #         # data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
    #         count = count + 1
    #     else:
    #         index = index + 1
    #         count = 0
    #         data['MFCCs_delta1_delta2'][index][count].append(big_array[i])
    #         # data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
    #         count = count + 1
    #     try:
    #         print("MFCCs[{}][{}] = {} - len = {}".format(index, count, data['MFCCs_delta1_delta2'][index][count],
    #                                                      len(data['MFCCs_delta1_delta2'][index][count])))
    #     except:
    #         continue
    # print("Shape after ", len(data['MFCCs_delta1_delta2'][0]), len(data['MFCCs_delta1_delta2'][0][0]))
    ######################## HMM
    ##############################

    #model.fit(X)

    # split our data into training and validation sets (50/50 split)
    # X_train = X[:X.shape[0] // 2]
    # X_validate = X[X.shape[0] // 2:]
    # y_train = y[:y.shape[0] // 2]
    # y_validate = y[y.shape[0] // 2:]
    #
    # print("X y ", len(X_train), len(y_train))



    # save data in json file
    # with open('JSON/hmm_test.json', "w") as fp:
    #     json.dump(data, fp, indent=4)
    return

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
           self.model = hmm.GMMHMM(n_components=self.n_components, algorithm='viterbi',  #n_mix=64,
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


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()
    plt.savefig("Acc E"+EPOCHS.__str__()+ "-BS"+BATCH_SIZE.__str__())
    plt.close()

    return

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)