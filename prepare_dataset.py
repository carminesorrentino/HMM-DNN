import librosa
import os
import json
import noisereduce as nr
from hmmlearn import hmm

#DATASET_PATH = "dataset"
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "C:/Users/alfre/Desktop/VoxCeleb Demo 10"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050
NUM_SPEAKER = 90
NUM_AUDIO = NUM_SPEAKER * 90
SIZE_SPECTROGRAM = 226 #[226 = log_mel_spectrogram ; 44 = log_spectrogram]

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    # """Extracts MFCCs from music dataset and saves them into a json file.
    # :param dataset_path (str): Path to dataset
    # :param json_path (str): Path to json file used to save MFCCs
    # :param num_mfcc (int): Number of coefficients to extract
    # :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    # :param hop_length (int): Sliding window for FFT. Measured in # of samples
    # :return:
    # """
    # audio_list = getAudio()
    #
    # # dictionary where we'll store mapping, labels, MFCCs and filenames
    # data = {
    #     "mapping": [],
    #     "labels": [],
    #     "MFCCs": [],
    #     "Delta1": [],
    #     "Delta2": [],
    #     "files": []
    # }
    # matrix = np.array([])
    # # loop through all sub-dirs
    # # for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    # #
    # #     # ensure we're at sub-folder level
    # #     if dirpath is not dataset_path:
    # #
    # #         # save label (i.e., sub-folder name) in the mapping
    # #         label = dirpath.split("/")[-1]
    # #         data["mapping"].append(label)
    # #         print("\nProcessing: '{}'".format(label))
    # count = 0
    # #         # process all audio files in sub-dir and store MFCCs
    # #         for f in filenames:
    # i = 0 #indice label
    # for index, file_path in enumerate(audio_list['path'][0:120]):
    #     #file_path = os.path.join(dirpath, f)
    #     print("Audio*", audio_list['path'])
    #     # load audio file and slice it to ensure length consistency among different files
    #     signal, sample_rate = librosa.load(file_path)
    #
    #     #####################
    #     #Rimozione rumore
    #     # reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate, thresh_n_mult_nonstationary=2, stationary=False)
    #     # signal = reduced_noise
    #
    #     #Rimozione silenzio
    #     # yt, count = librosa.effects.trim(signal, top_db=10, frame_length=256, hop_length=64)
    #     #
    #     # signal = yt
    #
    #    #  stft = librosa.stft(signal, n_fft=512, hop_length=hop_length)
    #    # # print('stft: {}'.format(stft))
    #    #
    #    #  # calculate abs values on complex numbers to get magnitude
    #    #  spectrogram = np.abs(stft)
    #    #  #print('stft.abs: {}'.format(spectrogram))
    #
    #
    #     # # apply logarithm to cast amplitude to Decibels
    #     # log_spectrogram = librosa.amplitude_to_db(spectrogram)
    #     #
    #     # # #Applicazione mel filters
    #     # mel_spectrogram = librosa.feature.melspectrogram(y=log_spectrogram, sr=sample_rate, n_fft=2048,
    #     #                                                  hop_length=512, n_mels=20)
    #
    #     # signal = spectrogram
    #     #####################
    #
    #
    #     # drop audio files with less than pre-decided number of samples
    #     if len(signal) >= SAMPLES_TO_CONSIDER:
    #         count = count + 1
    #         print("Count ", count)
    #         # ensure consistency of the length of the signal
    #         signal = signal[:SAMPLES_TO_CONSIDER]
    #
    #         ######## Preprocessing before compute MFCCs
    #         # Rimozione silenzio
    #         # yt, count = librosa.effects.trim(signal, top_db=10, frame_length=256, hop_length=64)
    #         #
    #         # signal = yt
    #         # Rimozione rumore
    #         reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate, thresh_n_mult_nonstationary=2, stationary=False)
    #         signal = reduced_noise
    #
    #         # pre-emphasis
    #         pre_emphasis = -0.97
    #         emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[: -1])
    #         signal = emphasized_signal
    #
    #         # Framing
    #         FRAME_SIZE = 0.025
    #         FRAME_STRIDE = 0.01
    #         signal_length = len(emphasized_signal)
    #         frame_length, frame_step = FRAME_SIZE * sample_rate, FRAME_STRIDE * sample_rate
    #         frame_length = int(round(frame_length))
    #         frame_step = int(round(frame_step))
    #         num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    #
    #         #Velardo preprocessing
    #         # https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/11-%20Preprocessing%20audio%20data%20for%20deep%20learning/code/audio_prep.py
    #         # hop_length = 512  # in num. of samples
    #         # n_fft = 2048  # window in num. of samples
    #
    #         # calculate duration hop length and window in seconds
    #         # hop_length_duration = float(hop_length) / sample_rate
    #         # n_fft_duration = float(n_fft) / sample_rate
    #
    #         #stft = librosa.stft(signal, n_fft=512, hop_length=hop_length)
    #         stft = librosa.stft(signal, n_fft=512, hop_length=hop_length)
    #
    #         # calculate abs values on complex numbers to get magnitude
    #         spectrogram = np.abs(stft)
    #
    #         # apply logarithm to cast amplitude to Decibels
    #         log_spectrogram = librosa.amplitude_to_db(spectrogram)
    #
    #
    #         # Applicazione mel filters
    #         mel_spectrogram = librosa.feature.melspectrogram(S=log_spectrogram, sr=sample_rate, n_fft=n_fft,
    #                                                          hop_length=hop_length, n_mels=20)
    #
    #
    #         log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    #
    #         ########
    #
    #         # extract MFCCs
    #         # MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
    #         #                              hop_length=hop_length, dct_type=2, lifter=(2*13))
    #
    #         MFCCs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sample_rate, n_mfcc=num_mfcc,
    #                                      n_fft=n_fft, hop_length=hop_length, dct_type=2, lifter=(2 * 13))
    #         #mfcc = librosa.feature.mfcc(S=log_spectrogram, n_mfcc=N_MFCCs, dct_type=2, lifter=(2 * N_MFCCs))
    #         delta_mfcc = librosa.feature.delta(MFCCs, order=1)
    #         delta2_mfcc = librosa.feature.delta(MFCCs, order=2)
    #
    #         # Concatena le caratteristiche MFCC, Delta1, Delta2 per ogni audio
    #         #Es: matrix =[
    #         #               [MFCCs[0], Delta1[0], Delta2[0]]
    #         #               [MFCCs[1], Delta1[1], Delta2[1]]
    #         #            ]
    #         array0 = np.array([])
    #         array1 = np.array([])
    #         for index, mfcc in enumerate (MFCCs.T):
    #             array = np.concatenate([MFCCs.T[index], delta_mfcc.T[index], delta2_mfcc.T[index]])
    #
    #             if(index == 0):
    #                 array0 = array
    #                 continue
    #
    #             if(index == 1):
    #                 array1 = array
    #                 continue
    #
    #             if(index == 2):
    #                 matrix = np.vstack([array0, array1])
    #
    #             matrix = np.vstack([matrix, array])
    #
    #             print("index matrix", index, matrix)
    #
    #
    #         #data["MFCCs"].append(MFCCs.T.tolist())
    #         data["MFCCs"].append(matrix.tolist()) # memorizza le caratteristiche [MFCCs, Delta1, Delta2]
    #         data["Delta1"].append(delta_mfcc.T.tolist())
    #         data["Delta2"].append(delta2_mfcc.T.tolist())
    #         data["labels"].append(i)
    #         data["files"].append(file_path)
    #         print("{}: {}".format(file_path, i))
    #         #print("MFCCs ", MFCCs.T.tolist())
    #
    #         # aggiornamento label
    #         if count == 40:
    #             count = 0
    #             i = i + 1
    #
    #
    # ##########------------PCA-----------########################
    # data['MFCCs'] = np.array(data['MFCCs'])
    # print("Shape MFCCs", data['MFCCs'].shape)
    # nsamples, nx, ny = data['MFCCs'].shape
    # data['MFCCs'] = data['MFCCs'].reshape((nsamples * nx, ny))
    #
    # matrix = data['MFCCs']
    # scalare = StandardScaler()
    # matrix = pd.DataFrame(scalare.fit_transform(matrix))
    # print("*** ", matrix)
    # pca = sklearn.decomposition.PCA().fit(matrix)
    # # print('****pca', pca.explained_variance_ratio_)
    # plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    # plt.plot(pca.explained_variance_ratio_, 'o-')
    # plt.xlabel('number of components', )
    # plt.ylabel('cumulative explained variance')
    # plt.show()
    #
    # ##Riduzione della dimensionalità lineare mediante Singular Value Decomposition dei dati per proiettarli in uno spazio dimensionale inferiore.
    # # I dati di input vengono centrati ma non ridimensionati (questo spiega l'operazione di fit_transform precedente).
    # N_COMPONENTS = 10
    # pca = sklearn.decomposition.PCA(n_components=N_COMPONENTS)
    # extraction_pca = pd.DataFrame(pca.fit_transform(matrix))
    # print("****** Extraction_pca ", extraction_pca)
    #
    # #data["MFCCs"] = extraction_pca
    # data["MFCCs"] = extraction_pca.to_numpy()
    # data["MFCCs"] = data["MFCCs"].reshape(NUM_AUDIO, 44, N_COMPONENTS)
    # print("** SHAPE ", data['MFCCs'].shape)
    # data["MFCCs"] = data["MFCCs"].tolist()
    # ######################## HMM ##############################
    #
    # #print("ARRAY INTERNO", len(data['MFCCs'][0]))

    data = load_data(JSON_PATH)

    hmm_models = []
    array_viterbi = []
    start = 0
    end = 40
    #### Training
    for speaker in range(0, NUM_SPEAKER):
        sub_audio = data["MFCCs"][start:end]  # recupera gli audio relativi a ciascun speaker (40 audio per ogni speaker)
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


    print("Shape before ", len(data['MFCCs'][0]), len(data['MFCCs'][0][0]))

    big_array = []
    for arr in array_viterbi:
        big_array = np.concatenate([big_array, arr])
        print("********* BIG ARRAY LEN" , len(big_array))

    print("len big_array", len(big_array))

    index = 0  # indice per l'audio (array esterno)
    count = 0 #indice per gli array di ciascun audio (array interno)
    for i in range (0, len(big_array)): # per ciascun array presente in array_viterbi
        # count = 0 #indice per gli array di ciascun audio (array interno)
        # print(array_viterbi[i])
        # for j, el in enumerate (array_viterbi[i]):
            #print("i, j, count , index ",i, j, count, index)
        if(count < SIZE_SPECTROGRAM):
            data['MFCCs'][index][count].append(big_array[i])
            #data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
            count = count + 1
        else:
            index = index + 1
            count = 0
            data['MFCCs'][index][count].append(big_array[i])
            #data['MFCCs'][index][count] = np.append(data['MFCCs'][index][count], big_array[i])
            count = count + 1
        try:
            print("MFCCs[{}][{}] = {} - len = {}".format(index, count, data['MFCCs'][index][count], len(data['MFCCs'][index][count])))
        except:
            continue
    print("Shape after ", len(data['MFCCs'][0]), len(data['MFCCs'][0][0] ))
    ######################## HMM
    ##############################







    # save data in json file
    with open('JSON/hmm.json', "w") as fp:
        json.dump(data, fp, indent=4)

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
    preprocess_dataset(DATASET_PATH, JSON_PATH)