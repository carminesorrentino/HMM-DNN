import librosa
import json
import noisereduce as nr
from utils import data_utils as du
import numpy as np
import pandas as pd
import librosa.display


DATASET_PATH = "csv/resized_dataset_path_90.csv"
JSON_PATH = "JSON/data_90_HMM_delta.json"
SAMPLES_TO_CONSIDER = du.SAMPLES_TO_CONSIDER #22050
NUM_SPEAKER = 90
AUDIO_FOR_SPEAKER = 90
NUM_AUDIO = NUM_SPEAKER * AUDIO_FOR_SPEAKER

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """
    audio_list = getAudio()

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs_delta1_delta2": [],
        # "Delta1": [],
        # "Delta2": [],
        "files": []
    }
    matrix = np.array([])
    count = 0 #count dei 40 audio per ciascun speaker
    i = 0 #indice label

    for file_path in (audio_list['path'][0:NUM_AUDIO]):


        # load audio file and slice it to ensure length consistency among different files
        signal, sample_rate = librosa.load(file_path)

        # drop audio files with less than pre-decided number of samples
        if len(signal) >= SAMPLES_TO_CONSIDER:
            count = count + 1
            #print("Count ", count)

            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # Rimozione rumore
            reduced_noise = nr.reduce_noise(y=signal, sr=sample_rate, thresh_n_mult_nonstationary=2, stationary=False)
            signal = reduced_noise

            # pre-emphasis
            pre_emphasis = -0.97
            emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[: -1])
            signal = emphasized_signal

            # Framing and windowing
            FRAME_SIZE = 0.025
            FRAME_STRIDE = 0.01
            signal_length = len(emphasized_signal)
            frame_length, frame_step = FRAME_SIZE * sample_rate, FRAME_STRIDE * sample_rate
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))


            # #stft = librosa.stft(signal, n_fft=512, hop_length=hop_length)
            stft = librosa.stft(signal, n_fft=512, hop_length=num_frames)
            # # calculate abs values on complex numbers to get magnitude
            spectrogram = np.abs(stft)
            #
            # # apply logarithm to cast amplitude to Decibels
            log_spectrogram = librosa.amplitude_to_db(spectrogram)
            #
            #
            # # application of mel filterbanks
            mel_spectrogram = librosa.feature.melspectrogram(S=log_spectrogram, sr=sample_rate,
                                                             n_fft=n_fft, hop_length=hop_length, n_mels=20)
            # # apply logarithm to cast amplitude to Decibels
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)


            # extract MFCCs
            # MFCCs = librosa.feature.mfcc(S=lmel_spectrogram, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
            #                              hop_length=hop_length, dct_type=2, lifter=(2*13))
            ##>>>PROBLEMA DI IERI:It looks to me like the problem is that your audio signal is too short.
            ##>>>SOLUZIONI: I'd suggest either shortening the filter, or padding out your signal to a minimum duration before processi
            ##y parameter = audio time series. Multi-channel is supported..
            ##S parameter = log-power Mel spectrogram
            ##n_mfcc = number of MFCCs to return
            ##dct_type parameter = Discrete cosine transform (DCT) type. By default, DCT type-2 is used.
            ##lifter parameter = f lifter>0, apply liftering (cepstral filtering) to the MFCCs.
            ##https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
            ##extract MFCCs
            MFCCs = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sample_rate, n_mfcc=num_mfcc,
                                         n_fft=n_fft, hop_length=hop_length, dct_type=2, lifter=(2 * 13))

            #first derivative
            delta_mfcc = librosa.feature.delta(MFCCs, order=1)
            #second derivative
            delta2_mfcc = librosa.feature.delta(MFCCs, order=2)

            # We'll show each in its own subplot
            # plt.figure(figsize=(12, 6))
            #
            # plt.subplot(3, 1, 1)
            # librosa.display.specshow(MFCCs)
            # plt.ylabel('MFCC')
            # plt.colorbar()
            #
            # plt.subplot(3, 1, 2)
            # librosa.display.specshow(delta_mfcc)
            # plt.ylabel('MFCC-$\Delta$')
            # plt.colorbar()
            #
            # plt.subplot(3, 1, 3)
            # librosa.display.specshow(delta2_mfcc, sr=sample_rate, x_axis='time')
            # plt.ylabel('MFCC-$\Delta^2$')
            # plt.colorbar()
            #
            # plt.tight_layout()
            # plt.show()
            # plt.savefig("mfcc_plot/"+i.__str__()+"-"+count.__str__())


            # Concatena le caratteristiche MFCC, Delta1, Delta2 per ogni audio
            #Es: matrix =[
            #               [MFCCs[0], Delta1[0], Delta2[0]]
            #               [MFCCs[1], Delta1[1], Delta2[1]]
            #            ]
            array0 = np.array([])
            array1 = np.array([])
            for index, mfcc in enumerate (MFCCs.T):
                array = np.concatenate([MFCCs.T[index], delta_mfcc.T[index], delta2_mfcc.T[index]])

                if(index == 0):
                    array0 = array
                    continue

                if(index == 1):
                    array1 = array
                    continue

                if(index == 2):
                    matrix = np.vstack([array0, array1])

                matrix = np.vstack([matrix, array])

                #print("index matrix", index, matrix)

            #data["MFCCs_delta1_delta2"].append(MFCCs.T.tolist()) # memorizza le caratteristiche [MFCCs, Delta1, Delta2]
            data["MFCCs_delta1_delta2"].append(matrix.tolist()) # memorizza le caratteristiche [MFCCs, Delta1, Delta2]
            # data["Delta1"].append(delta_mfcc.T.tolist())
            # data["Delta2"].append(delta2_mfcc.T.tolist())
            data["labels"].append(i)
            data["files"].append(file_path)
            print("{}: {}".format(file_path, i))


            # aggiornamento label
            if count == AUDIO_FOR_SPEAKER:
                count = 0
                i = i + 1


    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


    return


# carica gli audio da analizzare dal file resized_dataset_path.csv
def getAudio():

    df = pd.read_csv(DATASET_PATH)

    labels = df.loc[:, 'labels'].values
    path = df.loc[:, 'path'].values

    data = {
        "labels" : labels,
        "path" : path
    }

    return data


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)