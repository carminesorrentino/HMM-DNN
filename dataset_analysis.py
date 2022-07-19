import librosa, librosa.display
import os
import wave
import contextlib
import pandas as pd
import csv


def main():

    AUDIO_PATH = "D:/Università/VoxCeleb/wav_dev"
    #AUDIO_PATH = "C:/Users/alfre/Desktop/prova_audio/"

    audio_list = getAllAudio(AUDIO_PATH)

    for index, audio in enumerate(audio_list):
        print('>Audio {index}: {audio}'.format(index=index, audio=audio))

    return

#Legge tutti gli audio e restituisce una lista dei path dei file.wav
def getAllAudio(speaker_path):

    data = {
        "labels" : [],
        "path" : []
    }


    #lista di tutti gli audio del training set
    audio_list = []

    #info relative agli audio di ciascuno speaker
    list_speaker_naudio = []

    N_TRAINING_SET_SPEAKER = 1211 #numero di speaker nel training set
    total_max = 0
    total_min = 99999
    total_average = 0
    total_count = 0
    total_audio_duration = 0 #durata totale di tutti gli audio

    #Itera per ogni speaker
    for speaker in os.listdir(speaker_path):
        print(speaker)
        #Controllo per by-passare la directory ".DS_Store"
        if(speaker=='.DS_Store'):
            continue

        #Recupera il path assoluto relativo alla directory di ciascuno speaker (id10001, id10002, ...)
        speaker_id_path = os.path.join(speaker_path, speaker)

        #Accede ad ogni directory relativa ad uno speaker
        if os.path.isdir(speaker_id_path):

            count = 0  # count = conteggio del numero di audio per ogni speaker
            average = 0 # average = calcola la durata media degli audio associati a ciascuno speaker
            max = 0 # max = durata maggiore di un audio relativo allo speaker esaminato
            min = 99999 # min = durata più breve di un audio relativo allo speaker esaminato
            total_duration = 0 # total_duration = calcola la durata totale degli audio relativi ad uno speaker

            # Accede ad ogni video di uno speaker (1zcIwhmdeo4, 7gWzIy6yIIk,...)
            for video in os.listdir(speaker_id_path):

                # Recupera il path assoluto relativo alla directory di ciascuno speaker/video (id00001/1zcIwhmdeo4, id00001/7gWzIy6yIIk, ...)
                video_id_path = os.path.join(speaker_id_path, video)

                # Per ogni directory che rappresenta un video...
                if os.path.isdir(video_id_path):

                    # Accede ad ogni audio di uno speaker (00001.wav, 00002.wav,...)
                    for audio in os.listdir(video_id_path):

                        ## Recupera il path assoluto relativo alla directory di ciascuno speaker/video/audio (id00001/1zcIwhmdeo4/id00001.wav, id00001/1zcIwhmdeo4/id00002.wav, ...)
                        audio_id_path = os.path.join(video_id_path, audio)

                        # Legge ciascuno degli audio presenti nella directory id_speaker/video/...
                        if os.path.isfile(audio_id_path):

                            # Inserisce l'audio all'interno della lista audio_list
                            audio_list.append(audio_id_path)

                            count = count + 1 # Incrementa il numero di audio rilevati per lo speaker esaminato

                            # Apre ciascun file audio al fine di calcolarne la durata
                            with contextlib.closing(wave.open(audio_id_path, 'r')) as f:
                                frames = f.getnframes()
                                rate = f.getframerate()
                                duration = frames / float(rate) # Calcola la durata di ciascun audio

                                # Aggiorna la durata max e min relativa a ciascuno speaker (info presenti in training_set_speaker_analysis.csv)
                                if (duration > max):
                                    max = duration
                                if (duration < min):
                                    min = duration

                                # Aggiorna la durata max e min relativa a tutti gli speaker (info presenti in all_training_set_audio_stats.csv)
                                if (duration > total_max):
                                    total_max = duration
                                if (duration < total_min):
                                    total_min = duration

                                # Aggiorna la durata complessiva di tutti gli audio di un singolo speaker (info presente in training_set_speaker_analysis.csv)
                                total_duration = total_duration + duration

                                total_count = total_count + 1 # Incrementa il numero di audio complessivi del training set (info presente in all_training_set_audio_stats.csv)
                                total_audio_duration = total_audio_duration + duration # Aggiorna la durata complessiva di tutti gli audio di tutti gli speaker (info presente in all_training_set_audio_stats.csv)


        average = total_duration/count  # Calcola la durata media degli audio associati a ciascuno speaker
        total_average = total_audio_duration / total_count # Calcola la durata media relativa a tutti gli audio di tutti gli speaker

        # Lista della classe SpeakerNAudio che memorizza delle tuple da scrivere nel file set_speaker_analysis.csv (contiene una tupla per ogni speaker)
        list_speaker_naudio.append(speakerAudios(speaker, count, average, max, min, total_duration))


    # print('Durata massima tra tutti gli audio {total_max}'.format(total_max=total_max))
    # print('Durata minima tra tutti gli audio {total_min}'.format(total_min=total_min))
    # print('Durata media tra tutti gli audio {total_average}'.format(total_average=total_average))


    # Scriviamo il file set_speaker_analysis.csv contenente le informazioni su ciascuno speaker
    # speaker = id dello speaker considerato
    # num_audio = conteggio del numero di audio per ogni speaker
    # media = calcola la durata media degli audio associati a ciascuno speaker
    # max = durata maggiore di un audio relativo allo speaker esaminato
    # min = durata più breve di un audio relativo allo speaker esaminato
    # total_duration = calcola la durata totale degli audio relativi ad uno speaker

    #Inizializza file contenente i path del dataset da analizzare
    with open('csv/dataset_path.csv', 'w') as csvfile_dataset:
        writer_dataset = csv.writer(csvfile_dataset)
        writer_dataset.writerow(['id_speaker','path'])

    # Scrittura file csv stats speaker
    with open('csv/training_set_speaker_analysis.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['speaker', '#audio', 'durata_media', 'durata_max', 'durata_min','totale_durata_audio_x_speaker'])


        for item in list_speaker_naudio:
            writer.writerow([item.speaker, item.num_audio, round(item.media, 2), round(item.max, 2), round(item.min, 2),
                             round(item.total_duration, 2)])

            with open('csv/dataset_path.csv', 'a', encoding="UTF-8") as csvfile_dataset:
                writer_dataset = csv.writer(csvfile_dataset)
                # writer_dataset.writerow(['path'])

                if(item.num_audio > 97):
                    # Recupera i path di tutti gli audio di ciascun speaker con un numero di audio > 97
                    # Recupera il path assoluto relativo alla directory di ciascuno speaker (id10001, id10002, ...)
                    speaker_id_path = os.path.join(speaker_path, item.speaker)

                    data["labels"].append(item.speaker)  # memorizza id speaker come label

                    # Accede ad ogni video di uno speaker (1zcIwhmdeo4, 7gWzIy6yIIk,...)
                    for video in os.listdir(speaker_id_path):

                        # Recupera il path assoluto relativo alla directory di ciascuno speaker/video (id00001/1zcIwhmdeo4, id00001/7gWzIy6yIIk, ...)
                        video_id_path = os.path.join(speaker_id_path, video)

                        # Per ogni directory che rappresenta un video...
                        if os.path.isdir(video_id_path):

                            # Accede ad ogni audio di uno speaker (00001.wav, 00002.wav,...)
                            for audio in os.listdir(video_id_path):
                                ## Recupera il path assoluto relativo alla directory di ciascuno speaker/video/audio (id00001/1zcIwhmdeo4/id00001.wav, id00001/1zcIwhmdeo4/id00002.wav, ...)
                                audio_id_path = os.path.join(video_id_path, audio)

                                data["path"].append(audio_id_path) # memorizza audio_id_path come path

                                writer_dataset.writerow([item.speaker, audio_id_path]) #scrive nel csv gli speaker da analizzare


    # Scriviamo il file all_training_set_audio_stats.csv contenente le informazioni complessive circa il training set
    # total_audio_duration = durata complessiva degli audio presenti nel training set
    # total_count = numero totale di audio presenti nel training set
    # total_max = durata dell'audio più lungo presente nel training set
    # total_min = durata dell'audio più breve presente nel training set
    # total_average = durata media degli audio presenti nel training set
    # durata media audio per speaker = total_audio_duration/N_TRAINING_SET_SPEAKER = durata media degli audio relativi ad ogni speaker del training set

    with open('csv/all_training_set_audio_stats.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['durata_complessiva_audio','#totale_audio_training_set','durata_max', 'durata_min', 'durata_media_singolo_audio','durata_media_audio_x_speaker'])
        writer.writerow([round(total_audio_duration,2),total_count ,round(total_max,2), round(total_min,2), round(total_average,2), round(total_audio_duration/N_TRAINING_SET_SPEAKER)])

    return audio_list


class speakerAudios:
    def __init__(self, speaker, num_audio, media, max, min, total_duration):
        self.speaker = speaker
        self.num_audio = num_audio
        self.media = media
        self.max = max
        self.min = min
        self.total_duration = total_duration

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
