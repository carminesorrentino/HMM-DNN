import csv

import librosa
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


gauth = GoogleAuth()
drive = GoogleDrive(gauth)

#folder = '1ER0IJg0L8rCKSxsNadVm65Wu-ULIXCbi'
folder = '1-3sMC5v-r_-SYMu6qbqRFtgZOKl1-Feq'
#folder = '107RluIsvua-fI1wpuNpzGCrNWNLpEv1E'

DATASET_PATH = "csv/resized_dataset_path.csv"


def main():

    # inizializzazione file output.csv
    with open('csv/resized_drive_anydesk.csv', 'w', encoding="UTF-8", newline="\n") as csvfile:
        writer = csv.writer(csvfile)
        header = ['labels', 'path']
        writer.writerow(header)
        csvfile.close()

    labels = getAudio()

    data = {
        "labels" : [],
        "path" : []
    }

    speaker_list = []
    audio_path = []
    print(folder)
    # Download files
    file_list = drive.ListFile({'q': f"'{folder}' in parents"}).GetList()
    print("*FILE LIST ", file_list)

    for index, file in enumerate(file_list): #file = directory di ciascun speaker
        print(index + 1, 'file downloaded : ', file['title'], file.get('id'))

        if labels.__contains__(file['title']):
            print("********* OK ", file['title'])

            labels = file['title']

            count = 0 #conta gli audio di ciascuno speaker (40 audio)

            video_list = drive.ListFile({'q': f"'{file.get('id')}' in parents"}).GetList()

            for i, video in enumerate(video_list):  # video
                print(i + 1, " Video ", video.get('id'))

                audio_list = drive.ListFile({'q': f"'{video.get('id')}' in parents"}).GetList()

                if (count == 39):
                    break

                for j, audio in enumerate(audio_list): # Audio
                    print(j + 1, "Audio ", audio.get('id'), audio.get('title'), audio.get('id')+"/"+audio.get('title'))
                    count = count + 1
                    #file_path = "https://drive.google.com/drive/folders/"+audio.get('id')+"/"+audio.get('title')


                    file_path = "drive.google.com/file/d/1d2cnuRfIlaRUFSrBHN80SG0T0nNUKjKb"
                    print("file path da caricare ", file_path)

                    #audio = audio.GetContentFile(audio.get('title'))

                    # load audio file and slice it to ensure length consistency among different files
                    signal, sample_rate = librosa.load(audio)

                    print("AUDIO CARICATO ", file_path, sample_rate)

                    return

                    #Scrive nel csv la coppia [labels, path]
                    with open('csv/resized_drive_anydesk.csv', 'a', encoding="UTF-8", newline="\n") as csvfile:
                        writer = csv.writer(csvfile)

                        tup = [labels, audio.get('id') + "/" + audio.get('title')]

                        print("TUPLA {}".format(tup))

                        writer.writerow(tup)

                    if (count == 39):
                        break




        # speaker_list.append(video_list)
        # print("Video list ", video_list)


    # file.GetContentFile(file['title'])
        # signal, sample_rate = librosa.load(file['title'])
        # print("audio ", file['title'], sample_rate)




    return


# carica gli audio da analizzare dal file resized_dataset_path.csv
def getAudio():

    df = pd.read_csv(DATASET_PATH)

    labels = df.loc[:, 'labels'].values

    data = {
        "labels" : labels,
    }

    labels = list(set(data['labels']))

    return labels

if __name__ == "__main__":
    main()