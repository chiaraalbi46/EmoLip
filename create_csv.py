""" Create csv file for the dataset """

import os
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import math
import shutil


def append_txt(file, lines):
    with open(file, 'a') as f:
        for l in lines:
            f.writelines('\n'.join(l))
            f.write('\n')


def check_data(row_data, data):
    res = 0
    if data == 'null':  # subfolder 29,30-mar-22
        if row_data != '29-mar-22' and row_data != '30-mar-22':
            print("la data associata all'id non corrisponde alla data della cartella: ", data)
            res = 1
    else:
        if row_data != data.strftime('%d-%b-%y'):
            print("la data associata all'id non corrisponde alla data della cartella: ", data)
            res = 1

    return res


# non ho gestito i casi più 'articolati' perchè con i dati a disposizione non ci cadevo comunque
def check_label(row):
    # ['data', 'id', 'label', 'indice lipemico', 'indice emolitico']
    lab = row['label']
    lab = lab.lower()  # lowercase
    if '/' in lab:
        print("ho indecisione sull'etichettamento --> guardo le colonne indice lipemico, indice emolitico")
        if pd.isnull(row['indice lipemico']) and pd.isnull(row['indice emolitico']):
            print("non posso etichettare l'immagine --> non la considero")
            lab = 'null'
        elif row['indice lipemico'] == 'na':
            lab = 'emolitico'
        elif row['indice emolitico'] == 'na':
            lab = 'lipemico'
        elif row['indice emolitico'] > row['indice lipemico']:
            lab = 'emolitico'
        else:
            # row['indice lipemico'] > row['indice emolitico']
            lab = 'lipemico'

    return lab


def create_csv(base_folder, ods1, ods2, save_path, log_path):
    import locale
    locale.setlocale(locale.LC_ALL, 'it_IT')  # questo serve per usare datetime anche se le date sono in italiano

    f = open(log_path, 'w')
    f.write('Log file \n')

    # sorted is important because linux doesn't follow alphabetical order
    subfolders = sorted(os.listdir(base_folder))  # 15 lug, 28 feb etc

    # ods 1 # lista campioni
    ind_cols = [0, 1, 3, 5, 6]  # indici delle colonne di interesse (data, id, label, indice lipemico, indice emolitico)
    df = pd.read_excel(ods1, engine='odf')
    df1 = df.loc[:, df.columns[ind_cols]]
    df1.columns = ['data', 'id', 'label', 'indice lipemico', 'indice emolitico']
    data_col1 = df1['data']
    dates1 = np.unique(data_col1)
    timestamps = ((dates1 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    dates1 = [datetime.utcfromtimestamp(j) for j in timestamps if not math.isnan(j)]  # qui il nan lo perdo

    # ods 2 (15 luglio) # dati campioni particolari
    ind_cols = [0, 2, 3, 4, 5]  # indici delle colonne di interesse (data, id, label, indice lipemico, indice emolitico)
    df = pd.read_excel(ods2, engine='odf')
    df2 = df.loc[:, df.columns[ind_cols]]
    df2.columns = ['data', 'id', 'label', 'indice lipemico', 'indice emolitico']
    data_col2 = df2['data']
    dates2 = np.unique(data_col2)
    timestamps = ((dates2 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's'))
    dates2 = [datetime.utcfromtimestamp(j) for j in timestamps if not math.isnan(j)]

    lines = []

    with open(save_path, 'w') as csvfile:  # ciclo sulle subfolders (date)
        filewriter = csv.writer(csvfile)
        for i in range(len(subfolders)):
            dates = dates1
            dataframe = df1
            group_dates = 0
            dic = {}
            if subfolders[i] == '15-lug-22':
                print("Use dates 2")
                dates = dates2
                dataframe = df2

            if subfolders[i] == '29,30-mar-22':
                group_dates = 1
                data = 'null'
            else:
                data = datetime.strptime(subfolders[i], '%d-%b-%y')  # --> datetime.datetime(2022, 7, 15, 0, 0)

            # campioni lipemici aggiungi 27 luglio 2022
            if subfolders[i] == '27-lug-22':
                lines.append([subfolders[i]])
                # questi campioni non sono presenti negli ods ---> etichetto direttamente con lipemico
                date_path = base_folder + '/' + subfolders[i] + '/'  # .../27-lug-22/
                images = sorted(os.listdir(date_path))
                lab = 'lipemico'
                for im in images:
                    if im.startswith('S'):
                        # parsing
                        idi = im.rpartition('Rotation')[0][7:]  # a0feb
                        im_path = date_path + im
                        print("im path: ", im_path)
                        line = [subfolders[i], idi, im_path, lab]
                        filewriter.writerow(line)
            else:

                # il primo controllo devo farlo sulla data ... se non è contenuta nel csv non ho possibilità di fare il
                # labelling quindi non considero la cartella

                if data in dates or group_dates == 1:
                    print('ok')
                    lines.append([subfolders[i]])

                    date_path = base_folder + '/' + subfolders[i] + '/'  # .../15-lug-22/
                    print("date path: ", date_path)
                    images = sorted(os.listdir(date_path))

                    # check sul numero di campioni per ogni id (devo avere per tutti 8 immagini, sennò non considero l'id)
                    ids = [el.rpartition('Rotation')[0][7:] for el in images if el.startswith('S')]
                    unique, counts = np.unique(ids, return_counts=True)
                    result = np.column_stack((unique, counts))  # id, quanti elementi per id

                    r = np.where(result[:, 1] < str(8))  # individuo gli indici degli id che hanno meno di 8 immagini

                    for k in range(len(r[0])):
                        if int(result[r[0][k], 1]) >= 6:
                            line = [result[r[0][k], 0] + ' has ' + result[r[0][k], 1] + ' images, instead of 8 images']
                            if line not in lines:
                                lines.append(line)
                            na = []
                            # estraggo il blocco di immagini che contiene quell'id
                            [na.append(im) for im in images if result[r[0][k], 0] in im]
                            print("replico almeno un'immagine")
                            # copio la prima (00) e la seconda (01) e le rinomino 06 e 07
                            src_path = date_path + na[1]
                            op = [int(na[l].rpartition('Rotation')[2][2]) for l in range(len(na))]  # 0, 1, 2 etc
                            missing = sorted(set(range(op[0], op[-1]+1)) - set(op))
                            if len(missing) != 0:
                                dst_path = date_path + na[1][:21] + '0' + str(missing[0]) + na[1][23:]
                                line = ['image ' + na[1] + ' has been replicated. The new name is ' +
                                        na[1][:21] + '0' + str(missing[0]) + na[1][23:]]
                            else:
                                dst_path = date_path + na[1][:21] + '07' + na[1][23:]
                                line = ['image ' + na[1] + ' has been replicated. The new name is ' +
                                        na[1][:21] + '07' + na[1][23:]]
                            shutil.copy(src_path, dst_path)
                            print('Copied')

                            lines.append(line)

                            if int(result[r[0][k], 1]) == 6:
                                print("devo replicare due immagini")
                                # copio la prima (00) e la seconda (01) e le rinomino 06 e 07
                                src_path = date_path + na[0]
                                if len(missing) != 0:
                                    dst_path = date_path + na[0][:21] + '0' + str(missing[1]) + na[0][23:]
                                    line = ['image ' + na[0] + ' has been replicated. The new name is ' +
                                            na[0][:21] + '0' + str(missing[1]) + na[0][23:]]
                                else:
                                    dst_path = date_path + na[0][:21] + '06' + na[0][23:]
                                    line = ['image ' + na[0] + ' has been replicated. The new name is ' +
                                            na[0][:21] + '06' + na[0][23:]]
                                shutil.copy(src_path, dst_path)
                                print('Copied')

                                lines.append(line)
                        else:
                            print("Il numero di campioni per l'id è minore di 6/7: ", result[r[0][k], 0], result[r[0][k], 1])
                            line = [result[r[0][k], 0] + ' has ' + result[r[0][k], 1] + ' images < 6/7 images']
                            if line not in lines:
                                lines.append(line)

                    images = sorted(os.listdir(date_path))
                    # ricalcolo result ... devo escludere gli id per cui non ho replicato le immagini
                    ids = [el.rpartition('Rotation')[0][7:] for el in images if el.startswith('S')]
                    unique, counts = np.unique(ids, return_counts=True)
                    result = np.column_stack((unique, counts))  # id, quanti elementi per id
                    for im in images:
                        if im.startswith('S'):
                            # parsing
                            idi = im.rpartition('Rotation')[0][7:]  # a0feb
                            # trovo l'indice corrispondente ad idi in result
                            g = np.where(result[:, 0] == idi)
                            if int(result[g[0][0]][1]) != 8:
                                print("Il numero di campioni per l'id è diverso da 8: ", idi, result[g[0][0]][1])
                                line = [str(idi) + ' has ' + result[g[0][0]][1] + ' images, '
                                                                                  'and they have not been replicated']
                                if line not in lines:
                                    lines.append(line)
                            else:
                                # questo id devo cercarlo nel dataframe e vedere la colonna label per etichettare l'immagine
                                ind = np.where(dataframe['id'] == idi)
                                if len(ind[0]) != 0:
                                    print("id riscontrato: ", idi)
                                    if len(ind[0]) == 1:
                                        inde = ind[0][0]  # indice riga corrispondente all'id riscontrato --> mi serve per la label
                                    else:
                                        # lo stesso id è riscontrato su più righe
                                        # mi interessa solo quello corrispondente alla data che sto considerando
                                        for k in range(len(ind[0])):
                                            d = ind[0][k]
                                            res = check_data(dataframe.loc[d, 'data'].strftime('%d-%b-%y'), data)
                                            if res == 0:
                                                # la data coincide
                                                inde = d

                                    row = dataframe.loc[inde]
                                    if row['id'] in dic.keys():
                                        print("id già etichettato: ", row['id'])
                                        lab = dic[row['id']]
                                    else:
                                        lab = check_label(row)
                                        dic[row['id']] = lab

                                    if lab != 'null':
                                        if lab != 'itterico':
                                            print("etichetto: ", lab)
                                            im_path = date_path + im
                                            print("im path: ", im_path)
                                            line = [row['data'].strftime('%d-%b-%y'), idi, im_path, lab]
                                            filewriter.writerow(line)
                                        else:
                                            line = [str(idi) + ' is itterico']
                                            if line not in lines:
                                                lines.append(line)
                                    else:
                                        line = [str(idi) + ' not labelled']
                                        if line not in lines:
                                            lines.append(line)
                                else:
                                    print("id non riscontrato: ", idi)
                                    line = [str(idi) + ' not found']
                                    if line not in lines:
                                        lines.append(line)

                else:
                    print("la data non è tra quelle del csv --> non considero queste immagini: ", data)
                    line = [data.strftime('%d-%b-%y') + ' not presented in ods files']
                    if line not in lines:
                        lines.append(line)

    append_txt(log_path, lines)


def create_synthesis_file(csv_file, save_excel_file):
    cs = pd.read_csv(csv_file, names=['data', 'id', 'im', 'label'])
    ids = cs['id']
    unique, counts = np.unique(ids, return_counts=True)
    result = np.column_stack((unique, counts))

    indexes = []
    for k in unique:
        indices = cs.index[cs['id'] == k].tolist()
        indexes.append(indices[0])

    labs = cs.loc[indexes, 'label']
    dat = cs.loc[indexes, 'data']
    mann = np.column_stack((dat, result, labs))
    baba = pd.DataFrame(mann, columns=['data', 'id', 'counts', 'label'])
    baba.to_excel(save_excel_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create csv file for the dataset")

    parser.add_argument("--base_folder", dest="base_folder", default=None, help="path to the dataset folder")
    parser.add_argument("--ods1", dest="ods1", default="./ods/lista campioni.ods",
                        help="path to the ods file with the labelling of the samples (except for the ones of 15-lug-2022")
    parser.add_argument("--ods2", dest="ods2", default="./ods/Dati campioni particolari.ods",
                        help="path to the ods file with the labelling of the samples of 15-lug-2022")
    parser.add_argument("--save_path", dest="save_path", default="./dataset_files/open.csv",
                        help="path to the output csv file")
    parser.add_argument("--save_excel_file", dest="save_excel_file", default=None,
                        help="path to a synthesis excel file, to check number of images for id")
    parser.add_argument("--log_path", dest="log_path", default="./dataset_files/log.txt",
                        help="path to the output log file to track the composition of the dataset")

    args = parser.parse_args()

    create_csv(base_folder=args.base_folder, ods1=args.ods1, ods2=args.ods2, save_path=args.save_path,
               log_path=args.log_path)

    if args.save_excel_file is not None:
        create_synthesis_file(csv_file=args.save_path, save_excel_file=args.save_excel_file)

    # 117 id, 8 immagini per id --> 936 immagini in totale
    # 123 id, 8 immagini per id --> 984 immagini in totale (dopo l'aggiunta dei campioni lipemici del 27 luglio 2022)

    # Ex: python create_csv.py --base_folder C:/Users/chiar/Desktop/Progetto Bioimmagini/Campioniparticolari/dataset
    # --save_excel_path ./dataset_files/synthesis.xlsx
