""" Try to use data augmentation on the lipemic samples (minority class), in order to balance the dataset.
Samples created are saved on a new dataset folder. """

# individuo le view di lipemici, ne seleziono un certo numero tale per cui poi ho bilanciato il dataset
# attualmente, con l'aggiunta dei campioni del 27 luglio, abbiamo 96 view 'emolitico' e 27 view 'lipemico'
# --> dovremmo ricreare 69 view di lipemico ...
import os.path

import pandas as pd
import matplotlib.pyplot as plt

# nb: non posso fare data augmentation sulle immagini senza averle prima tagliate ... il crop non tornerebbe dopo :(


def read_image(path):
    from PIL import Image
    import numpy as np
    import random

    #######################
    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]
    ########################

    try:
        image_array = Image.open(path)
        image_array.load()
    except IOError:
        # print("Error opening file")
        # print("Attempting to re-generate it...")
        image_array = None

    if len(window_offset) != 0:
        y = random.randrange(-window_offset[0], window_offset[0])
        x = random.randrange(-window_offset[1], window_offset[1])
    else:
        x = 0
        y = 0

    originx = window_origin[1] + x
    originy = window_origin[0] + y

    if image_array is not None:
        image_array = np.array(image_array)
        image_array = image_array[originy:originy + window_size[0], originx:originx + window_size[1]]
        image_array = Image.fromarray(image_array, mode='RGB')

    return image_array


def augment_image(image_array):
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=2,  # rotation
                                translate=(0.10, 0.15),
                                # shift horizontal and vertical
                                scale=([1 - 0.05, 1 + 0.05]),  # zoom
                                shear=0.05)])  # shear

    return transform(image_array)


if __name__ == '__main__':

    from dataset import write_csv_split
    import argparse

    parser = argparse.ArgumentParser(description="Create augmented lipemic samples")

    parser.add_argument("--augmented_folder", dest="augmented_folder",
                        default="C:/Users/chiar/Desktop/Progetto_Bioimmagini/Campioniparticolari/augmented_samples/",
                        help="path to folder to store the lipemic samples augmented")
    parser.add_argument("--original_csv", dest="original_csv", default="./dataset_files/open_new.csv",
                        help="path to the current (not augmented) dataset csv")
    parser.add_argument("--augmented_csv", dest="augmented_csv", default="./dataset_files/augmented_dataset.csv",
                        help="path to new augmented dataset csv")
    parser.add_argument("--train_val_csv", dest="train_val_csv", default="./dataset_files/train_val_dataset_augmented_",
                        help="path to new train/val split augmented dataset csv (path + start name for the file)")

    parser.add_argument("--seed", dest="seed", default=42, help="seed for random train/validation split")
    parser.add_argument("--val_perc", dest="val_perc", default=20, help="% for validation set")

    args = parser.parse_args()

    if not os.path.exists(args.augmented_folder):
        os.makedirs(args.augmented_folder)

    augmented_path = args.augmented_folder

    datasetcsv = pd.read_csv(args.original_csv, names=['data', 'id', 'image', 'label'])
    label_group = datasetcsv.groupby(['label'])
    lip = label_group.get_group('lipemico')  # campioni lipemici
    id_lip = lip.groupby(['id'])

    letters = ['p', 'q', 'r', 's', 't', 'u', 'v', 'z']  # to change the ids

    dcsv = datasetcsv
    for p in range(3):  # ciclo sul numero di volte in cui devo runnare il ciclo interno per aumentare i campioni lipemici
        # le view lipemico sono 27 --> con 2 iterazioni arrivo a 54 e poi mi mancano 15 view da replicare
        if p == 2:
            en = enumerate(list(id_lip)[:15])
        else:
            en = enumerate(list(id_lip))

        for k, (id, view) in en:  # ciclo sui campioni lipemici da 'aumentare'
            print("id: ", id)
            images = view['image'].values.tolist()  # lista immagini per id
            data = view['data'].values[0]
            lab = view['label'].values[0]
            block = []
            for i in range(len(images)):
                # applico data augmentation ad ogni immagine, la salvo e ricreo un dataframe da aggiungere a datasetcsv
                im = images[i]
                ra = read_image(im)
                if ra is None:
                    new_im = augment_image(read_image(images[2]))
                else:
                    new_im = augment_image(ra)
                print("im name: ", im)
                im_array = new_im.permute(1, 2, 0).numpy()  # h, w, c
                # start_path = im.rpartition('Sample')[0]  # cartella che conterrà l'immagine
                data_fold = im.rpartition('Sample_')[0].split('/')[-2]  # 28-feb-22
                elements = im.rpartition('Sample_')[2].rpartition('Rotation')  # ('a0430', 'Rotation', '_00_Led_00.png')
                id_new = elements[0] + letters[p]
                # filename = start_path + id_new + elements[1] + elements[2]
                if not os.path.exists(augmented_path + data_fold):
                    os.makedirs(augmented_path + data_fold)
                filename = augmented_path + data_fold + '/' + 'Sample_' + id_new + elements[1] + elements[2]

                plt.imsave(filename, im_array)

                row = [data, id_new, filename, lab]
                block.append(row)
                # break
            block = pd.DataFrame(block, columns=datasetcsv.columns)
            dcsv = pd.concat([dcsv, block], ignore_index=True)
            # break

    # salva il nuovo dataframe su csv
    dcsv.to_csv(args.augmented_csv, index=False, header=False)
    # 1536 righe = 984 + (69*8)
    # 192 id = 123 + 69

    # import numpy as np
    # ids = dcsv['id'].values  # all the id
    # _, idx = np.unique(ids, return_index=True)
    # unique_id = ids[np.sort(idx)]

    csv_in = args.augmented_csv  # 'augmented_dataset.csv'
    csv_out = args.train_val_csv + args.seed + '.csv'
    write_csv_split(csv_in, csv_out, seed=int(args.seed), val_perc=int(args.val_perc))
    new_csv = pd.read_csv(csv_out, names=['data', 'id', 'image', 'label', 'split'])


