""" Custom Dataset class implementation """

import torch
from PIL import Image
import numpy as np

import torchvision.transforms as transforms

from torch.utils.data import Dataset


# map-style dataset (implements __getitem__() and __len__() protocols)
class TubeDataset(Dataset):
    def __init__(self, **kwargs):
        # self.root = root
        self.num_views = kwargs['num_views']
        # self.plug_label_map = self._get_plug_label_map(root + '/train.csv')
        self.label_encoder = kwargs['class_id_map']
        self.dataframe = kwargs['dataframe']
        self.ids = self._get_unique_ids()
        self.id_label_map = self._get_id_label_map()
        self.data_aug = kwargs['data_aug']
        self.norm = kwargs['norm']
        ##
        self.target_size = kwargs['target_size']
        self.window_size = kwargs['window_size']
        self.window_origin = kwargs['window_origin']
        self.window_offset = kwargs['window_offset']

    def _get_unique_ids(self):  # qui determino gli id
        ids = self.dataframe['id'].values  # all the id
        _, idx = np.unique(ids, return_index=True)
        unique_id = ids[np.sort(idx)]  # this keeps the order of the ids in the dataframe
        return unique_id

    def _get_id_label_map(self):  # qui determino le label associate a ciascun id
        id_group = self.dataframe.groupby(['id'])
        ids = self.ids  # unique id
        id_label_map = {}
        for id in ids:
            id_label_map[id] = id_group.get_group(id)['label'].values[0]
        return id_label_map

    def __len__(self):
        return len(self.ids)

    # def _transform(self, image, split):  # qui data augmentation ...
    def _transform(self, image, lab):
        # indipendentemente da train/val, data aug questi passaggi deve farli
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tr = transform(image)
        mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])
        # print("mean, std : ", mean, std)

        h = self.target_size[0]
        w = self.target_size[1]

        if self.norm == 1:
            # print("Normalization")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)
            ])

        if self.data_aug == 1 and lab == 'lipemico':  # this apply data augmentation on lipemic samples (minority class)
            # print("Data Augmentation on training set")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=2,  # rotation
                                        translate=(0.10, 0.15),
                                        # shift horizontal and vertical
                                        scale=([1 - 0.05, 1 + 0.05]),  # zoom
                                        shear=0.05)])  # shear
            # consider also transforms.Pad ...  fill_mode in tensorflow
            if self.norm == 1:
                # print("Normalization")
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.Normalize(mean, std),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine(degrees=2,  # rotation
                                            translate=(0.10, 0.15),
                                            # shift horizontal and vertical
                                            scale=([1 - 0.05, 1 + 0.05]),  # zoom
                                            shear=0.05)])  # shear

        return transform(image)

    def __getitem__(self, index):
        id_group = self.dataframe.groupby(['id'])
        selected_id = self.ids[index]
        # print("selected id: ", selected_id)

        # this ia a view (a group of 8 images with the same id) # nd.array
        im_group = id_group.get_group(selected_id)['image'].values
        # ora ho i path delle immagini, devo aprirle e concatenarle
        lab = id_group.get_group(selected_id)['label'].values[0]
        # split = id_group.get_group(selected_id)['split'].values[0]
        view = []
        for i in range(len(im_group)):
            im_array = self.read_image_as_array(im_group[i])  # pil or None
            if im_array is None:
                # print("Sostituisco l'immagine corrotta con la seconda immagine del gruppo")
                im_array = self._transform(self.read_image_as_array(im_group[2]), lab)
            else:
                im_array = self._transform(im_array, lab)
            view.append(im_array)

        view = torch.stack(view, 0)  # num_views, c, h, w view ?? un tensor
        y = self.label_encoder[lab]

        return selected_id, view, y

    def read_image_as_array(self, path):
        import random

        try:
            image_array = Image.open(path)
            image_array.load()
        except IOError:
            # print("Error opening file")
            # print("Attempting to re-generate it...")
            image_array = None

        if len(self.window_offset) != 0:
            y = random.randrange(-self.window_offset[0], self.window_offset[0])
            x = random.randrange(-self.window_offset[1], self.window_offset[1])
        else:
            x = 0
            y = 0

        originx = self.window_origin[1] + x
        originy = self.window_origin[0] + y

        if image_array is not None:
            image_array = image_array.convert('RGB')
            # nel caso di dataset aumentato ho delle immagini che sono gi?? croppate ...
            if image_array.height == 960 and image_array.width == 1280:  # dimensioni originali delle immagini
                image_array = np.array(image_array)
                # random rectangle roughly centered on the original window
                image_array = image_array[originy:originy + self.window_size[0], originx:originx + self.window_size[1]]
                image_array = Image.fromarray(image_array, mode='RGB')

        return image_array


def write_csv_split(csv_in, csv_out, seed, val_perc):  # open.csv, train_val_dataset.csv
    import csv
    import pandas as pd

    datasetcsv = pd.read_csv(csv_in, names=['data', 'id', 'image', 'label'])
    id_group = datasetcsv.groupby(['id'])
    ids = datasetcsv['id'].values
    _, idx = np.unique(ids, return_index=True)
    unique_id = ids[np.sort(idx)]  # this keeps the order of the ids in the dataframe

    labels = []  # all the labels based on the all dataset (train and validation)
    for id in unique_id:
        label = id_group.get_group(id)['label'].values[0]
        labels.append(label)
    id_label_map = {}
    for id in ids:
        id_label_map[id] = id_group.get_group(id)['label'].values[0]

    val_size = round((val_perc / 100) * len(unique_id))
    train_size = len(unique_id) - val_size

    train_split, val_split = torch.utils.data.random_split(unique_id, [train_size, val_size],
                                                           generator=torch.Generator().manual_seed(seed))
    ti = np.array(train_split.dataset)[train_split.indices]  # train_split.indices

    # ri etichetto il dataframe
    with open(csv_out, 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        for i in range(len(unique_id)):
            id = unique_id[i]
            # if i in train_split.indices:
            if id in ti:
                # cerco l'id nel dataframe ed etichetto tutto il gruppo delle otto immagini con train
                new_column = ['train'] * 8
            else:
                print("i, id: ", i, id)
                new_column = ['validation'] * 8

            data_group = id_group.get_group(id)
            for k in range(len(data_group)):
                row = data_group.iloc[k]
                line = [row[0], row[1], row[2], row[3], new_column[k]]
                filewriter.writerow(line)


if __name__ == '__main__':
    import pandas as pd
    from utils import get_class_distribution
    import argparse

    parser = argparse.ArgumentParser(description="Create a csv with train/validation split")

    parser.add_argument("--in_csv", dest="in_csv", default=None,
                        help="path to csv containing the dataset images and labels grouped by data and id")

    parser.add_argument("--val_perc", dest="val_perc", default=20, help="% for validation set")

    parser.add_argument("--seed", dest="seed", default=42, help="42 for saved split, a number different from 0 "
                                                                "to create a new train/validation split")
    args = parser.parse_args()

    class_id_map = {'emolitico': 0, 'lipemico': 1}

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    # IMG_HEIGHT = 125
    # IMG_WIDTH = 400

    #######################
    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]  # 1
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    ########################

    datasetcsv = pd.read_csv(args.in_csv, names=['data', 'id', 'image', 'label'])
    csv_out = './train_val_dataset_' + args.seed + '.csv'
    write_csv_split(args.in_csv, csv_out, seed=args.seed, val_perc=int(args.val_perc))
    new_csv = pd.read_csv(csv_out, names=['data', 'id', 'image', 'label', 'split'])

    # 20% di 192 = 38 val size --> 38*8 = 304 val images
    # 192 - 38 = 154 train size --> 154*8 = 1232 train images

    g = new_csv.groupby('split')
    train_group = g.get_group('train')  # dataframe train
    validation_group = g.get_group('validation')

    da_t = TubeDataset(dataframe=train_group,
                       window_origin=window_origin,
                       window_offset=window_offset,
                       window_size=window_size,
                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                       class_id_map=class_id_map,
                       num_views=8, data_aug=0, norm=0)

    cd_t = get_class_distribution(da_t)

    da_v = TubeDataset(dataframe=validation_group,
                       window_origin=window_origin,
                       window_offset=window_offset,
                       window_size=window_size,
                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                       class_id_map=class_id_map,
                       num_views=8, data_aug=0, norm=0)

    cd_v = get_class_distribution(da_v)

    # Ex: python dataset.py --in_csv ./dataset_files/open.csv
