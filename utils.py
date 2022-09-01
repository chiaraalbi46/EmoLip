""" Utils Function """

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_view(plug_name, dataset):
    ind = dataset.ids.tolist().index(plug_name)

    idi, view, lab = dataset.__getitem__(ind)

    plug_images = view  # .permute(0, 2, 3, 1)
    id_group = datasetcsv.groupby(['id'])
    plug_fnames = id_group.get_group(idi)['image'].values
    plug_fnames = [el.split('/')[-1] for el in plug_fnames]

    # Visualize Images
    plt.figure(figsize=(18, 8))
    for i in range(len(plug_images)):
        plt.subplot(2, 4, i + 1)
        # print("plug shape: ", plug_images[i].shape)
        if isinstance(plug_images[i], np.ndarray):
            plt.imshow(plug_images[i])
        else:
            plt.imshow(plug_images[i].permute(1, 2, 0))
        plt.title(os.path.basename(plug_fnames[i]))
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f'ID VIEW: {plug_name}', size=20)
    plt.show()


def compute_sample_weight(class_weights, y):
    assert isinstance(class_weights, dict)
    result = np.array([class_weights[i] for i in y])
    return result


def get_class_distribution(dataset_obj):
    count_dict = {0: 0, 1: 0}  # 0 emolitico, 1 lipemico

    for element in dataset_obj:
        # nb: id, view, label
        y_lab = element[2]  # 0, 1
        count_dict[y_lab] += 1

    return count_dict


if __name__ == '__main__':
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    #######################
    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]  # 1
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    ########################

    from dataset import TubeDataset
    import pandas as pd

    datasetcsv = pd.read_csv('train_val_dataset_augmented_42.csv', names=['data', 'id', 'image', 'label', 'split'])
    class_id_map = {'emolitico': 0, 'lipemico': 1}

    # choose data_aug, norm

    da = TubeDataset(dataframe=datasetcsv,
                     window_origin=window_origin,
                     window_offset=window_offset,
                     window_size=window_size,
                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                     class_id_map=class_id_map,
                     num_views=8, data_aug=1, norm=0)

    plug_name = 'a0fed'  # id view
    show_view(plug_name, da)


    # for a single image
    # tens = view[3, :, :, :]  # c, h, w
    # # convert this image to numpy array
    # tens = np.array(tens)
    #
    # # transpose from shape of (3,,) to shape of (,,3)
    # tens = tens.transpose(1, 2, 0)

    # # display the normalized image
    # plt.imshow(tens)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
