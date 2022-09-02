import time
from utils import compute_sample_weight


def train_model(model, dataloaders, criterion, optimizer, num_epochs, experiment, save_weights_path, bce): # class_weights
    # import matplotlib.pyplot as plt
    # import seaborn as sn
    import sklearn.metrics

    since = time.time()

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = []
            running_corrects = []
            all_preds = []
            all_labels = []
            # Iterate over data.
            for id, inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    if bce == 1:
                        # quando voglio usare bce loss e l'output del classifier è (b, 1)
                        labels = labels.unsqueeze(1).float()  # [10] int64 --> [10, 1] float32

                    loss = criterion(outputs, labels)

                    # Get model predictions
                    if bce == 1:
                        # quando voglio usare bce loss e l'output del classifier è (b, 1)
                        preds = outputs > 0.5
                        acc = (preds == labels).float().mean()
                    else:
                        preds = outputs.argmax(dim=-1)
                        acc = (preds == labels).float().mean()
                        # it is the same of (outputs == labels).sum().item() / labels.size(0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss.append(loss.item())
                running_corrects.append(acc.item())

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

            epoch_loss = sum(running_loss) / len(running_loss)
            epoch_acc = sum(running_corrects) / len(running_corrects)

            if bce == 1:
                all_labels = torch.cat(all_labels, 0).squeeze(1)
                all_preds = torch.cat(all_preds, 0).squeeze(1)
            else:
                all_labels = torch.cat(all_labels, 0)
                all_preds = torch.cat(all_preds, 0)

            epoch_f1_score = sklearn.metrics.f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
                                                      zero_division=0)
            epoch_balanced_acc_score = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(),
                                                                               all_preds.cpu().numpy())

            # epoch_weighted_acc = sklearn.metrics.accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
            #                                                     sample_weight=compute_sample_weight(class_weights,
            #                                                                                         all_labels.cpu().numpy()))

            # log metrics on comet ml
            experiment.log_metric(phase + '_epoch_loss', epoch_loss, step=epoch)
            experiment.log_metric(phase + '_epoch_acc', epoch_acc, step=epoch)
            # experiment.log_metric(phase + '_weighted_epoch_acc', epoch_weighted_acc, step=epoch)
            experiment.log_metric(phase + '_epoch_f1_score', epoch_f1_score, step=epoch)
            experiment.log_metric(phase + '_epoch_balanced_acc', epoch_balanced_acc_score, step=epoch)

            # long() --> torch.int64, int() --> torch.int32
            experiment.log_confusion_matrix(title=phase + '_confusion_matrix_' + str(epoch),
                                            y_true=all_labels.cpu().long(), y_predicted=all_preds.cpu().long(),
                                            labels=['emolitico', 'lipemico'], step=epoch,
                                            file_name=phase + '_confusion_matrix_' + str(epoch) + '.json')

            print('{} Loss: {:.4f} - Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
            # print('{} Loss: {:.4f} - Acc: {:.4f} - Weighted Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
            #                                                                     epoch_weighted_acc))

            print(sklearn.metrics.classification_report(all_labels.cpu(), all_preds.cpu(),
                                                        target_names=['emolitico', 'lipemico']))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # salva i pesi dell'epoca 'migliore'...

        # Save weights
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_weights_path + '/weights_' + str(epoch + 1) + '.pth')

        print()

    torch.save(model.state_dict(), save_weights_path + '/final.pth')

    experiment.end()
    print("End training loop")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


if __name__ == '__main__':
    from comet_ml import Experiment
    from MVCNN import MVCNN, MVCNN_small
    import torch
    import torch.optim as optim
    import os
    import pandas as pd
    import numpy as np
    import sklearn
    # from sklearn.model_selection import StratifiedShuffleSplit

    from torch.utils.data import DataLoader
    # from torch.utils.data.sampler import SubsetRandomSampler
    from dataset import TubeDataset, write_csv_split
    import argparse
    from utils import get_class_distribution

    parser = argparse.ArgumentParser(description="Train MVCNN")

    parser.add_argument("--epochs", dest="epochs", default=2, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=10, help="batch size")
    parser.add_argument("--lr", dest="lr", default=0.0005, help="learning rate train")  # 0.00005 for fine tuning

    parser.add_argument("--num_classes", dest="num_classes", default=1, help="number of classes of the dataset")

    parser.add_argument("--device", dest="device", default='0', help="choose GPU")
    parser.add_argument("--name_proj", dest="name_proj", default='Bioimmagini', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")

    parser.add_argument("--weights_path", dest="weights_path", default='./weights',
                        help="path to the folder where storing the model weights")
    parser.add_argument("--dataset_csv", dest="dataset_csv", default='./train_val_dataset.csv',  # open.csv
                        help="path to csv containing the dataset images and labels grouped by data and id")

    parser.add_argument("--fine_tune", dest="fine_tune", default=0, help="1 for fine tuning, 0 otherwise")
    parser.add_argument("--bce", dest="bce", default=1, help="1 for bce with logits loss, 0 for crossentropy")
    parser.add_argument("--loss_weights", dest="loss_weights", default=0,
                        help="1 to weight the loss function, 0 otherwise")

    parser.add_argument("--seed", dest="seed", default=42, help="42 for saved split, a number different from 0 "
                                                                "to create a new train/validation split")
    parser.add_argument("--val_perc", dest="val_perc", default=20, help="% for validation set")

    parser.add_argument("--data_aug", dest="data_aug", default=0,
                        help="1 for data augmentation (on train), 0 otherwise")
    parser.add_argument("--norm", dest="norm", default=0, help="1 to apply normalization on images, 0 otherwise")
    parser.add_argument("--small_net", dest="small_net", default=0, help="1 to use MVCNN_small, 0 otherwise")

    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # net
    if int(args.small_net) == 1:
        model = MVCNN_small(num_classes=int(args.num_classes))
    else:
        model = MVCNN(num_classes=int(args.num_classes))

    if int(args.fine_tune) == 0:
        # freeze the weights in the feature extraction block of the network (resnet base)
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        print("Fine tuning")
        # unfreeze the weights ... (meglio non tutti tutti)
        # for param in model.parameters():
        #     param.requires_grad = True
        ct = 0
        for child in model.features.children():
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1

    model.to(device)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)

    # dataset and dataloaders
    if int(args.seed) == 42:
        print("Load the saved split, using file: ", args.dataset_csv)
        datasetcsv = pd.read_csv(args.dataset_csv, names=['data', 'id', 'image', 'label', 'split'])
    else:
        print("Create a new csv file for split")
        csv_out = './train_val_dataset_' + str(args.seed) + '.csv'
        write_csv_split(args.dataset_csv, csv_out, seed=int(args.seed), val_perc=int(args.val_perc))
        datasetcsv = pd.read_csv(csv_out, names=['data', 'id', 'image', 'label', 'split'])

    g = datasetcsv.groupby('split')
    train_group = g.get_group('train')  # dataframe train
    validation_group = g.get_group('validation')

    class_id_map = {'emolitico': 0, 'lipemico': 1}

    #######################
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]  # 1
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    ########################

    train_dataset = TubeDataset(dataframe=train_group,
                                window_origin=window_origin,
                                window_offset=window_offset,
                                window_size=window_size,
                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                class_id_map=class_id_map,
                                num_views=8, data_aug=int(args.data_aug), norm=int(args.norm))

    validation_dataset = TubeDataset(dataframe=validation_group,
                                     window_origin=window_origin,
                                     window_offset=window_offset,
                                     window_size=window_size,
                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                     class_id_map=class_id_map,
                                     num_views=8, data_aug=0, norm=int(args.norm))  # no data aug on validation

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    data_loaders = {'train': train_loader, 'val': val_loader}

    # compute class weights ... try using the utils of sklearn
    # wj=n_samples / (n_classes * n_samplesj) j is the class
    id_group = datasetcsv.groupby(['id'])
    ids = datasetcsv['id'].values
    _, idx = np.unique(ids, return_index=True)
    unique_id = ids[np.sort(idx)]
    labels = []  # all the labels based on the all dataset (train and validation)
    for id in unique_id:
        label = id_group.get_group(id)['label'].values[0]
        labels.append(label)

    if int(args.bce) == 1:
        print("Using BCEWithLogitsLoss")

        if int(args.loss_weights) == 1:
            print("Weight the loss")
            cd = get_class_distribution(train_dataset)
            pos_weight = cd[0] / cd[1]  # #negativi/#positivi (#0/#1)  # ho usato la distribuzione nel training ...
            pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)

            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    else:
        print("Using CrossEntropyLoss")
        if int(args.loss_weights) == 1:
            print("Weight the loss")
            class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                            classes=np.unique(labels),
                                                                            y=labels)
            weight = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.lr))

    # Comet ml integration
    experiment = Experiment(project_name=args.name_proj)
    experiment.set_name(args.name_exp)

    hyper_params = {
        "num_classes": int(args.num_classes),
        "batch_size": BATCH_SIZE,
        "num_epochs": EPOCHS,
        "learning_rate": float(args.lr),
        "data_aug": int(args.data_aug),
        "normalization": int(args.norm),
        "loss_weights": int(args.loss_weights),
        "seed": int(args.seed),
        "val_perc": int(args.val_perc),
        "fine_tune": int(args.fine_tune),
        "small_net": int(args.small_net)
    }

    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(model)

    # Parameters counter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    experiment.log_other('num_parameters', params)

    # weights
    save_weights_path = os.path.join(args.weights_path, args.name_exp)
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                optimizer=optimizer, num_epochs=EPOCHS, experiment=experiment, save_weights_path=save_weights_path,
                bce=int(args.bce))

    # Ex: python main.py --epochs 40 --num_classes 1 --name_exp bce --dataset_csv train_val_datasetcsv.csv
    # --bce 1 --data_aug 1 --loss_weights 0

    # ## check on console
    # num_epochs = EPOCHS
    # dataloaders = data_loaders
    # bce = int(args.bce)
    #
    # # for epoch in range(1, num_epochs + 1):
    # for epoch in range(1, 2):
    #     print('Epoch {}/{}'.format(epoch, num_epochs))
    #     print('-' * 10)
    #
    #     # Each epoch has a training and validation phase
    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             model.train()  # Set model to training mode
    #         else:
    #             model.eval()  # Set model to evaluate mode
    #
    #         running_loss = []
    #         running_corrects = []
    #         all_preds = []
    #         all_labels = []
    #         # Iterate over data.
    #         for id, inputs, labels in dataloaders[phase]:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #
    #             # zero the parameter gradients
    #             optimizer.zero_grad()
    #
    #             # forward
    #             # track history if only in train
    #             with torch.set_grad_enabled(phase == 'train'):
    #                 # Get model outputs and calculate loss
    #                 outputs = model(inputs)
    #
    #                 if bce == 1:
    #                     # quando voglio usare bce loss e l'output del classifier è (b, 1)
    #                     labels = labels.unsqueeze(1).float()  # [10] int64 --> [10, 1] float32
    #
    #                 loss = criterion(outputs, labels)
    #
    #                 if bce == 1:
    #                     # quando voglio usare bce loss e l'output del classifier è (b, 1)
    #                     preds = outputs > 0.5
    #                     acc = (preds == labels).float().mean()
    #                 else:
    #                     preds = outputs.argmax(dim=-1)
    #                     acc = (preds == labels).float().mean()
    #                     # it is the same of (outputs == labels).sum().item() / labels.size(0)
    #
    #                 # backward + optimize only if in training phase
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()
    #
    #             # statistics
    #             running_loss.append(loss.item())
    #             running_corrects.append(acc.item())
    #
    #             all_preds.append(preds.cpu())
    #             all_labels.append(labels.cpu())
    #
    #         epoch_loss = sum(running_loss) / len(running_loss)
    #         epoch_acc = sum(running_corrects) / len(running_corrects)
    #         if bce == 1:
    #             all_labels = torch.cat(all_labels, 0).squeeze(1)
    #             all_preds = torch.cat(all_preds, 0).squeeze(1)
    #         else:
    #             all_labels = torch.cat(all_labels, 0)
    #             all_preds = torch.cat(all_preds, 0)
    #
    #         epoch_f1_score = sklearn.metrics.f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
    #                                                   zero_division=0)
    #         epoch_balanced_acc_score = sklearn.metrics.balanced_accuracy_score(all_labels.cpu().numpy(),
    #                                                                            all_preds.cpu().numpy())
    #
    #         # log metrics on comet ml
    #         experiment.log_metric(phase + '_epoch_loss', epoch_loss, step=epoch)
    #         experiment.log_metric(phase + '_epoch_acc', epoch_acc, step=epoch)
    #         # experiment.log_metric(phase + '_weighted_epoch_acc', epoch_weighted_acc, step=epoch)
    #         experiment.log_metric(phase + '_epoch_f1_score', epoch_f1_score, step=epoch)
    #         experiment.log_metric(phase + '_epoch_balanced_acc', epoch_balanced_acc_score, step=epoch)
    #
    #         # long() --> torch.int64, int() --> torch.int32
    #         experiment.log_confusion_matrix(title=phase + '_confusion_matrix_' + str(epoch),
    #                                         y_true=all_labels.cpu().long(), y_predicted=all_preds.cpu().long(),
    #                                         labels=['emolitico', 'lipemico'], step=epoch,
    #                                         file_name=phase + '_confusion_matrix_' + str(epoch) + '.json')
    #
    #         print('{} Loss: {:.4f} - Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
    #
    #         print(sklearn.metrics.classification_report(all_labels.cpu(), all_preds.cpu(),
    #                                                     target_names=['emolitico', 'lipemico']))

