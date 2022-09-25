""" Cross validation """

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


# todo: posizione statistics

def reset_weights(m):
    """
      Try resetting model weights to avoid
      weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


if __name__ == '__main__':

    from comet_ml import Experiment
    from MVCNN import MVCNN, MVCNN_small
    import torch
    import torch.optim as optim
    import os
    import pandas as pd

    from torch.utils.data import DataLoader
    from dataset import TubeDataset
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Train MVCNN using Cross Validation")

    parser.add_argument("--epochs", dest="epochs", default=1, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=10, help="batch size")
    parser.add_argument("--lr", dest="lr", default=0.0005, help="learning rate train")  # 0.00005 for fine tuning
    parser.add_argument("--weight_decay", dest="weight_decay", default=0., help="weight decay")

    parser.add_argument("--num_classes", dest="num_classes", default=1, help="number of classes of the dataset")

    parser.add_argument("--device", dest="device", default='0', help="choose GPU")
    parser.add_argument("--name_proj", dest="name_proj", default='Bioimmagini', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")

    parser.add_argument("--weights_path", dest="weights_path", default='./weights',
                        help="path to the folder where storing the model weights")
    parser.add_argument("--dataset_csv", dest="dataset_csv", default='./augmented_dataset.csv',  # open.csv
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
    parser.add_argument("--small_net", dest="small_net", default=1, help="1 to use MVCNN_small, 0 otherwise")

    parser.add_argument("--n_folds", dest="n_folds", default=2, help="number of folds for StratifiedKFold validation")

    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    bce = int(args.bce)

    if int(args.small_net) == 1:
        model = MVCNN_small(num_classes=int(args.num_classes))
    else:
        model = MVCNN(num_classes=int(args.num_classes))

    if int(args.fine_tune) == 0:
        # freeze the weights in the feature extraction block of the network (resnet base)
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        pass

    # Set fixed random number seed
    torch.manual_seed(int(args.seed))

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
        "bce:": bce,
        # "loss_weights": int(args.loss_weights),
        "seed": int(args.seed),
        # "val_perc": int(args.val_perc),  #  ???
        # "fine_tune": int(args.fine_tune),
        "small_net": int(args.small_net),
        "weight_decay": int(args.weight_decay)
    }

    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(model)

    # Parameters counter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    experiment.log_other('num_parameters', params)

    # Weights
    save_weights_path = os.path.join(args.weights_path, args.name_exp)
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    # For fold results
    results = {}

    datasetcsv = pd.read_csv(args.dataset_csv, names=['data', 'id', 'image', 'label'])
    class_id_map = {'emolitico': 0, 'lipemico': 1}

    #######################
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]  # 1
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    ########################

    dataset = TubeDataset(dataframe=datasetcsv,
                          window_origin=window_origin,
                          window_offset=window_offset,
                          window_size=window_size,
                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                          class_id_map=class_id_map,
                          num_views=8, data_aug=0, norm=int(args.norm))

    # check balancing
    # for i in dataset.ids[train_ids]:
    #     if dataset.id_label_map[i] == 'emolitico':
    #         emo.append(i)
    #     else:
    #         lip.append(i)

    labels = []  # all the labels based on the all dataset (train and validation)
    for id in dataset.ids:
        label = dataset.id_label_map[id]
        labels.append(label)

    # Define the K-fold Cross Validator
    k_folds = int(args.n_folds)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    if bce == 1:
        print("Using BCEWithLogitsLoss")
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        print("Using CrossEntropyLoss")
        criterion = torch.nn.CrossEntropyLoss()

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, labels)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        print("train ids, test ids: ", len(train_ids), len(test_ids))

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

        # Init the neural network
        network = model
        network.apply(reset_weights)
        network.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=float(args.lr), weight_decay=int(args.weight_decay))

        # Run the training loop for defined number of epochs
        for epoch in range(1, EPOCHS + 1):

            print('Epoch {}/{}'.format(epoch, EPOCHS))
            print('-' * 10)

            network.train()

            correct, total, running_loss = 0, 0, 0

            # Iterate over the DataLoader for training data
            for train_id, inputs, train_labels in train_loader:

                inputs = inputs.to(device)
                train_labels = train_labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                if bce == 1:
                    train_labels = train_labels.unsqueeze(1).float()

                # Compute loss
                loss = criterion(outputs, train_labels)

                # Get model predictions
                if bce == 1:
                    preds = outputs > 0.5
                    acc = (preds == train_labels).float().sum().item()  # .mean()
                else:
                    preds = outputs.argmax(dim=-1)
                    acc = (preds == train_labels).float().sum().item()  # .mean()

                # statistics todo: qui non va bene
                total += train_labels.size(0)
                correct += acc
                running_loss += loss.item()

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

            # epoch_loss = sum(running_loss) / len(running_loss)
            epoch_acc = correct / total
            epoch_loss = running_loss / len(train_loader)

            print('{} Loss: {:.4f} - Acc: {:.4f} '.format('Train', epoch_loss, epoch_acc))

            # log metrics on comet ml
            experiment.log_metric(str(fold) + '_train_epoch_loss', epoch_loss, step=epoch)
            experiment.log_metric(str(fold) + '_train_epoch_acc', epoch_acc, step=epoch)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        torch.save(network.state_dict(), save_weights_path + '/model_fold_' + str(fold) + '.pth')

        # Evaluationfor this fold
        network.eval()
        correct, total, running_loss = 0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for val_id, val_inputs, val_labels in val_loader:

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                # Generate outputs
                val_outputs = network(val_inputs)

                if bce == 1:
                    val_labels = val_labels.unsqueeze(1).float()

                val_loss = criterion(val_outputs, val_labels)

                # Set total and correct
                if bce == 1:
                    val_preds = val_outputs > 0.5
                    val_acc = (val_preds == val_labels).float().sum().item()
                else:
                    val_preds = val_outputs.argmax(dim=-1)
                    val_acc = (val_preds == val_labels).float().sum().item()

                all_preds.append(val_preds.cpu())
                all_labels.append(val_labels.cpu())

                total += val_labels.size(0)
                correct += val_acc
                running_loss += val_loss.item()

            if bce == 1:
                all_labels = torch.cat(all_labels, 0).squeeze(1)
                all_preds = torch.cat(all_preds, 0).squeeze(1)
            else:
                all_labels = torch.cat(all_labels, 0)
                all_preds = torch.cat(all_preds, 0)

            # Print accuracy
            print('Loss for fold %d: %f' % (fold, (running_loss / len(val_loader))))
            print('Accuracy for fold %d: %f' % (fold, (correct / total)))  # 100.0 *  # %%
            print('--------------------------------')
            results[fold] = correct / total

            # log metrics on comet ml
            experiment.log_metric(str(fold) + '_val_loss', (running_loss / len(val_loader)), step=fold)
            experiment.log_metric(str(fold) + '_val_acc', (correct / total), step=fold)

            experiment.log_confusion_matrix(title=str(fold) + '_val_confusion_matrix',
                                            y_true=all_labels.cpu().long(), y_predicted=all_preds.cpu().long(),
                                            labels=['emolitico', 'lipemico'], step=fold,
                                            file_name=str(fold) + '_val_confusion_matrix.json')

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')  # %
        sum += value
    print(f'Average: {sum / len(results.items())}')  # %
