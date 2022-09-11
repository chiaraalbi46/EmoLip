""" Focal Loss implementation """

import torch
import torch.nn.functional as F

# https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
class FocalLoss(torch.nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):

        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FL(torch.nn.Module):
    def __init__(self, gamma, alpha=-1, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(inputs=inputs, targets=targets, alpha=self.alpha, gamma=self.gamma,
                                  reduction=self.reduction)


if __name__ == '__main__':
    import pandas as pd
    from dataset import TubeDataset
    from torch.utils.data import DataLoader
    from MVCNN import MVCNN_small

    class_id_map = {'emolitico': 0, 'lipemico': 1}

    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    #######################
    window_offset = []
    window_size = [172, 534]
    window_origin = [379, 482]  # 1
    target_size = (IMG_HEIGHT, IMG_WIDTH)
    ########################

    BATCH_SIZE = 10

    csv_in = 'train_val_dataset_augmented_42.csv'
    new_csv = pd.read_csv(csv_in, names=['data', 'id', 'image', 'label', 'split'])

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

    train_loader = DataLoader(da_t, batch_size=BATCH_SIZE, shuffle=True)

    id, inputs, labels = next(iter(train_loader))

    model = MVCNN_small(num_classes=1)
    out = model(inputs)

    labels_ = labels.unsqueeze(1).float()
    fl = FL(reduction='mean', gamma=2.)
    loss = fl(inputs=out, targets=labels_)

    # ###
    # model = MVCNN_small(num_classes=2)
    # out = model(inputs)
    #
    # fl1 = FocalLoss(reduction='mean', gamma=2.)
    # loss1 = fl1(out, labels)
