""" Multi-View convolutional neural network (MVCNN) architecture """

import torch
from torchvision import models


class MVCNN(torch.nn.Module):
    def __init__(self, num_classes=1000):  # drop_value ...
        super(MVCNN, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # nb: si aspetta (B, C, H, W)
        fc_in_features = resnet.fc.in_features
        # print("fc in features: ", fc_in_features)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),  # di default p=0.5
            torch.nn.Linear(fc_in_features, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, num_classes)
        )

    def forward(self, inputs):
        # inputs = inputs.transpose(0, 1)
        # views x samples x channels x height x width se inputs.shape = samples x views x height x width x channels
        # inputs = inputs.permute(1, 0, 4, 2, 3)
        # se inputs.shape = samples x views x channels x height x width
        # (con la nuova versione di pre processing che usa transform e lavora sui tensori)
        inputs = inputs.permute(1, 0, 2, 3, 4)  # views x samples x channels x height x width
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            # print("view batch shape post resnet: ", view_batch.shape)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            # print("view batch shape: ", view_batch.shape)
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        # print("pooled views shape: ", pooled_views.shape)
        outputs = self.classifier(pooled_views)
        # print("classifier shape: ", outputs.shape)
        return outputs


class MVCNN_small(torch.nn.Module):
    def __init__(self, num_classes=1000):  # drop_value ...
        super(MVCNN_small, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # nb: si aspetta (B, C, H, W)
        fc_in_features = resnet.fc.in_features  # 512
        # print("fc in features: ", fc_in_features)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),  # di default p=0.5
            torch.nn.Linear(fc_in_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2, 3, 4)  # views x samples x channels x height x width
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            # print("view batch shape post resnet: ", view_batch.shape)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            # print("view batch shape: ", view_batch.shape)
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        # print("pooled views shape: ", pooled_views.shape)
        outputs = self.classifier(pooled_views)
        # print("classifier shape: ", outputs.shape)
        return outputs


# Computing the binary loss by providing the logits, and not the class probabilities
# is usually preferred due to numerical stability.


if __name__ == '__main__':
    import numpy as np
    # model = MVCNN(num_classes=2)
    # print(model)
    model = MVCNN_small(num_classes=2)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    ct = 0
    for child in model.features.children():
        print("ct: ", ct)
        if ct < 7:
            print("in")
            for param in child.parameters():
                param.requires_grad = False
        ct += 1

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)
