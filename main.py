from Resources.FaceLandmarksDataset import FaceLandmarksDataset
from Resources.CreateLandmarkDataset import CreateLandmarkDataset
from Resources.FafLoad import FafLoad
from Resources.Net import Net

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import datasets, models
import dlib
from PIL import Image
# Ignore warnings
import warnings
from torchvision import transforms
from skimage import io, transform

# Implementação e treinamento da rede
import torch
from torch import nn, optim

# Carregamento de Dados
from torch.utils.data import DataLoader
from torchvision import datasets


from torch import optim
import torch.utils.data as data_utils

# Plots e análises
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time

import os

warnings.filterwarnings("ignore")

# Configurando hiperparâmetros.
args = {
    'epoch_num': 5,     # Número de épocas.
    'lr': 1e-3,           # Taxa de aprendizado.
    'weight_decay': 1e-3,  # Penalidade L2 (Regularização).
    'batch_size': 20,     # Tamanho do batch.
}

# Definindo dispositivo de hardware
if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])


EFs = {
    "afirmativa" : 1,
    "condicional": 2,
    "duvida" : 3,
    "foco" : 1,
    "negativa" : 5,
    "qu" : 6,
    "relativa": 7,
    "s_n" : 8,
    "topicos" : 9
}
who = [
    # 'fernando',
    'felipe'
]

cld = CreateLandmarkDataset()


for w in who:
    for EF in EFs:
        path = 'data/faf/{}/{}'.format(w, EF)

        if not os.path.exists(path+'/crop'):
            if not os.path.isfile(path+'/face_landmarks.csv'):
                cld.createLandmarkDataset(root_dir=path)
            face_dataset = FaceLandmarksDataset(
                csv_file=path+'/face_landmarks.csv', root_dir=path)
            cld.cropImages(root_dir=path, face_dataset=face_dataset)

        if not os.path.isfile(path+'/crop/face_landmarks.csv'):
            cld.createLandmarkDataset(root_dir=path+'/crop')

        if os.path.isfile(path+'/crop/rotulos.txt'):
            targets = list(np.loadtxt(path+'/crop/rotulos.txt'))
        else:
            targets = list([])
        if (len(targets) < 1):
            cld.targetsCrop(root_dir=path)


datasets_train = []
datasets_test = []

data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

for w in who:
    for EF in EFs:
        label = EFs[EF]
        path = 'data/faf/{}/{}/crop'.format(w, EF)
        transformed_dataset = FafLoad(csv_file=path+'/face_landmarks.csv',
                                      root=path,
                                      transforms=data_transform,
                                      label=label
                                      )
        idx = int(len(transformed_dataset)*0.6)
        indices_train = torch.arange(start=0, end=idx)
        indices_tests = torch.arange(
            start=idx, end=len(transformed_dataset)-1)
        t = data_utils.Subset(transformed_dataset, indices_train)
        tt = data_utils.Subset(transformed_dataset, indices_tests)

        datasets_train.append(t)
        datasets_test.append(tt)

        print(w, EF, len(t), len(tt))

dt_train = torch.utils.data.ConcatDataset(datasets_train)
dt_test = torch.utils.data.ConcatDataset(datasets_test)

train_loader = DataLoader(dt_train,
                          batch_size=args['batch_size'],
                          shuffle=False,
                          drop_last=True)

test_loader = DataLoader(dt_test,
                         batch_size=args['batch_size'],
                         shuffle=False,
                         drop_last=True)

net = models.vgg16_bn(pretrained=True).to(args['device'])
print(net)

in_features = list(net.children())[-1][-1].in_features

new_classifier = list(net.classifier.children())[:-1]
new_classifier.append(nn.Linear(in_features, 2))

net.classifier = nn.Sequential(*new_classifier).to(args['device'])
print(net.classifier)

optimizer = optim.Adam([
    {'params': net.features.parameters(
    ), 'lr': args['lr']*0.2, 'weight_decay': args['weight_decay']*0.2},
    {'params': net.classifier.parameters(
    ), 'lr': args['lr'], 'weight_decay': args['weight_decay']}
], lr=0)
criterion = nn.CrossEntropyLoss().to(args['device'])

train_losses, test_losses = [], []
loop_nn = Net()
for epoch in range(args['epoch_num']):

    # Train
    train_losses.append(loop_nn.train(train_loader, net, epoch,
                        args, criterion, optimizer))

    # Validate
    test_losses.append(loop_nn.validate(
        test_loader, net, epoch, args, criterion, optimizer))
