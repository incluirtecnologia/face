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

from sklearn.metrics import confusion_matrix
import seaborn as sn


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
# Configurando hiperparâmetros.
args = {
    'epoch_num': 200,     # Número de épocas.
    'lr': 1e-3,           # Taxa de aprendizado.
    'weight_decay': 1e-3,  # Penalidade L2 (Regularização).
    'batch_size': 200,     # Tamanho do batch.
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
    "foco" : 4,
    "negativa" : 5,
    "qu" : 6,
    "relativa": 7,
    "s_n" : 8,
    "topicos" : 9
}
who = [
    #    'fernando',
       'felipe'
]

cld = CreateLandmarkDataset()

for w in who:
    for EF in EFs:
        path = 'data/faf/{}/{}'.format(w, EF)

        if os.path.isfile(path+'/crop/rotulosEF.txt'):
          cld.datasetWithoutZeros(root_dir=path)

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



data_transform = transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor()
])

datasets_train = []
datasets_test = []

for w in who:
    for EF in EFs:
        label = EFs[EF]
        path = 'data/faf/{}/{}/crop'.format(w, EF)
        transformed_dataset = FafLoad(csv_file=path+'/face_landmarks.csv',
                                      root=path,
                                      transforms=data_transform,
                                      label=label,
                                      rotulos = '/rotulos.txt'
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
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(dt_test,
                         batch_size=args['batch_size'],
                         shuffle=True,
                         drop_last=True)

print(len(train_loader.dataset), len(test_loader.dataset))


# Definindo a rede
net = nn.Sequential(
        ## ConvBlock 1
        nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),        # entrada: (b, 3, 32, 32) e saida: (b, 6, 28, 28)
        nn.BatchNorm2d(6),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),           # entrada: (b, 6, 28, 28) e saida: (b, 6, 14, 14)

        ## ConvBlock 2
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),       # entrada: (b, 6, 14, 14) e saida: (b, 16, 10, 10)
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),           # entrada: (b, 16, 10, 10) e saida: (b, 16, 5, 5)

        ## ConvBlock 3
        nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),     # entrada: (b, 16, 5, 5) e saida: (b, 120, 1, 1)
        nn.BatchNorm2d(120),
        nn.Tanh(),
        nn.Flatten(),  # lineariza formando um vetor                # entrada: (b, 120, 1, 1) e saida: (b, 120*1*1) = (b, 120)

        ## DenseBlock
        nn.Linear(120, 84),                                         # entrada: (b, 120) e saida: (b, 84)
        nn.Tanh(),
        nn.Linear(84, 10),                                          # entrada: (b, 84) e saida: (b, 10)
        )

# Subindo no hardware de GPU (se disponível)
net = net.to(args['device'])

criterion = nn.CrossEntropyLoss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

train_losses, test_losses = [], []
loop_nn = Net()
for epoch in range(args['epoch_num']):

    # Train
    train_losses.append(loop_nn.train(train_loader, net, epoch,
                        args, criterion, optimizer))

    # Validate
    test_losses.append(loop_nn.validate(
        test_loader, net, epoch, args, criterion, optimizer))



cm = confusion_matrix(test_losses[190][2][0], test_losses[190][1][0])

a_file = open("results/result.txt", "w")
for row in test_losses:
    np.savetxt(a_file, row)
a_file.close()

EFs['neutro'] = 0

df_cm = pd.DataFrame(cm, index = EFs,
                  columns = [i for i in EFs])
plt.figure(figsize = (20,14))
sn.heatmap(df_cm, annot=True, fmt='d')

plt.savefig('results/output.png', dpi=400)