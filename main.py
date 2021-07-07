from Resources.FaceLandmarksDataset import FaceLandmarksDataset
from Resources.CreateLandmarkDataset import CreateLandmarkDataset
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


EFs = [
    "afirmativa",
    "condicional",
    "duvida",
    "foco",
    "negativa",
    "qu",
    "relativa",
    "s_n",
    "topicos"
]
who = [
    'fernando',
    'felipe'
]

cld = CreateLandmarkDataset()


for w in who:
  for EF in EFs:
    path = 'data/faf/{}/{}'.format(w, EF)

    if not os.path.exists(path+'/crop'):
      if not os.path.isfile(path+'/face_landmarks.csv'):
        cld.createLandmarkDataset(root_dir=path)
      face_dataset = FaceLandmarksDataset(csv_file=path+'/face_landmarks.csv',root_dir=path) 
      cld.cropImages(root_dir=path, face_dataset=face_dataset)
      
    if not os.path.isfile(path+'/crop/face_landmarks.csv'):
      cld.createLandmarkDataset(root_dir=path+'/crop')
      
    
    if os.path.isfile(path+'/crop/rotulos.txt'):
      targets = list(np.loadtxt(path+'/crop/rotulos.txt'))
    else:
      targets = list([])
    if (len(targets) < 1):
      cld.targetsCrop(root_dir=path)
      


# datasets = []

# data_transform = transforms.Compose([
#                                      transforms.Resize(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                           std=[0.229, 0.224, 0.225])
#                   ])

# for w in who:
#   for EF in EFs:
#     path = '/content/drive/MyDrive/datasets/photos/{}/{}/crop'.format(w, EF)
#     transformed_dataset = FaceLandmarksDataset(csv_file=path+'/face_landmarks.csv',
#                                            root_dir=path,
#                                            transform=data_transform
#                                            )
#     datasets.append(transformed_dataset)
#     print(w,EF,len(transformed_dataset))

# auxTrain = torch.utils.data.ConcatDataset(datasets)