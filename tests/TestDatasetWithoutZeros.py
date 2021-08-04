import sys
sys.path.append('/home/incluir-fernando/Projetos/USP/face')

from Resources.FaceLandmarksDataset import FaceLandmarksDataset
from Resources.CreateLandmarkDataset import CreateLandmarkDataset


import os
import numpy as np

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
    'fernando',
    'felipe'
]

cld = CreateLandmarkDataset()
for w in who:
    for EF in EFs:
        path = 'data/faf/{}/{}'.format(w, EF)

        if not os.path.isfile(path+'/crop/rotulosEF.txt'):
          cld.datasetWithoutZeros(root_dir=path)

        if not os.path.exists(path+'/crop'):
            if not os.path.isfile(path+'/face_landmarks.csv'):
                cld.createLandmarkDataset(root_dir=path)
            face_dataset = FaceLandmarksDataset(
                csv_file=path+'/face_landmarks.csv', root=path, label = EFs[EF], transforms = data_transform)
            cld.cropImages(root_dir=path, face_dataset=face_dataset)

        if not os.path.isfile(path+'/crop/face_landmarks.csv'):
            cld.createLandmarkDataset(root_dir=path+'/crop')

        if os.path.isfile(path+'/crop/rotulos.txt'):
            targets = list(np.loadtxt(path+'/crop/rotulos.txt'))
        else:
            targets = list([])
        if (len(targets) < 1):
            cld.targetsCrop(root_dir=path)