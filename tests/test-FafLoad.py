from Resources.FafLoad import FafLoad
from torchvision import transforms

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

data_transform = transforms.Compose([
    transforms.Resize(224),
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
                                      label = label
                                      )
