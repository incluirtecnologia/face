import dlib
import glob
import csv
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import time, os
import pandas as pd
import numpy as np

class CreateLandmarkDataset:
    pass

    def createLandmarkDataset(self, root_dir):

      detector = dlib.get_frontal_face_detector()
      predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
      num_landmarks = 68

      with open(root_dir+'/face_landmarks.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        header = ['image_name']
        for i in range(num_landmarks):
          header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

        csv_writer.writerow(header)

        for f in glob.glob(root_dir+'/*.jpg'):
          try:
            testImg = Image.open(f) # open the image file
            testImg.verify() # verify that it is, in fact an image
            img = io.imread(f)
            dets = detector(img, 1)  # face detection
            row = [f]

            # ignore all the files with no or more than one faces detected.
            if len(dets) == 1:
              d = dets[0]
              # Get the landmarks/parts for the face in box d.
              shape = predictor(img, d)
              for i in range(num_landmarks):
                part_i_x = shape.part(i).x
                part_i_y = shape.part(i).y
                row += [part_i_x, part_i_y]

            csv_writer.writerow(row)
          except:
            print(f)

    def show_landmarks(self, image, landmarks):
      """Show image with landmarks"""
      plt.imshow(image)
      plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
      plt.pause(0.001)  # pause a bit so that plots are updated
      plt.show()

    def detect_faces(self, image):
      # Create a face detector
      face_detector = dlib.get_frontal_face_detector()

      # Run detector and get bounding boxes of the faces on image.
      detected_faces = face_detector(image, 1)
      face_frames = [(x.left(), x.top(),
                      x.right(), x.bottom()) for x in detected_faces]
      return face_frames

    def cropImages(self, root_dir, face_dataset):
      if not os.path.exists(root_dir+'/crop'):
        os.makedirs(root_dir+'/crop')
      landmarks_frame = pd.read_csv(root_dir+'/face_landmarks.csv')

      for i in range(len(face_dataset)):
        img_name = landmarks_frame.iloc[i, 0]
        image = io.imread(img_name)

        img_crop_name = img_name.replace(root_dir,'')
        # Detect faces
        detected_faces = self.detect_faces(image)

        # Crop faces and plot
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(image).crop(face_rect)
            face.save(root_dir+'/crop/'+img_crop_name)

    def targetsCrop(self, root_dir):
      targets = list(np.loadtxt(root_dir + '/rotulos.txt'))

      with open(root_dir+'/crop/rotulos.txt', 'w') as txtfile:
        lf = pd.read_csv(root_dir+'/face_landmarks.csv')
        lf_crop = pd.read_csv(root_dir+'/crop/face_landmarks.csv')
        for i in range(len(lf)):
          for j in range(len(lf_crop)):
            if ("/crop"+lf.iloc[i, 0].replace(root_dir,'') == lf_crop.iloc[j, 0].replace(root_dir,'')):
              txtfile.write(str(targets[i])+ '\n')
              break
        txtfile.close()

    def datasetWithoutZeros(self, root_dir):
      targets = list(np.loadtxt(root_dir + '/rotulos.txt'))
      lf_crop = pd.read_csv(root_dir+'/crop/face_landmarks.csv')

      with open(root_dir+'/crop/rotulosEF.txt', 'w') as rotulosEF:
        with open(root_dir+'/crop/fileEF.txt', 'w') as fileEF:
          for j in range(len(lf_crop)):
            if(targets[j]==1):
              rotulosEF.write(str(targets[j])+ '\n')
              fileEF.write(str(lf_crop.iloc[j, 0]+ '\n'))
      rotulosEF.close()
      fileEF.close()