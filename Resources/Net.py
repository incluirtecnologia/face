import torch
import time
from sklearn.metrics import accuracy_score
import numpy as np

class Net():
    """Face Landmarks dataset."""
    def train(self,train_loader, net, epoch, args, criterion, optimizer):

        # Training mode
        net.train()

        start = time.time()

        epoch_loss  = []
        pred_list, rotulo_list = [], []
        for batch in train_loader:

            dado, rotulo = batch
            rotulo = rotulo['labels']
            # Cast do dado na GPU
            dado = dado.to(args['device'])
            rotulo = rotulo.to(args['device'])

            # Forward
            ypred = net(dado)
            loss = criterion(ypred, rotulo)
            epoch_loss.append(loss.cpu().data)

            _, pred = torch.max(ypred, axis=1)
            pred_list.append(pred.cpu().numpy())
            rotulo_list.append(rotulo.cpu().numpy())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.asarray(epoch_loss)
        pred_list  = np.asarray(pred_list).ravel()
        rotulo_list  = np.asarray(rotulo_list).ravel()

        acc = accuracy_score(pred_list, rotulo_list)

        end = time.time()
        print('#################### Train ####################')
        print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

        return epoch_loss.mean()


    def validate(self, test_loader, net, epoch, args, criterion, optimizer):

        # Evaluation mode
        net.eval()

        start = time.time()

        epoch_loss  = []
        pred_list, rotulo_list = [], []
        arr_pred_list, arr_rotulo_list = [], []
        with torch.no_grad():
            for batch in test_loader:

                dado, rotulo = batch
                rotulo = rotulo['labels']

                # Cast do dado na GPU
                dado = dado.to(args['device'])
                rotulo = rotulo.to(args['device'])

                # Forward
                ypred = net(dado)
                loss = criterion(ypred, rotulo)
                epoch_loss.append(loss.cpu().data)

                _, pred = torch.max(ypred, axis=1)
                pred_list.append(pred.cpu().numpy())
                rotulo_list.append(rotulo.cpu().numpy())

        epoch_loss = np.asarray(epoch_loss)
        pred_list  = np.asarray(pred_list).ravel()
        rotulo_list  = np.asarray(rotulo_list).ravel()

        acc = accuracy_score(pred_list, rotulo_list)
        arr_pred_list.append(pred_list)
        arr_rotulo_list.append(rotulo_list)
        end = time.time()
        print('********** Validate **********')
        print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f\n' % (epoch, epoch_loss.mean(), epoch_loss.std(), acc*100, end-start))

        return epoch_loss.mean(), arr_pred_list, arr_rotulo_list