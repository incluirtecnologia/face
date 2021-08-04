import matplotlib.pyplot as plt

class VisualizarResultados:

    def plot_history(history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='lower left')
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.show()

