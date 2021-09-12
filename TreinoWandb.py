import joblib
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import os
import utils
import models
import data_builder
import train_datagen
from callbacks import early_stopping
# Ignore warnings
import warnings

import wandb
from wandb.keras import WandbCallback

warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = '809d3c86e8530a6fc2279dbae40625f2d1a5d17a'
os.environ["WANDB_MODE"] = "online"

wandb.login()

sweep_config = {
    'method': 'random',  # grid, random
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {
            'values': [50, 100, 150, 200]
        },
        'batch_size': {
            'values': [256, 128, 64, 32]
        },
        'learning_rate': {
            'values': [1e-1, 1e-2, 1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['adam', 'nadam']
        },
        'train_ratio': {
            'values': [0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'model': {
            'values': [
                'CNNModel',
                'CNN_ROI1_ROI2Model',
                'CNN_ROI1_ROI2_HOGFeat_Model',
                'CNN_ROI1_ROI2_KLDIST_Model',
                'VGG16'
            ]
        },
        'shuffle': {
            'values': [0, 1]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="face")

print("tensorflow ", tf.__version__, "\n")

args = {}

args['dataset'] = "FAF"
args['emotions'] = "afirmativa,condicional,duvida,foco,negativa,qu,relativa,s_n,topicos"
# args['emotions'] = "afirmativa"
args["train_datagen"] = None
args["save_confusion_matrix"] = True
DATA_PATH = "inputs/" + args["dataset"] + "/"
OUTPUT_PATH = "outputs/"
EMOTIONS = list(args["emotions"].split(","))

if not args["train_datagen"] is None:
    train_datagen = train_datagen.train_datagen[args["train_datagen"]]
else:
    train_datagen = None




def __enter__(self):
    return self


def train():

    config = {
        'epochs': 2,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'train_ratio': 0.5,
        'shuffle': 0,
        'model': 'VGG16'
    }
    wandb.init(config=config, project="face", entity='fernandoafreitas')
    print(wandb.config)
    config = wandb.config

    args["optimizer"] = config.optimizer
    args["model"] = config.model
    args["shuffle"] = config.shuffle
    args["train_ratio"] = config.train_ratio
    args["batch_size"] = config.batch_size
    args["epochs"] = config.epochs
    args["learning_rate"] = config.learning_rate
    args["learning_rate"] = config.learning_rate
    args["random_state"] = 42


    RUN_NAME = f"{config.model.__class__.__name__}_{args['dataset']}_{args['epochs']}_{args['batch_size']}_{args['shuffle']}_{args['train_ratio']}_{args['optimizer']}_{args['learning_rate']}_{len(EMOTIONS)}emo"

    if args["optimizer"] == "nadam":
        optim = optimizers.Nadam(args["learning_rate"])
    else:
        optim = optimizers.Adam(args["learning_rate"])

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=3), WandbCallback()]

    if args["model"] == "CNNModel":
        model = models.CNNModel()
        img_arr, img_label, label_to_text = data_builder.ImageToArray(
            DATA_PATH, EMOTIONS).build_from_directory()
        img_arr = img_arr / 255.

        X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, shuffle=args["shuffle"], stratify=img_label,
                                                            train_size=args["train_ratio"], random_state=args["random_state"])
        print(
            f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape} \n")

        model.train(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=args["batch_size"],
            epochs=args["epochs"],
            optim=optim,
            callbacks=callbacks,
            train_datagen=train_datagen,
        )

        if args["save_confusion_matrix"]:
            model.evaluate(X_test, y_test, OUTPUT_PATH +
                        "confusion_matrix/" + RUN_NAME + ".png")

    elif args["model"] == "CNN_ROI1_ROI2Model":
        model = models.CNN_ROI1_ROI2Model()
        roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(
            DATA_PATH, EMOTIONS).build_from_directory()

        img2arr_obj = data_builder.ImageToArray(
            DATA_PATH, EMOTIONS, img_to_exclude)
        img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
        img2arr_obj.class_image_count()

        img_arr = img_arr / 255.
        roi1_arr = roi1_arr / 255.
        roi2_arr = roi2_arr / 255.

        Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, y_train, y_test =\
            train_test_split(img_arr, roi1_arr, roi2_arr, img_label,
                            shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

        print(
            f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, y_train: {y_train.shape}")
        print(
            f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, y_test: {y_test.shape} \n")

        model.train(
            Xtrain_img, Xtrain_roi1, Xtrain_roi2,
            y_train,
            validation_data=([Xtest_img, Xtest_roi1, Xtest_roi2], y_test),
            batch_size=args["batch_size"],
            epochs=args["epochs"],
            optim=optim,
            callbacks=callbacks,
            train_datagen=train_datagen,
        )

        if args["save_confusion_matrix"]:
            model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2], y_test,
                        OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

    elif args["model"] == "CNN_ROI1_ROI2_HOGFeat_Model":
        model = models.CNN_ROI1_ROI2_HOGFeat_Model()

        roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(
            DATA_PATH, EMOTIONS).build_from_directory()
        hogfeat = data_builder.ImageToHOGFeatures(
            DATA_PATH, EMOTIONS, img_to_exclude).build_from_directory()

        img2arr_obj = data_builder.ImageToArray(
            DATA_PATH, EMOTIONS, img_to_exclude)
        img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
        img2arr_obj.class_image_count()

        img_arr = img_arr / 255.
        roi1_arr = roi1_arr / 255.
        roi2_arr = roi2_arr / 255.

        Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, Xtrain_hogfeat, Xtest_hogfeat, y_train, y_test =\
            train_test_split(img_arr, roi1_arr, roi2_arr, hogfeat, img_label,
                            shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

        print(f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, Xtrain_hogfeat: {Xtrain_hogfeat.shape}, y_train: {y_train.shape}")
        print(f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, Xtest_hogfeat: {Xtest_hogfeat.shape}, y_test: {y_test.shape} \n")

        model.train(
            Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat,
            y_train,
            validation_data=(
                [Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_hogfeat], y_test),
            batch_size=args["batch_size"],
            epochs=args["epochs"],
            optim=optim,
            callbacks=callbacks,
            train_datagen=train_datagen,
        )

        if args["save_confusion_matrix"]:
            model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_hogfeat],
                        y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

    elif args["model"] == "CNN_ROI1_ROI2_KLDIST_Model":
        model = models.CNN_ROI1_ROI2_KLDIST_Model()

        roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(
            DATA_PATH, EMOTIONS).build_from_directory()
        kl_dists = data_builder.ImageToKeyLandmarksDistances(
            DATA_PATH, EMOTIONS, img_to_exclude).build_from_directory()

        img2arr_obj = data_builder.ImageToArray(
            DATA_PATH, EMOTIONS, img_to_exclude)
        img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
        img2arr_obj.class_image_count()

        img_arr = img_arr / 255.
        roi1_arr = roi1_arr / 255.
        roi2_arr = roi2_arr / 255.

        Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, Xtrain_kldist, Xtest_kldist, y_train, y_test =\
            train_test_split(img_arr, roi1_arr, roi2_arr, kl_dists, img_label,
                            shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

        print(f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, Xtrain_kldist: {Xtrain_kldist.shape}, y_train: {y_train.shape}")
        print(f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, Xtest_kldist: {Xtest_kldist.shape}, y_test: {y_test.shape} \n")

        model.train(
            Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_kldist,
            y_train,
            validation_data=(
                [Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_kldist], y_test),
            batch_size=args["batch_size"],
            epochs=args["epochs"],
            optim=optim,
            callbacks=callbacks,
            train_datagen=train_datagen,
        )

        if args["save_confusion_matrix"]:
            model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_kldist],
                        y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

    elif args["model"] == "VGG16":
        model = models.VGG16()
        img_arr, img_label, label_to_text = data_builder.ImageToArray(
            DATA_PATH, EMOTIONS).build_from_directory()
        img_arr = img_arr / 255.

        X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, shuffle=args["shuffle"], stratify=img_label,
                                                            train_size=args["train_ratio"], random_state=args["random_state"])
        print(
            f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape} \n")

        model.train(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=args["batch_size"],
            epochs=args["epochs"],
            optim=optim,
            callbacks=callbacks,
            train_datagen=train_datagen,
        )

        if args["save_confusion_matrix"]:
            model.evaluate(X_test, y_test, OUTPUT_PATH +
                        "confusion_matrix/" + RUN_NAME + ".png")

    else:
        raise ValueError(
            f"Invalid model {args['model']}, only `CNNModel`, `CNN_ROI1_ROI2Model` and `CNN_ROI1_ROI2_HOGFeat_Model` are supported")

    try:
        model.save_model(OUTPUT_PATH + "models/" + RUN_NAME + ".h5")
        print(label_to_text)
        joblib.dump(label_to_text, OUTPUT_PATH +
                    "label2text/label2text_" + RUN_NAME + ".pkl")
        model.save_training_history(
            OUTPUT_PATH + "epoch_metrics/" + RUN_NAME + ".png")

        plot_model(model.model, show_shapes=True, show_layer_names=True, expand_nested=True,
                dpi=50, to_file=OUTPUT_PATH + "architectures/" + model.__class__.__name__ + ".png")
    except:
        print("An exception occurred")

    wandb.finish()

wandb.agent(sweep_id, function=train())