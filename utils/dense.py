import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import librosa
import random

DATA_PATH = "JSON/data_90.json"
#DATA_PATH = "JSON/data.json"
#DATA_PATH = "JSON/pca.json"
SAVED_MODEL_PATH = "../models/model.h5"
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 5
LEARNING_RATE = 0.0001
NUM_LABELS = 5 #300
AUDIO = 90
ENDING = NUM_LABELS * AUDIO


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)


    X = np.array(data["MFCCs_delta1_delta2"][0:ENDING])
    y = np.array(data["labels"][0:ENDING])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, shuffle=True, stratify=y_train)


    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=LEARNING_RATE):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """


    # # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))
    tf.keras.layers.Dropout(0.4)

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))
    tf.keras.layers.Dropout(0.4)

    # 3nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    tf.keras.layers.Dropout(0.4)

    # 4nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))


    # 1st conv layer
    # model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape,
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)
    #                                  ))
    # model.add(tf.keras.layers.BatchNormalization())
    # #model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # model.add(tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same'))
    # tf.keras.layers.Dropout(0.5)


    model.add(tf.keras.layers.Reshape((26, 32)))
    #model.add(tf.keras.layers.Reshape((108, 32)))
    #model.add(tf.keras.layers.Reshape((1998, 64)))
    model.add(tf.keras.layers.LSTM(1024))

    #model.add(tf.keras.layers.Reshape((1024, 1)))
    # model.add(tf.keras.layers.LSTM(128))
    # # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.7))
    #
    # model.add(tf.keras.layers.Reshape((128, 1)))
    #model.add(tf.keras.layers.LSTM(1024))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))

    # #3rd conv layer
    # model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)
    #                                  ))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))
    # tf.keras.layers.Dropout(0.5)
    #
    # #4rd conv layer
    # model.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)
    #                                  ))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    # tf.keras.layers.Dropout(0.5)
    #
    # # 4rd conv layer
    # model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
    #                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)
    #                                  ))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    # tf.keras.layers.Dropout(0.5)

    # model.add(tf.keras.layers.Reshape((108, 32)))
    # model.add(tf.keras.layers.LSTM(4))



    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    tf.keras.layers.Dropout(0.4)

    # softmax output layer
    model.add(tf.keras.layers.Dense(NUM_LABELS, activation='softmax'))

    #optimiser = tf.optimizers.Adam(learning_rate=learning_rate, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
    optimiser = tf.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) #, momentum=0.8, name='SGD', nesterov=True)
    #optimiser = tf.keras.optimizers.Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07, name='Adadelta')


    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])


    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation)
                       # ,callbacks=[earlystop_callback]
                        , callbacks=[learning_rate_scheduler]
                        )
    return history

def schedule(epoch):
    if epoch < 90:
        return 0.001
    else:
        return 0.0001
        # * tf.math.exp(-0.1)

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

    print("Y-Train", y_train, len(y_train))
    print("Y-Val", y_validation, len(y_validation))
    print("Y-Test", y_test, len(y_test))

    # count0 = 0
    # count1 = 0
    # count00 = 0
    # count11 = 0
    # for item in y_train:
    #     if item == 0:
    #         count0 = count0 + 1
    #     if item == 1:
    #         count1 = count1 + 1
    #
    # for item in y_validation:
    #     if item == 0:
    #         count00 = count00 + 1
    #     if item == 1:
    #         count11 = count11 + 1
    #
    # print("TRAIN Count0 / Count1 ", count0, count1)
    # print("VALIDATION Count00 / Count11 ", count00, count11)

    # create network
    print("X Shape ", X_train.shape)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # Evaluating the model on the training and testing set
    # training_loss, training_acc = model.evaluate(X_train, y_train, verbose=0)
    # print("\nTraining loss: {}, training accuracy: {}".format(training_loss, 100*training_acc))

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
