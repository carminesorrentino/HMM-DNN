import json
import numpy as np
import tensorflow as tf
import utils.layer_utils as lu
from utils.layer_utils import print_summary as summary
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = "D:/Speaker recognition da consegnare/JSON/hmm_pca_delta_.json"
SAVED_MODEL_PATH = "models/model_.h5"
EPOCHS = 800
BATCH_SIZE = 128
PATIENCE = 5
LEARNING_RATE = 0.01
NUM_LABELS = 90
AUDIO = 90
ENDING = NUM_LABELS * AUDIO


def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        lu.init()
        data = json.load(fp)
    ending = lu.NUM_LABELS * lu.AUDIO

    X = np.array(data["MFCCs_delta1_delta2"][0:ending])
    y = np.array(data["labels"][0:ending])

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


def build_model(input_shape, loss="sparse_categorical_crossentropy", LEARNING_RATE=LEARNING_RATE, NUM_LABELS=NUM_LABELS, PATIENCE=PATIENCE):
    """Build neural network using keras.
        :param num_labels: number of labels
        :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
        :param loss (str): Loss function to use
        :param learning_rate (float):
        :return model: TensorFlow model
        """

    # # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    #model.add(tf.keras.layers.BatchNormalization())


    #model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(24, kernel_initializer=tf.keras.initializers.HeUniform(), activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    tf.keras.layers.Dropout(0.7)


    model.add(tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    tf.keras.layers.Dropout(0.5)

    model.add(tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
    tf.keras.layers.Dropout(0.3)

    model.add(tf.keras.layers.Dense(NUM_LABELS, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=LEARNING_RATE, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
    #optimiser = tf.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # , momentum=0.8, name='SGD', nesterov=True)
    #optimiser = tf.keras.optimizers.Adadelta(learning_rate=0.01, rho=0.95, epsilon=1e-07, name='Adadelta')

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    summary(model)

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
                        #,callbacks=[earlystop_callback]
                        , callbacks=[learning_rate_scheduler]
                        )
    return history

def schedule(epoch):
    if epoch < 700:
        return 0.001
    else:
        return 0.001

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

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    model = build_model(input_shape, LEARNING_RATE=lu.LEARNING_RATE, NUM_LABELS=lu.NUM_LABELS, PATIENCE=lu.PATIENCE)

    # train network
    history = train(model, EPOCHS, lu.BATCH_SIZE, lu.PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # Evaluating the model on the training and testing set
    # training_loss, training_acc = model.evaluate(X_train, y_train, verbose=0)
    # print("\nTraining loss: {}, training accuracy: {}".format(training_loss, 100*training_acc))

    # evaluate network on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=lu.BATCH_SIZE)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()





