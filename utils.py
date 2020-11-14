#! usr/bin/python
"""
file:       utils.py
date:       13-04-2019
author:     Bart Stam
description:
    Utility functions for the CNN, like models and plotting functions
"""
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from keras.applications import VGG19, MobileNetV2, NASNetMobile
import os


def graph_history(history):
    """ Plots a graph of the train- and test accuracies

    Args:
        history:    A keras history object containing the accuracies
    
    """
    # Summarizing history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarizing history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Purples):
    """ Plots a confusion matrix of classification

    Args:
        y_true:             The true target variables (one-hot)
        y_pred:             The predicted target variables (one-hot)
        classes:            The list of class names corresponding to their indices
        normalize (opt):    Whether or not to normalize each row
        title (opt):        A title for the plot
        cmap (opt):         A matplotlib color map
    
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Calculate chart area size
    leftmargin = 1  # inches
    rightmargin = 1  # inches
    categorysize = 0.45  # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)

    fig = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    fig.subplots_adjust(left=leftmargin / figwidth, right=1 - rightmargin / figwidth, top=.9, bottom=.8)
    # fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def paths_list_from_directory(directory):
    """ Returns a list of file paths in the given directory

    Args:
        directory:  The folder with the data
    Returns:
        A list of file paths

    """
    file_paths = []
    for (dirpath, _, filenames) in os.walk(directory):
        file_paths.extend([os.path.join(dirpath, filename) for filename in filenames])

    return file_paths


def compute_zero_rule(labels):
    """ Computes the zero rule (0R) as accuracy baseline
            (DEPRECATED)
    Args:
        labels: The list of labels
    Returns:
        A value between 0-1 representing the accuracy of an untrained model

    """
    histogram = dict()
    for label in labels:
        key = str(np.where(label == 1.0)[0][0])
        if histogram.__contains__(key):
            histogram[key] += 1
        else:
            histogram[key] = 1
    highest_freq = 0
    for key, val in histogram.items():
        if val > highest_freq:
            highest_freq = val
    return highest_freq / len(labels)


def basic_model(input_shape, num_classes):
    """ Returns a simple Keras CNN model
            source: <link-to-source>

    Args:
        input_shape: A (width x height x channel) tuple of input dimensions
        num_classes: The number of classes (int)
    Returns:
        The Keras model

    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def alex_net(input_shape, num_classes):
    """ Returns a single-thread version of the AlexNet architecture

    Args:
        input_shape: A (width x height x channel) tuple of input dimensions
        num_classes: The number of classes (int)
    Returns:
        The Keras model

    """
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4),
                     activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='valid'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.1)))
    return model


def vgg19(input_shape, num_classes):
    """ Prepares a CNN model with the VGG19 architecture
            source: https://keras.io/applications
    
    Args:
        input_shape:    The dimensions of the input images (w x h x c)
        num_classes:    The number of classes
    Retruns:
        The VGG19 Keras model
    
    """
    base_model = VGG19(include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def mobilenet_v2(input_shape, num_classes):
    """ Prepares a CNN model with the MobileNet v2 architecture
            source: https://keras.io/applications
    
    Args:
        input_shape:    The dimensions of the input images (w x h x c)
        num_classes:    The number of classes
    Retruns:
        The MobileNet v2 Keras model
    
    """
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def nasnet_mobile(input_shape, num_classes):
    """ Prepares a CNN model with the NasNet Mobile architecture
            source: https://keras.io/applications
    
    Args:
        input_shape:    The dimensions of the input images (w x h x c)
        num_classes:    The number of classes
    Retruns:
        The NasNet Mobile Keras model
    
    """
    base_model = NASNetMobile(include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.1))(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
