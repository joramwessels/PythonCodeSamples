#! usr/bin/python
"""
file:       CNN.py
date:       12-04-2019
author:     Joram Wessels
description:
    A wrapper class that builds and trains a Keras model
"""
from keras import callbacks
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import nasnet
import utils
import numpy as np


class CNN:

    def __init__(self, train_dir='TrainImages', test_dir='TestImages', num_classes=40,
                 target_size=(224, 224), batch_size=32, summary=False):
        """ The CNN class trains-, and keeps track of-, a CNN model

        Args:
            train_dir:   Directory containing the images to train the model on
            test_dir:    Directory containing the images to test the model on
            target_size: The dimensions of the generated train and test images
            batch_size:  Number of images the data generator yields per batch

        """

        input_shape = (target_size[0], target_size[1], 3)
        self.model = utils.alex_net(input_shape, num_classes)

        self.train_batches = ImageDataGenerator().flow_from_directory(
            train_dir, target_size=target_size, batch_size=batch_size
        )
        self.test_batches = ImageDataGenerator().flow_from_directory(
            test_dir, target_size=target_size, batch_size=batch_size, shuffle=False
        )

        if summary:
            print(self.model.summary())

    def train(self, epochs=10):
        """ Trains the CNN model using the data provided in the constructor

        Returns:
            The results of the last evaluation
        
        """

        custom_lr = LearningRateScheduler(custom_learning_rate, 0)
        csv_logger = callbacks.CSVLogger('CSVLogger_results.log')
        checkpoint = callbacks.ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True)

        self.model.compile(SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False), loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit_generator(self.train_batches, epochs=epochs,
                                           validation_data=self.test_batches,
                                           callbacks=[csv_logger, custom_lr])

        return history
    
    def classify(self, image):
        """ Predicts the label of a new image using the trained model

        Args:
            image:  A (width x height x channel) numpy array
        Returns:
            The predicted class index
        
        """
        if not self.model:
            return -1
        return self.model.predict(image)

    def confusion_matrix(self, normalize=False):
        """ Plots a confusion matrix of the model
        """
        predictions = self.model.predict_generator(self.test_batches)
        predictions = np.argmax(predictions, axis=1)
        ground_truth = self.test_batches.classes
        classes = [*self.test_batches.class_indices]
        utils.plot_confusion_matrix(ground_truth, predictions, classes, normalize=normalize)

    def visualize_model(self):
        """ Creates a schematic image of the network architecture
        """
        plot_model(self.model, show_shapes=True)


def custom_learning_rate(epoch, lr):
    """ Function to be passed to LearningRateScheduler

    Args:
        epoch:  The current epoch
        lr:     The current learning rate
    Returns:
        The new learning rate
    
    """
    initial_lrate = 0.001
    k = 0.1
    return initial_lrate * np.exp(-k * epoch)
