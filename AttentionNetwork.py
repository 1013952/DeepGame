"""
Construct an AttentionNetwork class inheriting
from NeuralNetwork to handle the attention module

Author: Denitsa Markova
Email: denitsa.markova@cs.ox.ac.uk
"""

import cv2
import copy
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dot
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as Image
from keras import backend as K
from matplotlib import pyplot as plt

from basics import assure_path_exists
from DataSet import *


# Define a Attention Network class.
class AttentionNetwork(NeuralNetwork):
    # To train a neural network.
    def train_network(self):
        # Train an mnist model.
        if self.data_set == 'mnist':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols = 28, 28

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

            """
            Define network structure here 
            """
            # Only simple structure here so far
            hidden_size = 200 #TODO figure this out
            self_input = Input(shape=input_shape)
            query_transf = Dense(hidden_size, activation="linear")(self_input)
            key_transf = Dense(hidden_size, activation="linear")(self_input)
            value_transf = Dense(hidden_size, activation="linear")(self_input)
            attention = Dot(axes=[2, 2])([query_transf, key_transf])
            attention = Activation('softmax')(attention)
            context = Dot(axes=[2, 1])([attention, value_transf])

            model = Model(self_input, context)
            model.add(Dense(256, activation = 'relu'))
            model.add(Dense(num_classes, activation='softmax'))

            # Arbitrary choice for optimizer
            #TODO figure this out
            model.compile(loss='categorical_crossentropy',
                          optimizer= keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

            self.model = model

        # Train a cifar10 model.
        elif self.data_set == 'cifar10':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols, img_chls = 32, 32, 3
            data_augmentation = True

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
            input_shape = (img_rows, img_cols, img_chls)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

            """
            Define network structure here 
            """
            # Only simple structure here so far
            hidden_size = 200 #TODO figure this out
            num_heads = 10 #TODO configure

            self_input = Input(shape=input_shape)
            key_trasf = []
            value_transf = []
            context = []
            for i in range(num_heads):
                query_transf_t = Dense(hidden_size, activation="linear")(self_input)
                key_transf_t = Dense(hidden_size, activation="linear")(self_input)
                value_transf_t = Dense(hidden_size, activation="linear")(self_input)
                attention_t = Dot(axes=[2, 2])([query_transf, key_transf])
                attention_t = Activation('softmax')(attention)
                context_t = Dot(axes=[2, 1])([attention, value_transf])

                key_transf.append(key_transf_t)
                query_transf.append(query_transf_t)
                value_transf.append(value_transf_t)
                context.append(context_t)

            context = Concat(context)
            model = Model(self_input, context)
            model.add(Dense(256, activation = 'relu'))
            model.add(Dense(num_classes, activation='softmax'))

            # Arbitrary choice for optimizer
            #TODO figure this out
            model.compile(loss='categorical_crossentropy',
                          optimizer= keras.optimizers.Adadelta(),
                          metrics=['accuracy'])


            if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
            else:
                print("Using real-time data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=False)

                datagen.fit(x_train)
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)

            scores = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", scores[0])
            print("Test accuracy:", scores[1])

            self.model = model

            # Train a gtsrb model.
        elif self.data_set == 'gtsrb':
            batch_size = 128
            num_classes = 43
            epochs = 50
            img_rows, img_cols, img_chls = 48, 48, 3
            data_augmentation = True

            train = DataSet('gtsrb', 'training')
            x_train, y_train = train.x, train.y
            test = DataSet('gtsrb', 'test')
            x_test, y_test = test.x, test.y
            input_shape = (img_rows, img_cols, img_chls)

            """
            Define network structure here 
            """
            # Only simple structure here so far
            hidden_size = 200 #TODO figure this out
            self_input = Input(shape=input_shape)
            query_transf = Dense(hidden_size, activation="linear")(self_input)
            key_transf = Dense(hidden_size, activation="linear")(self_input)
            value_transf = Dense(hidden_size, activation="linear")(self_input)
            attention = Dot(axes=[2, 2])([query_transf, key_transf])
            attention = Activation('softmax')(attention)
            context = Dot(axes=[2, 1])([attention, value_transf])

            model = Model(self_input, context)
            model.add(Dense(256, activation = 'relu'))
            model.add(Dense(num_classes, activation='softmax'))

            # Arbitrary choice for optimizer
            #TODO figure this out
            model.compile(loss='categorical_crossentropy',
                          optimizer= keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
            else:
                print("Using real-time data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=False)

                datagen.fit(x_train)
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)

            scores = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", scores[0])
            print("Test accuracy:", scores[1])

            self.model = model

        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10' or 'gtsrb'." % self.data_set)
        self.save_network()

    # To save the neural network to disk.
    def save_network(self):
        if self.data_set in ['mnist', 'cifar10', 'gtsrb']:
            self.model.save('models/attention_'+ self.data_set + ".h5")
            print("Neural network saved to disk.")
        else:
            print("save_network: Unsupported dataset.")


    # To load a neural network from disk.
    def load_network(self):
        if self.data_set in ['mnist', 'cifar10', 'gtsrb']:
            self.model = load_model('models/attention_'+ self.data_set + ".h5")
            print("Neural network loaded from disk.")
        else:
            print("load_network: Unsupported dataset.")
