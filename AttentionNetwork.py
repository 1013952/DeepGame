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
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Reshape, Permute,  Dropout, Flatten, Conv2D, MaxPooling2D, Input, Multiply, Dot, Activation, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as Image
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from matplotlib import pyplot as plt

from basics import assure_path_exists
from DataSet import *
from NeuralNetwork import *

from residual_models import AttentionResNetCifar10

# Define a Attention Network class.
class AttentionNetwork(NeuralNetwork):
    def __init__(self, n_heads=32, data_set='cifar10'):
        self.data_set = data_set
        self.model = Model()
        assure_path_exists("%s_pic/" %self.data_set)
        self.n_heads = n_heads

    # To train a neural network.
    def train_network(self, n_type="baby"):
        num_classes = 10
        # Load the correct dataset
        if self.data_set == 'mnist':
            batch_size = 128
            self.num_classes = 10
            epochs = 50
            img_rows, img_cols = 28, 28
            data_augmentation = False

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            # x_train /= 255
            # x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

        elif self.data_set == 'cifar10':
            batch_size = 128
            self.num_classes = 10
            epochs = 75
            img_rows, img_cols, img_chls = 32, 32, 3
            data_augmentation = True

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
            # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
            input_shape = (img_rows, img_cols, img_chls)

            # x_train = x_train.astype('float32')
            # x_test = x_test.astype('float32')
            # x_train /= 255
            # x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

        elif self.data_set == 'gtsrb':
            batch_size = 128
            self.num_classes = 43
            epochs = 50
            img_rows, img_cols, img_chls = 48, 48, 3
            data_augmentation = True

            train = DataSet('gtsrb', 'training')
            x_train, y_train = train.x, train.y
            test = DataSet('gtsrb', 'test')
            x_test, y_test = test.x, test.y
            input_shape = (img_rows, img_cols, img_chls)
        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10' or 'gtsrb'." % self.data_set)



        # Choose correct network structure
        if n_type == 'residual':
            model, mask_model = AttentionResNetCifar10(n_classes=10)
            model.compile(keras.optimizers.Adam(lr=0.0001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
        elif n_type == 'transformer':
            print("Transformer model not yet implemented.")
        else:
            print("Using baby attention model.")
            model, mask_model = self.babymodel(input_shape)

        # Train and score model
        if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
        else:
                print("Using real-time data augmentation.")
                # define generators for training and validation data
                train_datagen = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

                test_datagen  = ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True)


                # compute quantities required for featurewise normalization
                # (std, mean, and principal components if ZCA whitening is applied)
                train_datagen.fit(x_train)
                test_datagen.fit(x_test)


                # prepare usefull callbacks
                lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=7, min_lr=10e-7, epsilon=0.01, verbose=1)
                early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1)
                callbacks= [lr_reducer, early_stopper]

                batch_size = 32
                model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train)//batch_size, epochs=200,
                    validation_data=test_datagen.flow(x_test, y_test, batch_size=batch_size), 
                    validation_steps=len(x_test)//batch_size,
                    callbacks=callbacks, initial_epoch=0,
                    workers=4)

        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        self.model = model
        self.mask_model = mask_model
        self.save_network()


    # Generate network model for baby attention network
    def babymodel(self, input_shape):
        self_input = Input(shape = input_shape)
        output = Conv2D(self.n_heads, (3, 3), activation='relu')(self_input)
        output = Conv2D(self.n_heads, (3, 3), activation='relu')(output)
        output =  MaxPooling2D((2, 2))(output)

        output_soft_mask = Conv2D(self.n_heads, (3, 3))(self_input)
        output_soft_mask = Conv2D(self.n_heads, (3, 3))(output_soft_mask)
        output_soft_mask = Activation('sigmoid')(output_soft_mask)
        output_soft_mask = MaxPooling2D((2, 2))(output_soft_mask)

        # output_soft_mask = Permute((3, 1, 2))(output_soft_mask)
        # output_soft_mask = Reshape((self.n_heads, 14*14))(output_soft_mask)
        # output_soft_mask = Activation('softmax')(output_soft_mask)
        # output_soft_mask = Reshape((self.n_heads, 14, 14))(output_soft_mask)
        # output_soft_mask = Permute((2, 3, 1))(output_soft_mask)


        output = Multiply()([output, output_soft_mask])
        output = Conv2D(self.n_heads, (3,3), activation = 'relu')(output)
        output = Conv2D(self.n_heads, (3,3), activation = 'relu')(output)
        output = MaxPooling2D((2, 2))(output) 
        output = Flatten()(output)

        output = Dense(256, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(self.num_classes, activation='softmax')(output)

        model = Model(self_input, output)

        # Arbitrary choice for optimizer
        #TODO figure this out
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=0.0001),
                          metrics=['accuracy'])

        mask_model = Model(self_input, output_soft_mask)
        mask_model.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=0.0001),
                        metrics=['accuracy'])
        return model, mask_model


    # To save the neural network to disk.
    def save_network(self):
        if self.data_set in ['mnist', 'cifar10', 'gtsrb']:
            self.model.save('models/attention_'+ self.data_set + ".h5")
            self.mask_model.save('models/attention_'+ self.data_set + '_mask.h5')
            print("Neural network saved to disk.")
        else:
            print("save_network: Unsupported dataset.")


    # To load a neural network from disk.
    def load_network(self):
        if self.data_set in ['mnist', 'cifar10', 'gtsrb']:
            self.model = load_model('models/attention_'+ self.data_set + ".h5")
            self.mask_model = load_model('models/attention_' + self.data_set + '_mask.h5')
            print("Neural network loaded from disk.")
        else:
            print("load_network: Unsupported dataset.")

    def get_partition_model(self):
        return self.mask_model
