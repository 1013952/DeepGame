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
    def __init__(self, n_heads=64):
        self.model = Model()
        self.n_heads = n_heads
        print('Using attention network')

    # To train a neural network.
    def train_network(self, n_type="baby", data_set_name='cifar10'):
        self.data_set_name = data_set_name
        assure_path_exists("%s_pic/" %self.data_set_name)

        # Load the correct dataset
        if self.data_set_name == 'mnist':
            batch_size = 128
            self.num_classes = 10
            epochs = 50
            img_rows, img_cols = 28, 28
            data_augmentation = False

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        elif self.data_set_name == 'cifar10':
            batch_size = 32
            self.num_classes = 10
            epochs = 200
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

            y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        elif self.data_set_name == 'gtsrb':
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
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10' or 'gtsrb'." % self.data_set_name)



        # Choose correct network structure
        if n_type == 'residual':
            model, mask_model = AttentionResNetCifar10(n_classes=10)
            model.compile(keras.optimizers.Adam(lr=0.0001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
        elif n_type == 'standard':
            print("Using standard attention model.")
            model, mask_model = self.standardmodel(input_shape)
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

                lr_reducer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10, min_lr=10e-7, min_delta=0.0001, verbose=1)
                early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1)
                callbacks= [lr_reducer, early_stopper]

                datagen.fit(x_train)
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    callbacks = callbacks,
                                    workers=4)

        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        model.summary()

        self.model = model
        self.mask_model = mask_model
        self.save_network()


    # Generate network model for standard attention network
    def standardmodel(self, input_shape):
        self_input = Input(shape = input_shape)
        output = Conv2D(self.n_heads, (3, 3), activation='relu')(self_input)
        output = Conv2D(self.n_heads, (3, 3), activation='relu')(output)    
        # output =  MaxPooling2D((2, 2))(output)

        output_soft_mask = Conv2D(self.n_heads, (3, 3))(self_input)
        output_soft_mask = Conv2D(self.n_heads, (3, 3))(output_soft_mask)
        # output_soft_mask = Activation('sigmoid')(output_soft_mask)
        # output_soft_mask = MaxPooling2D((2, 2))(output_soft_mask)

        output_soft_mask = Permute((3, 1, 2))(output_soft_mask)
        output_soft_mask = Reshape((self.n_heads, 24*24))(output_soft_mask)
        output_soft_mask = Activation('softmax')(output_soft_mask)
        output_soft_mask = Reshape((self.n_heads, 24, 24))(output_soft_mask)
        output_soft_mask = Permute((2, 3, 1))(output_soft_mask)


        output = Multiply()([output, output_soft_mask])
        output = MaxPooling2D((2, 2))(output)
        output = Conv2D(2*self.n_heads, (3,3), activation = 'relu')(output)
        output = Conv2D(2*self.n_heads, (3,3), activation = 'relu')(output)
        output = MaxPooling2D((2, 2))(output) 
        output = Flatten()(output)
        output = Dense(256, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(self.num_classes, activation='softmax')(output)

        model = Model(self_input, output)

        # Arbitrary choice for optimizer
        #TODO figure this out
        # prepare usefull callbacks

        model.compile(keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        mask_model = Model(self_input, output_soft_mask)
        mask_model.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=0.001),
                        metrics=['accuracy'])
        return model, mask_model


    # Generate network model for baby attention network
    def babymodel(self, input_shape):
        self.n_heads = 8

        self_input = Input(shape = input_shape)
        output = Conv2D(self.n_heads, (3, 3), activation='relu')(self_input)

        output_soft_mask = Conv2D(self.n_heads, (3, 3))(self_input)
        # output_soft_mask = Permute((3, 1, 2))(output_soft_mask)
        # output_soft_mask = Reshape((self.n_heads, 26*26))(output_soft_mask)
        output_soft_mask = Activation('sigmoid')(output_soft_mask)
        # output_soft_mask = Reshape((self.n_heads, 26, 26))(output_soft_mask)
        # output_soft_mask = Permute((2, 3, 1))(output_soft_mask)


        output = Multiply()([output, output_soft_mask])
        output = MaxPooling2D((2, 2))(output)
        # output = Conv2D(self.n_heads, (3,3), activation = 'relu')(output)
        # output = MaxPooling2D((2, 2))(output) 
        output = Flatten()(output)
        # output = Dense(16, activation='relu')(output)
        # output = Dropout(0.5)(output)
        output = Dense(self.num_classes, activation='softmax')(output)

        model = Model(self_input, output)

        # Arbitrary choice for optimizer
        #TODO figure this out
        # prepare usefull callbacks

        model.compile(keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        mask_model = Model(self_input, output_soft_mask)
        mask_model.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=0.001),
                        metrics=['accuracy'])
        return model, mask_model


    # To save the neural network to disk.
    def save_network(self):
        if self.data_set_name in ['mnist', 'cifar10', 'gtsrb']:
            self.model.save('models/attention_'+ self.data_set_name + ".h5")
            self.mask_model.save('models/attention_'+ self.data_set_name + '_mask.h5')
            print("Neural network saved to disk.")
        else:
            print("save_network: Unsupported dataset.")


    # To load a neural network from disk.
    def load_network(self, path = None, data_set_name = 'cifar10'):
        self.data_set_name = data_set_name
        assure_path_exists("%s_pic/" %self.data_set_name)

        if path is None:
            path = 'models/attention_' + self.data_set_name

        if self.data_set_name in ['mnist', 'cifar10', 'gtsrb']:
            self.model = load_model(path + ".h5")
            self.mask_model = load_model(path + '_mask.h5')
            print("Neural network loaded from disk.")
        else:
            print("load_network: Unsupported dataset.")

    def get_partition_model(self):
        return self.mask_model
