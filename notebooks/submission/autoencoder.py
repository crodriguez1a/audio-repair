import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Dense, UpSampling2D
from keras.models import Model
from keras import backend as K

import numpy as np
import math
from scipy.fftpack import rfft, irfft
from tensorflow.contrib import ffmpeg # TODO upgrade to tensorflow_io: https://github.com/tensorflow/io
from tensorflow.contrib.framework.python.ops import audio_ops

# Inspired by: https://blog.goodaudience.com/using-tensorflow-autoencoders-with-music-f871a76122ba

class AutoEncoder:
    __slots__ = ['_params']

    def __init__(self, hyper_params:dict={}):
        self._params = hyper_params

    def layers(self, input_size, unit_sizes:tuple=(128, 64, 32, 64, 128)) -> tuple:
        """
        De-noising Autoencoder
        # Credit: https://medium.com/datadriveninvestor/deep-autoencoder-using-keras-b77cd3e8be95
        """
        # TODO: move this
        epochs = 1000
        batches = 10
        l2 = 0.0001
        lr = 0.0001

        batch_size = 1 # 50

        inputs = input_size
        sz1, sz2, sz3, sz4, sz5 = unit_sizes

        # input audio
        X = Input(shape=(inputs,))

        # TODO: Decide on regularization
        l2_regularizer = regularizers.l2(l2)

        # encoded and decoded layer for the autoencoder
        encoded = Dense(units=sz1,
                        activation='elu',
                        kernel_regularizer=l2_regularizer,)(X)

        encoded = Dense(units=sz2, activation='elu')(encoded)
        encoded = Dense(units=sz3, activation='elu')(encoded)

        decoded = Dense(units=sz4, activation='elu')(encoded)
        decoded = Dense(units=sz5, activation='elu')(decoded)
        decoded = Dense(units=input_size, activation='sigmoid')(decoded)

        # Building autoencoder
        autoencoder = Model(X, decoded)

        #extracting encoder
        encoder = Model(X, encoded)

        return (autoencoder, encoder)


if __name__ == '__main__':

    from .audio_segment import AudioSegment

    path = 'data/audio_sources/wav/damaged/izotope_rustle_samp.wav'
    segment:AudioSegment = AudioSegment(path)

    mfcc:np.ndarray = segment.mfcc_features
    input_size:int = mfcc.T[:,1].size

    ae:AutoEncoder = AutoEncoder()
    autoencoder, encoder = ae.layers(input_size)

    autoencoder.summary()
    encoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # training
    # TODO: train test validate then?
    autoencoder.fit(mfcc, mfcc,
                    epochs=5,# 50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(mfcc, mfcc)) # TODO validation set

    encoded_audio = encoder.predict(mfcc)

    predicted = autoencoder.predict(mfcc)

    mfcc_2_audio:np.ndarray = segment.mfcc_to_audio(predicted)


    print('segment.audio.shape', segment.audio.shape)
    print('mfcc.shape', mfcc.shape)
    print('encoded_audio.shape', encoded_audio.shape)
    print('predicted.shape', predicted.shape)
    print('mfcc_2_audio.shape', mfcc_2_audio.shape)
