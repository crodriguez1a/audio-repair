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

class AudioSegment:
    def __init__(self, tf_sess, params={}):
        # TODO hyperparams
        self.tf_sess = tf_sess

    def analyze_wav(self, path:str) -> tuple:
        """
        Decode audio, represent each channel
        as fourier features
        """
        audio_binary = tf.io.read_file(path)

        # Initialize tf decoder
        decoder = audio_ops.decode_wav(audio_binary,
                                       desired_channels=2)

        # TODO: requires a session to evaluate
        self.sample_rate:int = decoder.sample_rate.eval(session=self.tf_sess)
        audio:np.ndarray = decoder.audio.eval(session=self.tf_sess)

        # represent as numpy array
        audio:np.ndarray = np.array(audio)

        # represent each channel as discrete fourier features
        wv_ch1:np.ndarray = rfft(audio[:,0])
        wv_ch2:np.ndarray = rfft(audio[:,1])

        return ([wv_ch1], [wv_ch2])

    def autoencoder(self, input_size) -> tuple:
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
        hidden_1_units = int(input_size/4)
        hidden_2_units = int(input_size/6)
        hidden_3_units = int(input_size/8)

        # input audio
        X = Input(shape=(inputs,))

        # TODO: Decide on regularization
        l2_regularizer = regularizers.l2(l2)

        # encoded and decoded layer for the autoencoder
        encoded = Dense(units=inputs,
                        activation='elu',
                        kernel_regularizer=l2_regularizer,)(X)

        encoded = Dense(units=hidden_1_units, activation='elu')(encoded)
        encoded = Dense(units=hidden_2_units, activation='relu')(encoded)

        decoded = Dense(units=hidden_1_units, activation='elu')(encoded)
        decoded = Dense(units=hidden_2_units, activation='elu')(decoded)
        decoded = Dense(units=input_size, activation='sigmoid')(decoded)

        # Building autoencoder
        autoencoder = Model(X, decoded)

        #extracting encoder
        encoder = Model(X, encoded)

        return (autoencoder, encoder)

    @staticmethod
    def reshape_channel(channel:list, inputs:int) -> np.ndarray:
        """
        TODO: annotate
        """
        return np.array(channel[:inputs]).reshape(1, inputs)

    def reshape_channels(self, wav_arr_ch1:list, wav_arr_ch2:list, inputs:int) -> tuple:
        """
        TODO: annotate
        """
        ch1:list = [self.reshape_channel(wav_arr_ch1, inputs)]
        ch2:list = [self.reshape_channel(wav_arr_ch2, inputs)]

        return (np.array(ch1), np.array(ch2))

    @staticmethod
    def synthesize_audio(ch1, ch2) -> np.ndarray:
        """
        Re-combine channels,
        inverse discrete Fourier transform
        """
        ch1 = irfft(np.hstack(np.hstack(ch1)))
        ch2 = irfft(np.hstack(np.hstack(ch2)))

        audio_arr = np.hstack(np.array((ch1, ch2)).T)

        return audio_arr

    def write_audio(self, audio_arr, path:str='out.wav'):
        """
        Write audio array to file
        """
        cols = 2
        rows = math.floor(len(audio_arr)/cols)
        audio_to_encode = audio_arr.reshape(rows, cols)

        wav_encoder = ffmpeg.encode_audio(audio_to_encode,
                                          file_format='wav',
                                          samples_per_second=self.sample_rate)

        f = open(path, 'wb')
        # TODO figure this out wihout a tf session
        wav_file = self.tf_sess.run(wav_encoder)
        f.write(wav_file)

if __name__ == '__main__':
    noisy_path = 'data/audio_sources/wav/damaged/izotope_rustle_samp.wav'
    gt_path = 'data/audio_sources/wav/gt/izotope_rustle.wav'

    with tf.compat.v1.Session() as sess:
        segment = AudioSegment(sess)

        # TODO re-visit nested arrays
        wv_ch1, wv_ch2 = segment.analyze_wav(noisy_path)
        input_size:int = wv_ch1[0].size
        print(input_size)
        print(wv_ch1[0].shape, wv_ch1[0].size, wv_ch2[0].shape, wv_ch2[0].size)

        # TODO re-visit nested arrays
        ch1_song, ch2_song = segment.reshape_channels(wv_ch1, wv_ch2, inputs=input_size)
        print(ch1_song.shape, ch2_song.shape,)

        autoencoder, encoder = segment.autoencoder(input_size)

        autoencoder.summary()
        encoder.summary()

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # re-visit/explain shape
        total_songs = np.hstack([ch1_song, ch2_song])
        x_batch = total_songs[0]

        wav_output = segment.synthesize_audio(ch1_song, ch2_song)

        # training
        # what is train test then?
        # autoencoder.fit(x_batch, x_batch,
        #                 epochs=5,# 50,
        #                 batch_size=256,
        #                 shuffle=True,
        #                 validation_data=(x_batch, x_batch)) # TODO validation set
        #
        # encoded_audio = encoder.predict(x_batch)
        #
        #
        # predicted = autoencoder.predict(x_batch)
