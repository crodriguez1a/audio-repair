import numpy as np
import librosa

class AudioSegment:
    __slots__ = ['_path', '_audio', '_sr']

    def __init__(self, path:str):
        self._path = path
        audio, sample_rate = librosa.load(path)
        self._audio = audio
        self._sr = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def audio(self) -> int:
        return self._audio

    @property
    def mfcc_features(self) -> np.ndarray:
        y:np.ndarray = self._audio
        sr:int = self._sr
        return librosa.feature.mfcc(y=y, sr=sr)

    def mfcc_to_audio(self, M:np.ndarray) -> np.ndarray:
        sr:int = self._sr
        original:np.ndarray = self._audio
        return librosa.feature.inverse.mfcc_to_audio(M, sr=sr, length=len(original))

    @property
    def mel_features(self) -> np.ndarray:
        y:np.ndarray = self._audio
        sr:int = self._sr
        return librosa.feature.melspectrogram(y=y, sr=sr)

    def mel_to_audio(self, M:np.ndarray) -> np.ndarray:
        sr:int = self._sr
        original:np.ndarray = self._audio
        return librosa.feature.inverse.mel_to_audio(M, sr=sr, length=len(original))

if __name__ == '__main__':
    samp_path = 'data/audio_sources/wav/damaged/izotope_rustle_samp.wav'
    segment:AudioSegment = AudioSegment(samp_path)
    feat:np.ndarray = segment.mel_features
    feat_2_audio:np.ndarray = segment.mel_to_audio(feat)

    # TODO: move to unit test
    print('segment._audio', segment._audio.shape)
    print('feat.shape', feat.shape)
    print('feat_2_audio.shape', feat_2_audio.shape)
