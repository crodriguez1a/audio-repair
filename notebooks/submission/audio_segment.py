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
        """
        Mel-frequency cepstral coefficients (MFCCs)
        are coefficients that collectively make up an MFC[1].
        They are derived from a type of cepstral representation
        of the audio clip (a nonlinear "spectrum-of-a-spectrum").
        The difference between the cepstrum and the mel-frequency
        cepstrum is that in the MFC, the frequency bands are
        equally spaced on the mel scale, which approximates
        the human auditory system's response more closely
        than the linearly-spaced frequency bands used in
        the normal cepstrum.
        Source: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        """
        y:np.ndarray = self._audio
        sr:int = self._sr
        return librosa.feature.mfcc(y=y, sr=sr)

    def mfcc_to_audio(self, M:np.ndarray) -> np.ndarray:
        sr:int = self._sr
        original:np.ndarray = self._audio
        return librosa.feature.inverse.mfcc_to_audio(M, sr=sr, length=len(original))

    @property
    def mel_features(self) -> np.ndarray:
        """
        The mel scale, named by Stevens, Volkmann, and Newman in 1937,[1]
        is a perceptual scale of pitches judged by listeners to be equal
        in distance from one another.
        Source: https://en.wikipedia.org/wiki/Mel_scale
        """
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
    mel:np.ndarray = segment.mel_features
    mel_2_audio:np.ndarray = segment.mel_to_audio(mel)

    # TODO: move to unit test
    print('segment.audio.shape', segment.audio.shape)
    print('mel.shape', mel.shape)
    print('mel_2_audio.shape', mel_2_audio.shape)
