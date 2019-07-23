import numpy as np
import librosa

class AudioSegment:
    __slots__ = ['_path', '_audio', '_sr']

    def __init__(self,
                 path:str,
                 preserve_sampling:bool=True,
                 duration:float=1.,):

        self._path = path
        sr:int = None if preserve_sampling else 22050 # librosa default
        audio, sample_rate = librosa.load(path, sr=None, duration=duration,)
        self._audio = audio
        self._sr = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def audio(self) -> int:
        return self._audio

    @property
    def mfcc_features(self, htk:bool=True) -> np.ndarray:
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
        # HTK is a toolkit for research in automatic speech recognition (http://htk.eng.cam.ac.uk/docs/faq.shtml)
        return librosa.feature.mfcc(y=y, sr=sr, htk=htk)

    def mfcc_to_audio(self, M:np.ndarray) -> np.ndarray:
        sr:int = self._sr
        original:np.ndarray = self._audio
        return librosa.feature.inverse.mfcc_to_audio(M, sr=sr, length=len(original))

    def _to_mel(self, params:dict={}) -> np.ndarray:
        # abstract librosa to provide additional params when needed
        return librosa.feature.melspectrogram(**params)

    @property
    def mel_features(self) -> np.ndarray:
        """
        Magnitude spectrogram S is first computed, and then mapped onto the Mel Scale

        The mel scale, named by Stevens, Volkmann, and Newman in 1937,[1]
        is a perceptual scale of pitches judged by listeners to be equal
        in distance from one another.

        Sources:
        https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram
        https://en.wikipedia.org/wiki/Mel_scale
        """
        y:np.ndarray = self._audio
        sr:int = self._sr
        return self._to_mel({'y':y, 'sr':sr})

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
