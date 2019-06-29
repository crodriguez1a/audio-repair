# Inspired by: https://musicinformationretrieval.com/kmeans.html#Feature-Extraction

import sys

import numpy as np
import pandas as pd
import scipy

import librosa
import mira_eval
from impyute.imputation.cs import fast_knn

#Increase the recursion limit of the OS
sys.setrecursionlimit(100000)

def detect_onsets():
    pass

def split_segments():
    pass

def concat_segments():
    pass

class AudioSegment:
    __slot__ = ['_wav', '_sample_rate', '_clusterer', '_imputer']

    def __init__(self, wav, sample_rate, clusterer, imputer):
        self._wav = wav # from librosa.load
        self._sample_rate = sample_rate
        self._clusterer = clusterer
        self._imputer = imputer

    def extract_features(self):
        pass

    def scale_features(self):
        pass

    @property
    def clusterer(self) -> object:
        return self._clusterer

    def analyze_outliers(self):
        # should the outlier be imputed?
        pass

    def replace_outliers(self, replace_with:object=None):
        pass

    @property
    def imputer(self) -> object:
        return self._imputer

    @property
    def imputed_wav(self) -> np.ndarray:
        pass
