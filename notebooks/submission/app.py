# Inspired by: https://musicinformationretrieval.com/kmeans.html#Feature-Extraction

import sys

import numpy as np
import pandas as pd
import scipy

import librosa
import mir_eval
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

    def __init__(self, wav, ground_truth_wav, sample_rate, clusterer, imputer):
        self._wav = wav # from librosa.load
        self._sample_rate = sample_rate
        self._clusterer = clusterer # initalized?
        self._imputer = imputer # not intialized
        self._gt_wav = ground_truth_wav

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
        # this is the final output
        pass

class Results:
    __slots__ = ['_damaged', '_ground_truth', '_imputed']

    def __init__(self, damaged, ground_truth, imputed):
        self._damaged = damaged
        self._ground_truth = ground_truth
        self._imputed = imputed

    @property
    def benchmark_score(self) -> float:
        pass

    @property
    def damaged_silhouette_score(self) -> float:
        pass

    @property
    def imputed_silhouette_score(self) -> float:
        pass

    @property
    def ground_truth_silhouette_score(self) -> float:
        pass
