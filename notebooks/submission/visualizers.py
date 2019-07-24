import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display as libro_display
from scipy.stats import multivariate_normal
import math

def vis_norm_distribution(x:np.ndarray, mean:float=2.5, cov:float=0.5):
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
    plt.plot(x, multivariate_normal.pdf(x, mean=mean, cov=cov))
    plt.show()

def vis_outliers(x:np.ndarray,
                 outliers:float,
                 title:str='',
                 xlabel:str='',
                 ylabel:str='',):

    plt.title(title)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.scatter(x=x[:,0], y=x[:,1], s=50, linewidth=0, c='orange', alpha=0.5)
    plt.scatter(x=x[outliers][:,0], y=x[outliers][:,1], s=50, linewidth=0, c='blue', alpha=0.5)

def vis_spec(spec:np.ndarray,
             figsize:tuple=(10,4),
             title:str='Mel spectrogram',
             y_axis:str='mel',
             x_axis:str='time',
             fmax:int=8000,
             sr:int=44100,):

    plt.figure(figsize=figsize)
    libro_display.specshow(librosa.power_to_db(spec, ref=np.max),
                                               y_axis=y_axis,
                                               fmax=fmax,
                                               x_axis=x_axis,
                                               sr=sr,)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
