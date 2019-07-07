import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import librosa

def vis_outliers(x:np.ndarray, outliers:float):
    plt.scatter(x=x[:,0], y=x[:,1], s=50, linewidth=0, c='orange', alpha=0.5)
    plt.scatter(x=x[outliers][:,0], y=x[outliers][:,1], s=50, linewidth=0, c='blue', alpha=0.5)

    # Pair-wise Scatter Plots
    # df = pd.DataFrame(x)
    # pp = sns.pairplot(df, size=1.8, aspect=1.8,
    #                   plot_kws=dict(edgecolor="k", linewidth=0.5),
    #                   diag_kind="kde", diag_kws=dict(shade=True))

def vis_spec(spec:np.ndarray,
             figsize:tuple=(10,4),
             title:str='Mel spectrogram',
             y_axis:str='mel',
             x_axis:str='time',
             fmax:int=8000,):

    plt.figure(figsize=figsize)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max),
                                                 y_axis=y_axis,
                                                 fmax=fmax,
                                                 x_axis=x_axis)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
