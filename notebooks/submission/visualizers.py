import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def vis_outliers(x:np.ndarray, outlier_scores:float, quantile:float=0.9):
    threshold = pd.Series(outlier_scores).quantile(quantile)
    outliers = np.where(outlier_scores > threshold)[0]

    plt.scatter(x=x[:,0], y=x[:,1], s=50, linewidth=0, c='orange', alpha=0.5)
    plt.scatter(x=x[outliers][:,0], y=x[outliers][:,1], s=50, linewidth=0, c='blue', alpha=0.5)

    # Pair-wise Scatter Plots
    df = pd.DataFrame(x)
    pp = sns.pairplot(df, size=1.8, aspect=1.8,
                      plot_kws=dict(edgecolor="k", linewidth=0.5),
                      diag_kind="kde", diag_kws=dict(shade=True))
