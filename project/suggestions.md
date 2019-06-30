# Report

Suggested: Please, in your final report, you might link your intro section with academic references (in the reference section, doing like this ... some reference text [1]... where [1] is the reference number).

Suggested: remember to use a business language (less technical) to explain the data findings and the model results to your reader.

# Scoring + Metrics

I do think that using the silhouette score to measure your unsupervised learning/clustering is probably a good approach. However, I'd also suggest using **a measurement of error** to compare the altered audio with the ground truth audio. Since this is essentially a regression task, you should use a regression metric to compare the predicted audio with the ground truth. **RMSE** might be a good option for tuning your KNN model as well.

# Learning

For the supervised learning part of your project, the XGBoost and LightGBM models could be good supervised learning approaches to try here.

One approach that you might consider during the clustering is to detect the anomalies, but not remove them or replace them with null values. You could then train the KNN model by using the sections of data that are identified as anomalous as the input features (where the corresponding ground truth data is the labels).

# Noise

I think you have a good Solution Statement here, as it is clear that you have thought a lot about what your approach is to this problem.

If you have enough data, you might also even look into using a [denoising autoencoder](https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2) approach.

I like the Anomaly Detection with DBSCAN idea. You might also even look into using an [Isolation Forest](https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e) as well.

The AMELIA library might also be something to look into as well for replace missing data points.

# Imputation

[Amelia](https://cran.r-project.org/web/packages/Amelia/Amelia.pdf)
