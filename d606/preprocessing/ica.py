from sklearn.decomposition import FastICA
from d606.preprocessing.dataextractor import load_data

numComponents = 25

""" RUNNINGTIME: No problem when the data have run through pca first """

""" http://scikit-learn.org/stable/auto_examples/decomposition/
    plot_ica_blind_source_separation.html

    http://scikit-learn.org/stable/modules/generated/sklearn.
    decomposition.FastICA.html#sklearn.decomposition.FastICA

    Fast algorithm for Independent Component Analysis
    Matrix needs to be en n * x matrix, where n is
    samples, and m is the components eg. number of
    measurements (electrodes) """

tuples = load_data(1, "T")
matrix, trials, classes, artifacts = tuples[0]

matrix /= matrix.std(axis=0)  # Standardize data, not quite sure why...

# Compute ICA
ica = FastICA(n_components=numComponents,   # Specify number of components
              algorithm='deflation',  # The algorithm use (deflation, parallel)
              whiten=True,)  # Removing whitening or not

S_ = ica.fit_transform(matrix)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

""" We could use datavisualisation with matplotlib to see the waves from A_
    after the transformation """
