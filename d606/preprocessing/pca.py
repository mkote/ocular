from sklearn.decomposition import PCA
from d606.preprocessing.dataextractor import load_data

""" number of components, might be a constant in the beginning,
    but might come in handy if it were
    to be returned from loadEEGFile """

numComponents = 17  # Number of components to keep

""" http://scikit-learn.org/stable/auto_examples/
    decomposition/plot_ica_blind_source_separation.html

    http://scikit-learn.org/stable/modules/decomposition.html#pca
    Algorithm for Principal Component Analysis

    Data X needs to be en n * x matrix, where n is
    samples, and m is the components eg. number of
    measurements (electrodes) """

tuples = load_data(1, 'T')
matrix, trials, classes, artifacts = tuples[0]

matrix /= matrix.std(axis=0)  # Standardize data, not quite sure why...

# Compute PCA
# Specify number of components 'mle' gives a guess
pca = PCA(n_components=numComponents,  # number of components 'mle' gives guess
          whiten=False)

H = pca.fit_transform(matrix)  # Reconstruct signals from orthogonal components

""" We could use datavisualisation with matplotlib to
    see the waves from H after the transformation """
