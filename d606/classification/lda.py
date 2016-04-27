from math import floor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from preprocessing.dataextractor import load_data

'''NUM_COMPONENTS = 2                              # Why 2?!

tuples = load_data(1, 'T')
matrix, trials, classes, artifacts = tuples[3]

m = matrix.shape[0]
C = [0, 1] * floor(m / 2) + [0] * (m % 2)       # Definitely not optimal
lda = Lda(n_components=NUM_COMPONENTS)
Z = lda.fit_transform(matrix, C)'''


def lda(num_components, data_matrix, target_vector):
    l = Lda(n_components=num_components)
    return l.fit_transform(data_matrix, target_vector)
