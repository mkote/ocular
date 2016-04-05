import numpy as np
import matplotlib.pyplot as plt

# TODO: fix plotting to be more flexible and need refactoring
# Taken from
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
# Mesh step size is the unit size of the plot.
# Classifiers is a list of classifiers to be plotted in we want to plot
# several classifiers side-by-side. In the case of n classifiers,
# we need a list of n titles.


def plot_svm(self, feature_values, target_values,
             mesh_step_size, classifiers, titles, x_axis_label, y_axis_label):
    # create a mesh to plot in
    x_min, x_max = feature_values[:, 0].min() - 1, \
        feature_values[:, 0].max() + 1
    y_min, y_max = feature_values[:, 1].min() - 1, \
        feature_values[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    for i, clf in classifiers:
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(feature_values[:, 0], feature_values[:, 1],
                    c=target_values, cmap=plt.cm.Paired)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
