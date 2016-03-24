from sklearn.ensemble import RandomForestClassifier


def random_forest_learner(attribute_values, target_values, num_estimators):
    # from scikit learn doc example
    clf = RandomForestClassifier(n_estimators=num_estimators)
    clf = clf.fit(attribute_values, target_values)
    return clf
