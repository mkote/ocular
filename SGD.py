from sklearn.linear_model import SGDClassifier
X = [[1,2,3,2,3,4,1,2,3], [4,3,2,5,6,7,8,4,3], [2,3,1,2,5,6,7,4,3]] 
y = [0, 1, 0]
clf = SGDClassifier(loss="oacl_log", penalty="l2", fit_intercept = True)
model = clf.fit(X, y)
print "Thetas: "
print model.coef_
print "bias: "
print model.intercept_