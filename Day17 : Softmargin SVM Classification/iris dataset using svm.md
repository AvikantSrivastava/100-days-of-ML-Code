
# import part



```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
```

# importing the dataset


```python
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
```

# softmargin svm classification


```python
svm_clf = Pipeline([
("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1, loss="hinge"))])
svm_clf.fit(X, y)
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('linear_svc',
                     LinearSVC(C=1, class_weight=None, dual=True,
                               fit_intercept=True, intercept_scaling=1,
                               loss='hinge', max_iter=1000, multi_class='ovr',
                               penalty='l2', random_state=None, tol=0.0001,
                               verbose=0))],
             verbose=False)




```python
svm_clf.predict([[5.5, 1.7]])
```




    array([1.])


