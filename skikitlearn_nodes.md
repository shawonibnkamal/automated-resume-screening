# Skikit-learn tutorial

Simple and efficient tool for data mining and analysis. Built on Numpy, Scipy and matplotlib.

Using skikit-learn:

- We can achieve 
  - classification: identify which category and object belongs to 
  - Regression: predicting an attribute associated with an object
  - clustering: automatic grouping of similar objects into sets
  - model selection: comparing, validating and choosing parameters and models
  - Dimensionality reduction: reducing the number of random variables to consider
  - Pre-processing: Feature extraction and normalization

### Importing requred projects

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    #fromsklearn.linear_model import SGDClassifer
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    %matplotlib inline

### Preprocessing data

```python
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
```