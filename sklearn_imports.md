# Scikit-learn import commands


### **General Scikit-learn Imports**
```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
```

---

### **Data Preprocessing**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import PolynomialFeatures
```

---

### **Model Selection**
```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
```

---

### **Metrics and Scoring**
```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
```

---

### **Feature Selection**
```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
```

---

### **Dimensionality Reduction**
```python
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
```

---

### **Supervised Learning Models**
#### Linear Models
```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
```

#### Support Vector Machines
```python
from sklearn.svm import SVC
from sklearn.svm import SVR
```

#### Tree-based Models
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
```

#### Ensemble Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
```

---

### **Unsupervised Learning Models**
#### Clustering
```python
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
```

#### Gaussian Models
```python
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
```

---

### **Model Evaluation and Validation**
```python
from sklearn.metrics import make_scorer
from sklearn.metrics import silhouette_score
```

---

### **Pipeline and Utilities**
```python
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
```

---