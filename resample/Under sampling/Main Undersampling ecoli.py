import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    get_ipython().magic("matplotlib inline")
except:
    plt.ion()
#matplotlib inline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score,roc_curve, roc_auc_score
import imblearn
from imblearn.metrics import classification_report_imbalanced
from collections import Counter
from random import randint

from imblearn.metrics import geometric_mean_score
df_columns=['age_c','assess_c','compfilm_c','density_c',
            'famhx_c','gg','rr','cancer_c']
df = pd.read_csv("ecoli_4.csv",skiprows=0,header=0,names=df_columns)
#r=randint(2,100)
#df1= df.sample(n=10000, random_state=r)
#print(df1.shape)
k = 1
base_classifier = "DT"
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import Normalizer,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
X_df = df.drop('cancer_c', axis=1)


y_df = df.cancer_c


X_train_def, X_test_def, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

print("yt",y_train.shape)
scl = StandardScaler()
X_train_normal = scl.fit_transform(X_train_def)
X_test_normal = scl.transform(X_test_def)
X_train=np.ascontiguousarray(X_train_normal,dtype=np.float64)
X_test=np.ascontiguousarray(X_test_normal,dtype=np.float64)

X_train_majority = X_train[y_train==0]
X_train_minority = X_train[y_train==1]

y_train_majority = y_train[y_train==0]
y_train_minority = y_train[y_train==1]


y_train_minority = y_train_minority.reset_index(drop=True)


y_train_majority = y_train_majority.reset_index(drop=True)


import pydpc
from pydpc import Cluster
dpc = Cluster(X_train_majority,fraction=0.001,autoplot=True)


import clustering_selection1
cluster_index,clusters_density,cluster_distance,cluster_ins_den = clustering_selection1.clustering_dpc(
    X_train_majority,X_train_minority,y_train_majority,y_train_minority,0,0)

alpha = 0.7
beta = 0.3

X_train_balanced, y_train_balanced, indexs = clustering_selection1.selection(X_train_majority, X_train_minority, y_train_majority,
                                                                    y_train_minority, cluster_index, clusters_density,
                                                                    cluster_distance, alpha, beta, cluster_ins_den)
print(X_train[1])

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100,algorithm="SAMME", learning_rate=0.2)
model.fit(X_train_balanced, y_train_balanced)

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


