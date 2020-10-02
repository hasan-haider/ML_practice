import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

churn = pd.read_csv("dataset.csv", sep=",")
data_size = churn.shape
# print(data_size)
col_names = list(churn.columns)
# print(col_names)
# print(churn.describe())
# print(churn.head())
churn_target = churn["Churn?"]
# print(churn_target)
cols_to_drop = ["Phone", "Churn?"]
churn_feature = churn.drop(cols_to_drop, axis=1)
# print(churn_feature.describe())
churn_categorical = churn_feature.select_dtypes([object])
# print(churn_categorical)
yes_no_cols = ["Int'l Plan", "VMail Plan"]
churn_feature[yes_no_cols] = churn_feature[yes_no_cols] == "yes"
# print(churn_feature[yes_no_cols])
label_encoder = preprocessing.LabelEncoder()
churn_feature['Area Code'] = label_encoder.fit_transform((churn_feature["Area Code"]))
# print(churn_feature["Area Code"])
print(churn_feature.shape, len(churn_feature["State"].unique()))
churn_dumm = pd.get_dummies(churn_feature, columns=["State"], prefix=["State"])
print(churn_dumm.shape)
churn_matrix=churn_dumm.values.astype(np.float)
seed=7
train_data,test_data,train_label,test_label=train_test_split(churn_matrix,churn_target,test_size=.1,random_state=seed)
print(len(train_data))
#Decision tree
#init
classifier=DecisionTreeClassifier(random_state=seed)
#training
classifier=classifier.fit(train_data,train_label)
#predict
churn_pred_target=classifier.predict(test_data)
#evaluate
score=classifier.score(test_data,test_label)
print(score)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=5, n_estimators=15, max_features=60,random_state=seed)
classifier = classifier.fit(train_data, train_label)
score=classifier.score(test_data, test_label)
print('Random Forest classification after model tuning',score)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_label,churn_pred_target))
from sklearn.metrics import confusion_matrix
print('Confusion Matrix',confusion_matrix(test_label,churn_pred_target))
