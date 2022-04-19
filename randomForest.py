import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np


data = pandas.read_csv("sample-1-25.csv", nrows=2000)

features = ['APPRD_AUTHZN_CNT', 'AVG_DLY_AUTHZN_AMT', 'MRCH_CATG_CD',
	'POS_ENTRY_MTHD_CD', 'RCURG_AUTHZN_IND', 'DISTANCE_FROM_HOME',
    'ACCT_CURR_BAL', 'AUTHZN_AMT', 'AUTHZN_OUTSTD_AMT', 'PLSTC_ISU_DUR']

X = data[features]
Y = data['FRD_IND']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(x_train, y_train)
pred = forest.predict(x_test)
print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, pred))

'''
fig = plt.figure(figsize=(15, 10))
plot_tree(forest.estimators_[0],
            feature_names = features,
            class_names = ['FRD_IND'],
            filled = True,
            impurity = True,
            rounded = True)

fig.savefig('forest.png')
'''
