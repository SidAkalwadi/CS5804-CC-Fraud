import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# Overall thoughts...
# Find a way to randomly select rows from our csv
# Make one half our training and the other half our test data
# Source of code: https://www.w3schools.com/python/python_ml_decision_tree.asp


# Importing Data
training1 = pandas.read_csv("sample-1-25.csv", nrows=1000)
test1 = pandas.read_csv("sample-1-25.csv", skiprows=1000, nrows=1000)

print(training1)

# Telling it our features versus target column

features = ['APPRD_AUTHZN_CNT',	'AVG_DLY_AUTHZN_AMT',	'MRCH_CATG_CD',
	'POS_ENTRY_MTHD_CD',	'RCURG_AUTHZN_IND',	'DISTANCE_FROM_HOME',
    'ACCT_CURR_BAL',	'AUTHZN_AMT',	'AUTHZN_OUTSTD_AMT',	'PLSTC_ISU_DUR']

X = training1[features]
y = training1['FRD_IND']

print(X)
print(y)

# Creating the decision tree and saving it as an example

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('sample-1-25-tree.png')

img=pltimg.imread('sample-1-25-tree.png')
imgplot = plt.imshow(img)
plt.show()

# Need to develop a way to test our data...