from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#import the data
data = pandas.read_csv("sample-1-25.csv", nrows=2000)

features = ['APPRD_AUTHZN_CNT', 'AVG_DLY_AUTHZN_AMT', 'MRCH_CATG_CD',
	'POS_ENTRY_MTHD_CD', 'RCURG_AUTHZN_IND', 'DISTANCE_FROM_HOME',
    'ACCT_CURR_BAL', 'AUTHZN_AMT', 'AUTHZN_OUTSTD_AMT', 'PLSTC_ISU_DUR']

X = data[features]
Y = data['FRD_IND']

#split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#create instance of log res
logisticRegr = LogisticRegression(solver = 'lbfgs')

#fit the data
logisticRegr.fit(x_train, y_train)

#make the predictions
predictions = logisticRegr.predict(x_test)

# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

#plot confusion matrix with colors
plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)