#Importing essential libraries
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

transactions = pd.read_csv('datasets/PSP_Jan_Feb_2019.csv')
transactions.head()
print(transactions.shape)

transactions['tmsp'] = pd.to_datetime(transactions['tmsp'])
transactions['date'] = transactions['tmsp'].dt.date
transactions['weekday'] = transactions['tmsp'].dt.weekday
transactions['hour'] = transactions['tmsp'].dt.hour
display(transactions.head())

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le=LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in transactions.columns.to_numpy():
    # Compare if the dtype is object
    if transactions[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        transactions[col]=le.fit_transform(transactions[col])
        
display(transactions.head())


y = transactions[['success']]
X = transactions.drop(['id', 'tmsp','date','success'], axis=1)

#Use SMOTE to oversample the minority class
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X, y)
over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.2, stratify=over_y)
#Build SMOTE SRF model
SMOTE_SRF = RandomForestClassifier(n_estimators=150, random_state=0)
#Create Stratified K-fold cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scoring = ('f1', 'recall', 'precision')
#Evaluate SMOTE SRF model
scores = cross_validate(SMOTE_SRF, over_X, over_y, scoring=scoring, cv=cv)
#Get average evaluation metrics
print('Mean f1: %.3f' % mean(scores['test_f1']))
print('Mean recall: %.3f' % mean(scores['test_recall']))
print('Mean precision: %.3f' % mean(scores['test_precision']))

#Randomly spilt dataset to test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#Train SMOTE SRF
SMOTE_SRF.fit(over_X_train, over_y_train)

#Display model performance
final_model.append(model_run('Random Forest Classifier (SMOTE)', SMOTE_SRF, X=X_test, y=y_test))

# Print model label
final_model_df = pd.DataFrame(final_model)
final_model_df.set_index('Model', inplace=True)

display(final_model_df)

pickle.dump(SMOTE_SRF, open('model/model.pkl', 'wb'))