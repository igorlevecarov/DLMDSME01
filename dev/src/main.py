import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#3.1 Feature Engineering
transactions = pd.read_csv('datasets/PSP_Jan_Feb_2019.csv')
transactions.head()
print(transactions.shape)

transactions['tmsp'] = pd.to_datetime(transactions['tmsp'])
transactions['date'] = transactions['tmsp'].dt.date
transactions['weekday'] = transactions['tmsp'].dt.weekday
transactions['hour'] = transactions['tmsp'].dt.hour
display(transactions.head())

#3.2 Data Formatting

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

#3.3 Data Selection
# Import train_test_split
from sklearn.model_selection import train_test_split

transactions_data = transactions.drop(['id', 'tmsp','date'], axis=1)

# Create train and test sets
train, test = train_test_split(transactions_data, test_size=0.2, random_state=42)

# Train: X and y split
X_train = train.drop('success', axis=1)
y_train = train[['success']]
display(X_train.head())
display(y_train.head())

# Train: X and y split
X_test = test.drop('success', axis=1)
y_test = test[['success']]
display(X_test.head())
display(y_test.head())

# Import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler and use it to rescale X_train and X_test
scaler = StandardScaler()
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

display(rescaledX_train)

X_train_e_l_n = pd.DataFrame(data=rescaledX_train, index=X_train.index, columns=X_train.columns)    
X_test_e_l_n = pd.DataFrame(data=rescaledX_test, index=X_test.index, columns=X_test.columns) 

display(X_train_e_l_n.describe())

def model_run(model, clf, X=X_test, y=y_test):

    # Evaluate error rates and append to lists
    y_test_predict = clf.predict(X)
    auc = roc_auc_score(y, y_test_predict)
    precision =precision_score(y, y_test_predict)
    recall = recall_score(y, y_test_predict)
    accuracy = clf.score(X, y)

    # Print classification report
    print(f'== {model} - Classification report ==')
    print(classification_report(y, y_test_predict))
    
    # Plot confusion matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_test_predict)), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title(f'{model}- Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')   
    plt.show()

    return {'Model':model, \
            'Accuracy': round(accuracy,5) , \
            'Precision':round(precision,5) , \
            'Recall':round(recall,5)}


#4. Modeling

#4.1 Logistic Regression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
import time

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.utils import resample,shuffle
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

# Define list to store result 
final_result = list()
# Define list to store result 
final_model = list()

# Instantiate a LogisticRegression classifier with default parameter values
lrc = LogisticRegression()

# Fit logreg to the train set
lrc.fit(rescaledX_train,y_train)

#Display model performance
final_model.append(model_run('Logistic regression', lrc, X=rescaledX_test, y=y_test))


#4.2 Gradient Boosting
import matplotlib.pyplot as plt

# Instantiate a GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=42)

# Fit GradientBoostingClassifier to the train set
gbc.fit(rescaledX_train,y_train)


#4.2.3 Random forest
#Display model performance
final_model.append(model_run('Gradient Boosting Classifier', gbc, X=rescaledX_test, y=y_test))

## Instantiate a RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=135, max_depth=6,criterion='gini', random_state=42,max_features='auto')

# Fit RandomForestClassifier to the train set
rfc.fit(rescaledX_train,y_train)

#Display model performance
final_model.append(model_run('Random Forest Classifier', rfc, X=rescaledX_test, y=y_test))

# Print model label
final_model_df = pd.DataFrame(final_model)
final_model_df.set_index('Model', inplace=True)

display(final_model_df)


