import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import itertools
df = pd.read_csv('loan_train.csv')
pd.set_option('display.max_columns', None)
print(df.head())
print(df.shape)
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head())
print(df['loan_status'].value_counts())
import seaborn as sns
# creating histogram of loan status along with principal based on gender
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
# creating similar histograms but replacing principal
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
print(df['dayofweek'].value_counts())
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
print(df.head())
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature = df[['Principal','terms','age','Gender','weekend']]
print(Feature.head())
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
print(Feature.head())
X = Feature
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])
y = df['loan_status'].values
print(y[0:5])
test_df = pd.read_csv('loan_test.csv')
print(test_df.head())
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
Test_Feature = pd.concat([Test_Feature,pd.get_dummies(test_df['education'])], axis=1)
Test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
print(Test_Feature.head())
X_test = Test_Feature
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
print(X_test[0:5])
y_test = test_df['loan_status'].values
print(y_test[0:5])
from sklearn.model_selection import train_test_split
X, X_val, y, y_val = train_test_split( X_test, y_test, test_size=0.2, random_state=4)
print ('Train set:', X.shape,  y.shape)
print ('Validation set:', X_val.shape,  y_val.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# for loop that tests KNN to find best k and produces plot with accuracy values, this is how I determined K = 5 was the best value
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
   # Train Model and Predict
   neigh = KNeighborsClassifier(n_neighbors=n).fit(X, y)
   yhat = neigh.predict(X_val)
   mean_acc[n - 1] = metrics.accuracy_score(y_val, yhat)
   std_acc[n - 1] = np.std(yhat == y) / np.sqrt(yhat.shape[0])

print(mean_acc)
print( "The best accuracy was", mean_acc.max(), "with k=", mean_acc.argmax()+1)
plt.plot(range(1,Ks),mean_acc,'g')
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
# building models
#KNN
neigh = KNeighborsClassifier(n_neighbors=7).fit(X, y)
print(neigh)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y, neigh.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, neigh.predict(X_val)))
print (classification_report(y, neigh.predict(X)))
# tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# setting reasonable max depth to prevent overfitting
loantree = DecisionTreeClassifier(criterion='entropy', max_depth= 4, random_state=0)
loantree = loantree.fit(X, y)
print("Train set Accuracy: ", metrics.accuracy_score(y, loantree.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, loantree.predict(X_val)))
print (classification_report(y, loantree.predict(X)))
# printing tree in text representation
text_representation = tree.export_text(loantree)
#print(text_representation)
fig = plt.figure(figsize=(25,20))
sklearn.tree.plot_tree(loantree, feature_names=Feature.columns[0:7], class_names= np.unique(y), filled=True)
fig.savefig("loan_tree.png")
# SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf', random_state=0)
clf.fit(X, y)
print(clf)
print("Train set Accuracy: ", metrics.accuracy_score(y, clf.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, clf.predict(X_val)))
print (classification_report(y, clf.predict(X)))

# LR
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=.1, solver='liblinear', random_state=0).fit(X, y)
print(LR)
print("LR Train set Accuracy: ", metrics.accuracy_score(y, LR.predict(X)))
print("LR Validation set Accuracy: ", metrics.accuracy_score(y_val, LR.predict(X_val)))
print (classification_report(y, LR.predict(X)))
from sklearn.model_selection import GridSearchCV
# An example of a parameter gridsearch (useful on lightweight models)
parameters = {'C': [0.1,1,10,100,1000]}
grid_search = GridSearchCV(estimator=LR, param_grid=parameters)
grid_search.fit(X, y)
print(grid_search.best_params_)
print(grid_search.best_score_)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
random_forest.fit(X, y)
print(random_forest.feature_importances_)
print("Train set Accuracy: ", metrics.accuracy_score(y, random_forest.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, random_forest.predict(X_val)))
print (classification_report(y, random_forest.predict(X)))
# Gradient Boosting using a 1000 estimators and setting reasonable learning rate and max depth
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=0)
gb.fit(X, y)
#print(gb)
print("Train set Accuracy: ", metrics.accuracy_score(y, gb.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, gb.predict(X_val)))
print (classification_report(y, gb.predict(X)))
# Multi Layer Perceptron - we will use the quasi newton solver, as it is faster to converge than adam due to its
# second order approximation accounting for curvature better, and since our dataset is smaller
# using it makes sense. Model also has hidden layers of size 4 and 1, using relu
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 1), random_state=0).fit(X, y)
print(mlp)
print("Train set Accuracy: ", metrics.accuracy_score(y, mlp.predict(X)))
print("Validation set Accuracy: ", metrics.accuracy_score(y_val, mlp.predict(X_val)))
print (classification_report(y, mlp.predict(X)))
# Evaluation metrics for KNN
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
yhat_knn = neigh.predict(X_test)
print('KNN Jaccard Score:', jaccard_score(y_test, yhat_knn,pos_label='PAIDOFF'))
print('KNN F1 Score:', f1_score(y_test, yhat_knn, average='weighted'))
# Evluation metrics for decision tree
yhat_tree = loantree.predict(X_test)
print('Decision Tree Jaccard Score:', jaccard_score(y_test, yhat_tree,pos_label='PAIDOFF'))
print('Decision Tree F1 Score:', f1_score(y_test, yhat_tree, average='weighted'))
# Evaluation metrics for SVM
yhat_SVM = clf.predict(X_test)
print('SVM Jaccard Score:', jaccard_score(y_test, yhat_SVM,pos_label='PAIDOFF'))
print('SVM F1 Score:', f1_score(y_test, yhat_SVM, pos_label='PAIDOFF', average='weighted'))
# Evaluation metrics for Logistic Regression
yhat_LR = LR.predict(X_test)
yhat_LR_prob = LR.predict_proba(X_test)
print('Logistic Regression Jaccard Score:', jaccard_score(y_test, yhat_LR,pos_label='PAIDOFF'))
print('Logistic Regression F1 Score:', f1_score(y_test, yhat_LR, average='weighted'))
print('Logistic Regression Log Loss:', log_loss(y_test, yhat_LR_prob))

# metrics for Random Forest
yhat_RF = random_forest.predict(X_test)
print('Random Forest Jaccard Score:', jaccard_score(y_test, yhat_RF,pos_label='PAIDOFF'))
print('Random Forest F1 Score:', f1_score(y_test, yhat_RF, average='weighted'))
# metrics for Gradient Boosting
yhat_GB = gb.predict(X_test)
print('Gradient Boosting Jaccard Score:', jaccard_score(y_test, yhat_GB,pos_label='PAIDOFF'))
print('Gradient Boosting F1 Score:', f1_score(y_test, yhat_GB, average='weighted'))

# metrics for Multi Layer Perceptron
yhat_MLP = mlp.predict(X_test)
yhat_MLP_prob = mlp.predict_proba(X_test)
print('Multi Layer Perceptron Jaccard Score:', jaccard_score(y_test, yhat_MLP,pos_label='PAIDOFF'))
print('Multi Layer Perceptron F1 Score:', f1_score(y_test, yhat_MLP, average='weighted'))
print('Multi Layer Perceptron Log Loss:', log_loss(y_test, yhat_MLP_prob))
# Since the MLP did the best, let's compute the confusion matrix, Roc curve and use SHAP to explain the model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from shap import KernelExplainer # This is essentually a way of computing the marginal contributions of each feature to the model
# based on each feature's marginal contributions and is rooted in game theory, allows us to open
# up the black box for non-intuitive models
# compute confusion matrix
cm = confusion_matrix(y_test, yhat_MLP)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, yhat_MLP_prob[:,1], pos_label='PAIDOFF')
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
# compute ROC AUC score
print('ROC AUC Score:', roc_auc_score(y_test, yhat_MLP_prob[:,1]))
# compute SHAP values
explainer = KernelExplainer(mlp.predict_proba, X)
shap_values = explainer.shap_values(X_test)
#print('SHAP values:', shap_values)
# plot SHAP values
import shap
shap.summary_plot(shap_values, X_test, plot_type='bar')
# from the summary plot, it seems the high school or below feature is the most important
# as per the model, followed by weekend, and gender
# Interestingly enough, the most important features for the most successful model were
# related to social factors, and not the contents of the loan itself
# This dataset is limited though, and is missing key information, such as the purpose
# of the loan, income, credit rating, etc.
