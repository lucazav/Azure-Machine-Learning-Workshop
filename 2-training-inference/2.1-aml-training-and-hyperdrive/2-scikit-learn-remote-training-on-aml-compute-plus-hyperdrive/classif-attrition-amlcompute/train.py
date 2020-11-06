import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn_pandas import DataFrameMapper
import os
import sklearn_pandas
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import azureml.core

import argparse
from azureml.core import Run, Workspace, Experiment

os.system('pip freeze')

# If you would like to use Azure ML's tracking and metrics capabilities, you will have to add Azure ML code inside your training script:
# In this train.py script we will log some metrics to our Azure ML run. To do so, we will access the Azure ML Run object within the script:
from azureml.core.run import Run
run = Run.get_context()

# Check library versions
print("SDK version:", azureml.core.VERSION)
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The joblib version is {}.'.format(joblib.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The sklearn_pandas version is {}.'.format(sklearn_pandas.__version__))

# Get Script Arguments for the hyper-parameters
parser = argparse.ArgumentParser(description="Training Script")

# LBFGS stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno. It approximates the second derivative matrix updates with gradient evaluations.
# For more details about Logistic Regression solvers, take a look here: https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
parser.add_argument('--C', type=float, default='1.0',
                    help='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.')

parser.add_argument('--solver', type=str, default='lbfgs',
                    help='Algorithm to use for the ML task')

parser.add_argument('--penalty', type=str, default='l2',
                    help='Used to specify the norm used in the penalization')

parser.add_argument('--l1_ratio', type=float, default=0.5,
                    help='Used to specify the l1 ratio used in case of elasticnet penalty')

args, leftovers = parser.parse_known_args()
# args = parser.parse_args()
    
print('Loading dataset')
if(run.id.startswith("OfflineRun")):
    ws = Workspace.from_config()
    experiment = Experiment(ws, "Train-Explain-Interactive")
    is_remote_run = False
    run = experiment.start_logging(outputs=None, snapshot_directory=".")
    attritionData = ws.datasets['IBM-Employee-Attrition'].to_pandas_dataframe()
else:
    ws = run.experiment.workspace
    attritionData = run.input_datasets['attrition'].to_pandas_dataframe()
    is_remote_run = True

print(attritionData.head())

print('The Scikit-Learn algorithm to use is {}.'.format(np.str(args.solver)))
print('The penalty used is {}.'.format(np.str(args.penalty)))

run.log('C', np.str(args.C))
run.log('Algorithm', np.str(args.solver))
run.log('Penalty', np.str(args.penalty))
run.log('L1 Ratio (only for ElasticNet)', np.float(args.l1_ratio))

# --- Data cleaning -----------------------------------------------

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

attritionData = attritionData.drop(['Over18'], axis=1)

# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)
target = attritionData["Attrition"]

attritionXData = attritionData.drop(['Attrition'], axis=1)
# -----------------------------------------------------------------

# --- Data Transformations -----------------------------------------------

# Collect the categorical column names in separate list
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)
        
# Collect the numerical column names in separate list
numerical = attritionXData.columns.difference(categorical)

# Create data processing pipelines
numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations_pipeline = numeric_transformations + categorical_transformations

# Append classifier algorithm to preprocessing pipeline.
# Now we have a full prediction pipeline.
model_pipeline = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations_pipeline)),
                                 ('classifier', LogisticRegression(C=args.C, solver=args.solver, penalty=args.penalty, l1_ratio=args.l1_ratio))])

# Check Scikit-Learn docs to see the hyper-parameters available for the LogisticRegression:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


# +

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)


# +

OUTPUT_DIR='./outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# write y_test out (placed into /outputs) as a pickle file for later external usage
y_test_pkl = 'y_test.pkl'
joblib.dump(value=y_test, filename=os.path.join(OUTPUT_DIR, y_test_pkl))

# write x_test out (placed into /outputs) as a pickle file for later external usage
x_test_pkl = 'x_test.pkl'
joblib.dump(value=x_test, filename=os.path.join(OUTPUT_DIR, x_test_pkl))

print('Training the model')
# preprocess the data and train the classification model
trained_model = model_pipeline.fit(x_train, y_train)


# +
# Calculate performance metrics for x_test

# Make Multiple Predictions
y_predictions = trained_model.predict(x_test)

accuracy = accuracy_score(y_test, y_predictions)
rocauc = roc_auc_score(y_test, y_predictions)
average_precision = average_precision_score(y_test, y_predictions)
# (Method 2 with svm model) accuracy = svm_model_linear.score(x_test, y_test)

print('Accuracy of LogisticRegression classifier on test set: {:.2f}'.format(accuracy))
print('ROC-AUC of LogisticRegression classifier on test set: {:.2f}'.format(rocauc))
print('Avg Precision of LogisticRegression classifier on test set: {:.2f}'.format(average_precision))

# Set Accuracy metric to the run.log to be used later by Azure ML's tracking and metrics capabilities
run.log("Accuracy", float(accuracy))
run.log("ROC-AUC", float(rocauc))
run.log("Avg Precision", float(average_precision))

# creating a confusion matrix
cm = confusion_matrix(y_test, y_predictions)
print(cm)

# Save model as -pkl file to the outputs/ folder to use outside the script
model_file_name = 'classif-empl-attrition.pkl'
joblib.dump(value=trained_model, filename=os.path.join(OUTPUT_DIR, model_file_name))


if not is_remote_run:
    run.complete()

print('completed')

