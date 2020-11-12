import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

#sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23.
#Please import this functionality directly from joblib, which can be installed with: pip install joblib.

#from sklearn.externals import joblib
import joblib

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{'Age': 41, 'BusinessTravel': 'Travel_Rarely', 'DailyRate': 1102, 'Department': 'Sales', 'DistanceFromHome': 1, 'Education': 2, 'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': 2, 'Gender': 'Female', 'HourlyRate': 94, 'JobInvolvement': 3, 'JobLevel': 2, 'JobRole': 'Sales Executive', 'JobSatisfaction': 4, 'MaritalStatus': 'Single', 'MonthlyIncome': 5993, 'MonthlyRate': 19479, 'NumCompaniesWorked': 8, 'OverTime': 0, 'PercentSalaryHike': 11, 'PerformanceRating': 3, 'RelationshipSatisfaction': 1, 'StockOptionLevel': 0, 'TotalWorkingYears': 8, 'TrainingTimesLastYear': 0, 'WorkLifeBalance': 1, 'YearsAtCompany': 6, 'YearsInCurrentRole': 4, 'YearsSinceLastPromotion': 0, 'YearsWithCurrManager': 5}])
output_sample = np.array([0])


def init():
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    global model
    
    model_path = os.getenv('AZUREML_MODEL_DIR')
    if (model_path is None):
        model_path = '.'
    
    model_path = os.path.join(model_path, 'classif-empl-attrition.pkl')
    print(model_path)
    
    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    
# Test the functions if run locally
if __name__ == "__main__":
    init()
    
    prediction = run(input_sample)

    print(prediction)
