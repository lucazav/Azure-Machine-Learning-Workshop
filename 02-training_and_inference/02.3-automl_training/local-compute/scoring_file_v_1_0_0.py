# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Age": pd.Series(["57"], dtype="int64"), "BusinessTravel": pd.Series(["Travel_Rarely"], dtype="object"), "DailyRate": pd.Series(["334"], dtype="int64"), "Department": pd.Series(["Research & Development"], dtype="object"), "DistanceFromHome": pd.Series(["24"], dtype="int64"), "Education": pd.Series(["2"], dtype="int64"), "EducationField": pd.Series(["Life Sciences"], dtype="object"), "EnvironmentSatisfaction": pd.Series(["3"], dtype="int64"), "Gender": pd.Series(["Male"], dtype="object"), "HourlyRate": pd.Series(["83"], dtype="int64"), "JobInvolvement": pd.Series(["4"], dtype="int64"), "JobLevel": pd.Series(["3"], dtype="int64"), "JobRole": pd.Series(["Healthcare Representative"], dtype="object"), "JobSatisfaction": pd.Series(["4"], dtype="int64"), "MaritalStatus": pd.Series(["Divorced"], dtype="object"), "MonthlyIncome": pd.Series(["9439"], dtype="int64"), "MonthlyRate": pd.Series(["23402"], dtype="int64"), "NumCompaniesWorked": pd.Series(["3"], dtype="int64"), "OverTime": pd.Series(["1"], dtype="int64"), "PercentSalaryHike": pd.Series(["16"], dtype="int64"), "PerformanceRating": pd.Series(["3"], dtype="int64"), "RelationshipSatisfaction": pd.Series(["2"], dtype="int64"), "StockOptionLevel": pd.Series(["1"], dtype="int64"), "TotalWorkingYears": pd.Series(["12"], dtype="int64"), "TrainingTimesLastYear": pd.Series(["2"], dtype="int64"), "WorkLifeBalance": pd.Series(["1"], dtype="int64"), "YearsAtCompany": pd.Series(["5"], dtype="int64"), "YearsInCurrentRole": pd.Series(["3"], dtype="int64"), "YearsSinceLastPromotion": pd.Series(["1"], dtype="int64"), "YearsWithCurrManager": pd.Series(["4"], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        path = os.path.normpath(model_path)
        path_split = path.split(os.sep)
        log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
