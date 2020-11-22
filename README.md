# Azure Machine Learning Workshop

## Target Audience
Anyone who wants a comprehensive understanding of Azure ML.

## Key Goals
1.	Understand the product E2E
2.	Get confident with the basis of the Azure ML Python SDK

## Agenda

### Workspace Concepts
1. [Set up your workspace and compute](./01-workspace_concepts/01-setup_azure_machine_learning.md)
2. [Register a dataset](./01-workspace_concepts/02-datasets_and_datastores.md)
3. [Run AutoML from the UI](./01-workspace_concepts/03-automated_machine_learning.md)
4. [Azure ML Designer](./01-workspace_concepts/04-azure_ml_designer.md)
1. [Compute Instance - Clone Git Repo](./01-workspace_concepts/05-clone_git_repo.md)


### Training and Logging 

#### Azure ML Training & HyperDrive 

1.	[Training of a classification model and interactive logging on Azure ML](./02-training_and_inference/02.1-aml_training_and_hyperdrive/1-scikit-learn-local-training-on-notebook-plus-aml-ds-and-log/binayclassification-employee-attrition-notebook.ipynb)
2.	[Remote training via Azure ML Cluster](./02-training_and_inference/02.1-aml_training_and_hyperdrive/2-scikit-learn-remote-training-on-aml-compute-plus-hyperdrive/binayclassification-employee-attrition-aml-compute-notebook.ipynb)

### Azure ML Interpretability

1. [Setup](./02-training_and_inference/02.2-aml_interpretability/00-setup.ipynb)
2. [Explain binary classification model predictions locally](./02-training_and_inference/02.2-aml_interpretability/01-explain_model_predictions_locally.ipynb)

### Training Using AutoML

1. [AutoML on Local Compute](./02-training_and_inference/02.3-automl_training/local-compute/01-binay_classification_employee_attrition_autoaml_local_compute.ipynb)
2. [AutoML on Remote Compute](./02-training_and_inference/02.3-automl_training/remote-compute/02-binay_classification_employee_attrition_autoaml_remote_amlcompute.ipynb)

### Deploy the Model on an Azure Web Services

1. [Deploying a web service to Azure Container Instance (ACI) or Azure Kubernetes Services (AKS)](./02-training_and_inference/02.4-model_deployment/01-deploying_a_web_service_to_azure_container_instance.ipynb)
