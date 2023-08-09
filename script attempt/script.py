from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace
import os 

#create an environemnt
env = Environment.from_pip_requirements(name="exam_env", file_path="azure_requirements.txt")

# Create a script config
script_config = ScriptRunConfig(script='training.py', environment=env) 

#Workspace
ws = Workspace.from_config(path="config.json")

# Submit the experiment
experiment = Experiment(workspace=ws, name='training-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion()