import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import json
from allegroai import Task
from clearml import Dataset
import argparse
import time

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='Pipeline Demos', task_name='Step 2 - Training a model',
    output_uri=True)
# Set default docker
task.set_base_docker(docker_image="python:3.9-bullseye")

#Argparser
parser = argparse.ArgumentParser(description='Do basic training on a dataset')
parser.add_argument('--eval_metric', help='metric to evaluate', default="rmse")
parser.add_argument('--objective', help='error evaluation for multiclass training', default='reg:squarederror')
parser.add_argument('--best_model_id', help='best model ID so far, if exists', default=None)
parser.add_argument('--dataset_name', help='name to use, if exists', default="demo_dataset")
parser.add_argument('--project_name', help='project dataset is in', default="Pipeline Demos")
parser.add_argument('--test_size', help='our test size', default=0.2)
parser.add_argument('--random_state', help='random seed to use', default=42)
parser.add_argument('--num_boost_round', help='number of iterations', default=100)
args = parser.parse_args()

# Config files
config_fpath = task.connect_configuration("./config.json")

with open(config_fpath) as fptr:
    file_args = argparse.Namespace(**json.load(fptr))

# Load our Dataset
local_copy = Dataset.get(dataset_name=args.dataset_name, dataset_project=args.project_name, alias="Input data").get_local_copy()
iris = pickle.load(open("{}/iris.pkl".format(local_copy), 'rb'))

# Split data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train
params = vars(args)
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=args.num_boost_round,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=file_args.verbose_eval,
)

bst.save_model("best_model")

for i in range(40):
    time.sleep(1)
    print(i)

print(file_args.final_msg)
print("Done")

