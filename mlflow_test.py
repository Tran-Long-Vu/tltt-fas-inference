from configs.config import *
from libs import *
from engines.scrfd import SCRFD
from data_script.image_dataset import ImageDataset
from configs.config import *
import sklearn.metrics as metrics
import pandas as pd
import onnx
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# todo - configs file
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow VHT Quickstart")
class Tester():
    def __init__(self) -> None:
        # self.model_backbone = "rn18" # change between 'rn18' and 'mnv3'
        # self.attack_type = ATTACK_TYPE
        # self.model = self.load_model()
        # self.optimizer = optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr = 1e-5, # 0.0001
        # )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.save_ckp_dir = PATH_TO_SAVE_CHECKPOINT
        self.epochs = NO_EPOCHS
    
    # init video dataset format # later
    def load_video_dataset(self):
        return 0    
    # run printing attack dataset
    def train_printing_attack(self,):
        # Load the Iris dataset
        X, y = datasets.load_iris(return_X_y=True)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define the model hyperparameters
        params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": 8888,
        }

        # Train the model
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        # Predict on the test set
        y_pred = lr.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        with mlflow.start_run(run_name = 'epoch 5, cvpr set on resnet18, printing'):
            # mlflow.set_tag("Testing", "epoch 5, cvpr set on resnet18, printing attack")
            
            # Log the hyperparameters
            mlflow.log_params(params)

            # Log the loss metric
            mlflow.log_metric("accuracy", accuracy)

            # Set a tag that we can use to remind ourselves what this run was for
            # smlflow.set_tag("Training Info", "Basic LR model for iris data")

            # # Infer the model signature
            # signature = infer_signature(X_train, lr.predict(X_train))

            # # Log the model
            # model_info = mlflow.sklearn.log_model(
            #     sk_model=lr,
            #     artifact_path="iris_model",
            #     signature=signature,
            #     input_example=X_train,
            #     registered_model_name="tracking-quickstart",
            # )
            # Load the model back for predictions as a generic Python Function model
            # inference
            # loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

            # predictions = loaded_model.predict(X_test)

            # iris_feature_names = datasets.load_iris().feature_names

            # result = pd.DataFrame(X_test, columns=iris_feature_names)
            # result["actual_class"] = y_test
            # result["predicted_class"] = predictions

            # result[:4]
        
        pass
    def train_replay_attack(self):
        
        pass
    def visualize():
        pass

if __name__ == '__main__':
    mlflowtest = Tester()
    mlflowtest.train_printing_attack()
    pass