import pandas as pd     
import numpy as np             
import matplotlib.pyplot as plt                       
import seaborn as sns             
import warnings
import os
import sys         
from sklearn.metrics import  r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random_Forest":RandomForestRegressor(),
                "Decision_Tree":DecisionTreeRegressor(),
                "Gradient_Boosting":GradientBoostingRegressor(),
                "Linear_Regression":LinearRegression(),
                "K-Neighbours":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting_Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost_Regressor":AdaBoostRegressor(),
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            ###Taking The Best Model
            best_model_score=max(sorted(model_report.values()))
            
            ###GETTING THE BEST MODEL NAME
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]            
            best_model=models[best_model_name]
            if best_model_score<0.75:
                raise CustomException("No Best model found")
            
            logging.info(f"Best model found on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
            
            
            
        except Exception as e:
            raise CustomException(e,sys)    