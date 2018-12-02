'''
author: Rostyslav Shevchenko
email: shevchenko.rostislav@gmail.com
date: 01.12.2018
'''

import pandas as pd 
import numpy as np

from flask_restful import reqparse, Resource
from model import MLModel

class NetworkData:
    
    def __init__(self, file):
        '''Arguments:
        data - pandas DF with the input data
        columns - features used to make predictions
        '''
        self.columns_ = ['B_network_feature_1', 'A_network_feature_1',
                         'B_listed_count', 'A_following_count', 'B_following_count',
                         ]
        self.data_ = pd.read_csv(file, sep=',')
        # Check whether features are consistent with trained model
        if not self.check_data():
            raise ValueError(
                'Input data do not included required features: {}'.format(self.columns_))

    def load_data(self,file):
        '''Load .csv data

        '''
        self.data_ = pd.read_csv(file, sep=',')
        # Check whether features are consistent with trained model
        if not self.check_data():
            raise ValueError(
                'Input data do not included required features: {}'.format(self.columns_))

    def check_data(self):
        '''Check whether data is consistent with trained model
    
        '''
        return all(feature in self.data_.columns for feature in self.columns_)

    def add_derived_features(self,df):
        '''Add derived features

        '''
        minus_features = ['follower_count', 'listed_count']
        for col in minus_features:
            df['AmB_' + col] = df['A_'+col] - df['B_'+col]
        for cand in ['A', 'B']:
            df[cand + '_follower_m_following_count'] = df[cand +
                                                      '_follower_count'] - df[cand + '_following_count']

    def transform_data(self):
        '''Transform the DF

        '''
        self.add_derived_features(self.data_)

    def data_to_predict(self):
        '''Return data for the predictions

        '''
        return self.data_[self.columns_]

    def df(self):
        '''Get the input DF

        '''
        return self.data_

class InputParser():
    def __init__(self):
        self.parser_ = reqparse.RequestParser()
        self.add_arguments_()
        self.args_ = self.parser_.parse_args()

    def add_arguments_(self):
        self.parser_.add_argument(
            'input_file', required=True)
        self.parser_.add_argument(
            'classifier', type=str, help='Classifier: XGBoost or RandomForest', default='XGBoost')
    
    def get_parser(self):
        return self.parser_

    def get_input_file(self):
        return self.args_['input_file']

    def get_clf(self):
        return self.args_['classifier']

class PredictInfluence(Resource):

    def __init__(self):
        # use parser and find the user's input file and ML model
        parser = InputParser()
        input_file = parser.get_input_file()
        clf_name = parser.get_clf()

        #Prepare the inpit data
        self.data = NetworkData(input_file)
        self.data.transform_data()

        #Get the model
        self.clf = MLModel(clf_name)

    def predict_proba(self):
        # predict probabilities 
        predict_prob = self.clf.predict_proba(self.data.data_to_predict())
        # Convert probabilities to DF
        predictions_prob_df = pd.DataFrame(predict_prob[:, 0])
        predictions_prob_df.index = np.arange(1, len(predictions_prob_df)+1)
        return predictions_prob_df

    def predict(self):
        # predict
        predict = self.clf.predict(self.data.data_to_predict())
        # Convert probabilities to DF
        predictions_df = pd.DataFrame(predict[:, 0])
        predictions_df.index = np.arange(1, len(predictions_df)+1)
        return predictions_df

    def get(self):
        # Make Prediction
        predict_prob = self.predict_proba()
        # create JSON object
        output = predict_prob.to_json()
        return 'lol'
        #return output