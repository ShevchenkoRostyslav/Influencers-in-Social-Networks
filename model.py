'''
author: Rostyslav Shevchenko
email: shevchenko.rostislav@gmail.com
date: 01.12.2018
'''

# load ML model
import pickle

available_models = ['XGBoost', 'RandomForest']

class MLModel(object):
    """Simple ML model"""

    def __init__(self,name):
        """
        Attributes:
            clf_: ML classifier model
            name_: name of the ML model
        """
        self.name_ = name
        path = self.path_from_name_(name)
        self.load_model(path)

    def path_from_name_(self,name):
        """Convert the name of the model to the path

        """
        if name not in available_models:
            raise ValueError('Model ' + name + ' is not available. Available models: {}'.format(available_models))
        else:
            return 'lib/models/' + name + '.pkl'

    def load_model(self,path):
        """Load the model from the path

        """
        with open(path, 'rb') as f:
            self.clf_ = pickle.load(f)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        
        """
        y_proba = self.clf_.predict_proba(X)
        return y_proba

    def predict(self, X):
        """Returns the predicted class in an array
        
        """
        y_pred = self.clf_.predict(X)
        return y_pred

