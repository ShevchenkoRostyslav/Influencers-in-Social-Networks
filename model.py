'''
author: Rostyslav Shevchenko
email: shevchenko.rostislav@gmail.com
date: 01.12.2018
'''

# load ML model
import pickle

class MLModel(object):
    """Simple ML model"""

    def __init__(self,model_path):
        """
        Attributes:
            clf: ML classifier model

        """
        with open(model_path,'r') as f:
            self.clf = pickle.load(f)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred