from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
from utils import PredictInfluence

'''API application

'''
app = Flask(__name__)
api = Api(app)

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictInfluence, '/')

if __name__ == '__main__':
    app.run(debug=True)
