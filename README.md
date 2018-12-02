# Influencers-in-Social-Networks

Kaggle competition: https://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network

The dataset, provided by Peerindex, comprises a standard, pair-wise preference learning task. Each datapoint describes two individuals, A and B. For each person, 11 pre-computed, non-negative numeric features based on twitter activity (such as volume of interactions, number of followers, etc) are provided.

The binary label represents a human judgement about which one of the two individuals is more influential. A label '1' means A is more influential than B. 0 means B is more influential than A. The goal of the challenge is to train a machine learning model which, for pairs of individuals, predicts the human judgement on who is more influential with high accuracy.

## File Structure
* app_name
  * app.py: Flask API application
  * model.py: class object for ML classifier
  * PrepareMLModel.ipynb: prepare a simple ML model, studied in other branches, trains the model, and pickle
  * util.py: helper classes used to work with user-input data, parsing the input arguments and make predictions
  * lib
      * data: directory that contains the data files from and cleaned dataset prepared in other branches
      * models: directory that contains the pickled model files


## Testing the API
1. Run the Flask API locally for testing. Go to directory with `app.py`.

```bash
python app.py
```

2. GET request at the URL of the API.

* curl

```bash
curl -X GET http://127.0.0.1:5000/\?input_file\=https://raw.githubusercontent.com/ShevchenkoRostyslav/Influencers-in-Social-Networks/master/lib/data/test.csv
```

* In browser:

http://127.0.0.1:5000/?input_file=https://raw.githubusercontent.com/ShevchenkoRostyslav/Influencers-in-Social-Networks/master/lib/data/test.csv

* run test/test_api.ipynb
