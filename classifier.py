import pandas as pd
from sklearn import model_selection
import pickle
from sklearn.pipeline import Pipeline


class SpamClassifier(object):
    def __init__(self):
        df = pd.read_parquet("training.parquet")
        train, test = model_selection.train_test_split(df, random_state=43)
        X_train = train["text"]
        y_train = train["label"]

        ## loading in feature extraction pipeline
        filename = 'feature_pipeline.sav'
        feat_pipeline = pickle.load(open(filename, 'rb'))

        ## loading model
        filename = 'model.sav'
        model = pickle.load(open(filename, 'rb'))
        pipeline = Pipeline([
        ('features',feat_pipeline),
        ('model',model)
        ])
        pipeline.fit(X_train,y_train)
        self._pipeline = pipeline

    def predict(self, X):
        predictions = self._pipeline.predict(X)
        return predictions
