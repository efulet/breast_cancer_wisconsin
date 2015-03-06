"""
@created_at 2014-10-06
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


import os
import traceback
import sys
import datetime

from lib import *

import numpy as np
from sklearn import datasets, cross_validation, linear_model

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_version():
    """Python v2.7 es requerida por el curso, entonces verificamos la version"""
    if sys.version_info[:2] != (2, 7):
        raise Exception("Parece que python v2.7 no esta instalado en el sistema")

def db_path():
    """Retorna el path de las base de datos"""
    pathfile = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pathfile, "db")


if __name__ == "__main__":
    try:
        # Check python version
        check_version()
        
        # Load dataset
        # Diagnosis (M = malignant = 1, B = benign = 0)
        # @attribute 'class' {1, 0}
        # @attribute 'radius_mean' real
        # @attribute 'texture_mean' real
        # @attribute 'perimeter_mean' real
        # @attribute 'area_mean' real
        # @attribute 'smoothness_mean' real
        # @attribute 'compactness_mean' real
        # @attribute 'concavity_mean' real
        # @attribute 'concave_mean' real
        # @attribute 'symmetry_mean' real
        # @attribute 'fractal_mean' real
        # @attribute 'radius_se' real
        # @attribute 'texture_se' real
        # @attribute 'perimeter_se' real
        # @attribute 'area_se' real
        # @attribute 'smoothness_se' real
        # @attribute 'compactness_se' real
        # @attribute 'concavity_se' real
        # @attribute 'concave_se' real
        # @attribute 'symmetry_se' real
        # @attribute 'fractal_se' real
        # @attribute 'radius_worst' real
        # @attribute 'texture_worst' real
        # @attribute 'perimeter_worst' real
        # @attribute 'area_worst' real
        # @attribute 'smoothness_worst' real
        # @attribute 'compactness_worst' real
        # @attribute 'concavity_worst' real
        # @attribute 'concave_worst' real
        # @attribute 'symmetry_worst' real
        # @attribute 'fractal_worst' real
        X, y  = datasets.load_svmlight_file(open(os.path.join(db_path(), "wdbc.svm")))
        
        # Split the data into a training set and a test set. 
        # We can now quickly sample a training set while holding out 30% of the 
        # data for testing (evaluating) our classifier
        #X_train, y_train = datasets.load_svmlight_file(open(os.path.join(db_path(), "train_wdbc.svm")))
        #X_test, y_test = datasets.load_svmlight_file(open(os.path.join(db_path(), "test_wdbc.svm")))
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
        
        print "*" * 80
        print "Data set dimension: %d" % X.shape[1]
        print "N instances data set: %d" % X.shape[0]
        print "N instances training set: %d" % X_train.shape[0]
        print "N instances testing set: %d" % X_test.shape[0]
        print "*" * 80
        
        # Create a metrics instance for showing a report in the console
        fmetrics = FMetrics()
        
        # **********************************************************************
        # Modelo lineal
        # **********************************************************************
        print "*" * 80
        print "Modelo Lineal"
        print "*" * 80
        
        # Create linear regression instance
        classifier = linear_model.LinearRegression(fit_intercept=False, normalize=True)
        
        # Train the model using the training sets
        start = datetime.datetime.now()
        classifier.fit(X_train, y_train)
        print "Linear regression fit time:", str(datetime.datetime.now() - start)
        
        # Assign our beta_0 as b_0 and beta_n as coeff. Only save those variables so 
        # we can re-use them later in our manual verification against the 
        # simple linear model equation.
        #b_0   = classifier.intercept_
        #coeff = classifier.coef_
        #print("Coefficients: ", coeff)
        
        # Compute the prediction over the X_train
        y_train_pred = classifier.predict(X_train)
        y_train_pred = np.absolute(y_train_pred)
        y_train_pred[y_train_pred>=0.5] = 1
        y_train_pred[y_train_pred<0.5] = 0
        
        # Compute the prediction over the X_test
        start = datetime.datetime.now()
        y_pred = classifier.predict(X_test)
        print "Linear regression predict time:", str(datetime.datetime.now() - start)
        
        # First, convert decimal values to 0 or 1
        y_pred = np.absolute(y_pred)
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0
        
        # Create the report
        fmetrics.report(y_train, y_train_pred, y_test, y_pred)
        
        
        # **********************************************************************
        # Modelo rectangular
        # **********************************************************************
        print "*" * 80
        print "Modelo Rectangular"
        print "*" * 80
        
        # Create rectangular model instance
        rec_classifier = RectangularClassifier()
        
        # Train the model using the training sets
        start = datetime.datetime.now()
        rec_classifier.fit(X_train, y_train)
        print "Rectangular classifier fit time:", str(datetime.datetime.now() - start)
        
        # Compute the prediction over the X_train
        y_train_pred = rec_classifier.predict(X_train)
        
        # Compute the prediction over the X_test
        start = datetime.datetime.now()
        y_pred = rec_classifier.predict(X_test)
        print "Rectangular classifier predict time:", str(datetime.datetime.now() - start)
        
        # Create the report
        fmetrics.report(y_train, y_train_pred, y_test, y_pred)
    except Exception, err:
        print traceback.format_exc()
    finally:
        sys.exit()

