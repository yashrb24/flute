from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.python.training.adam import AdamOptimizer
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import numpy as np

def random_forest_classifier(X_train, y_train, X_test, y_test):
    # Create the model and train it
    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(X_train, y_train)

    # Test the model
    y_pred = rf_model.predict(X_test)

    # Get the report, handle zero-division
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def svm_classifier(X_train, y_train, X_test, y_test):
    # Create the model and train it
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)

    # Test the model
    y_pred = svm_model.predict(X_test)

    # Get the report, handle zero-division
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def knn_classifier(X_train, y_train, X_test, y_test):
    # Create the model and train it
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    # Test the model
    y_pred = knn_model.predict(X_test)

    # Get the report, handle zero-division
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def xgboost_classifier(X_train, y_train, X_test, y_test):
    # Create the model and train it
    xgb_model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)

    # Test the model
    y_pred = xgb_model.predict(X_test)

    # Get the report, handle zero-division
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report


def mlp_classifier(X_train, y_train, X_test, y_test):
    # Create the model and train it
    mlp_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')
    ])
    mlp_model.compile(optimizer=AdamOptimizer(learning_rate=1e-3), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    print("MLP Training....")
    # Train the model
    mlp_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.1, verbose=False)
    print("MLP Training Done")
    
    # Test the model
    y_pred = mlp_model.predict(X_test, verbose=False)
    y_pred = np.argmax(y_pred, axis=1)
    
    
    # Get the report, handle zero-division
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return report
