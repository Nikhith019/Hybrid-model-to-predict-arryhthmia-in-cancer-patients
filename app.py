from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File name is empty'})

    try:
        dataset = pd.read_csv(file)

        # Basic validation
        if dataset.shape[1] < 2:
            return jsonify({'error': 'Dataset must have at least 1 feature and 1 label column'})

        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        # Handle binary classification labels if not in 0/1 format
        y = pd.factorize(y)[0]

        # Feature Scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)

        # Apply LDA (optional dimensionality reduction)
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

        # Logistic Regression
        log_reg_model = LogisticRegression()
        log_reg_model.fit(X_train, y_train)
        y_pred_log_reg = log_reg_model.predict(X_test)

        # KNN
        knn_model = KNeighborsClassifier(n_neighbors=7)
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)

        # Neural Network
        X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_nn_train, y_nn_train, epochs=20, batch_size=8, verbose=0)

        nn_loss, nn_accuracy = model.evaluate(X_nn_test, y_nn_test, verbose=0)
        y_pred_nn = (model.predict(X_nn_test) > 0.5).astype(int).flatten()
        cm_nn = confusion_matrix(y_nn_test, y_pred_nn).tolist()

        # Accuracies
        accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)

        # Confusion Matrices
        cm_log_reg = confusion_matrix(y_test, y_pred_log_reg).tolist()
        cm_knn = confusion_matrix(y_test, y_pred_knn).tolist()

        return jsonify({
            'accuracy_log_reg': f"{accuracy_log_reg:.4f}",
            'accuracy_knn': f"{accuracy_knn:.4f}",
            'accuracy_nn': f"{nn_accuracy:.4f}",
            'cm_log_reg': cm_log_reg,
            'cm_knn': cm_knn,
            'cm_nn': cm_nn
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
