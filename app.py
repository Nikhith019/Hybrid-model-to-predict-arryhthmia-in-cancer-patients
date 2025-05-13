from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

app = Flask(__name__)

# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload dataset and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load the dataset and perform machine learning tasks
    dataset = pd.read_csv(file)

    # Extract features and labels
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Apply LDA
    lda = LDA(n_components=1)
    X = lda.fit_transform(X, y)

    # Logistic Regression
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X, y)
    y_pred_log_reg = log_reg_model.predict(X)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X, y)
    y_pred_knn = knn_model.predict(X)

    # Calculate accuracy
    accuracy_log_reg = accuracy_score(y, y_pred_log_reg)
    accuracy_knn = accuracy_score(y, y_pred_knn)

    # Confusion Matrix
    cm_log_reg = confusion_matrix(y, y_pred_log_reg).tolist()
    cm_knn = confusion_matrix(y, y_pred_knn).tolist()

    # Return results as JSON with formatted accuracy
    return jsonify({
        'accuracy_log_reg': f"{accuracy_log_reg:.4f}",  # Formatting to 4 decimal places
        'accuracy_knn': f"{accuracy_knn:.4f}",
        'cm_log_reg': cm_log_reg,
        'cm_knn': cm_knn
    })

if __name__ == '__main__':
    app.run(debug=True)
