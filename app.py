from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np
import re
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification

app = Flask(__name__)  # Flask automatically serves from a /static folder

#########################################
# LOAD ALL MODELS
#########################################

# Traditional ML models
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
logistic_model = joblib.load('models/logictic_model.pkl')
random_forest_model = joblib.load('models/RandomForest_model.pkl')
gradient_boosting_model = joblib.load('models/GradientBoosting_model.pkl')

# BERT model
bert_tokenizer = BertTokenizerFast.from_pretrained('models/bert_tokenizer')
bert_model = TFBertForSequenceClassification.from_pretrained(
    'models/bert_model', 
    from_pt=False
)

#########################################
# TEXT PREPROCESSING
#########################################

def custom_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#########################################
# MODEL-SPECIFIC PREDICTION FUNCTIONS
#########################################

def get_prediction_logistic(text):
    processed = custom_preprocess(text)
    vec = vectorizer.transform([processed])
    pred = logistic_model.predict(vec)[0]
    conf = logistic_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_random_forest(text):
    processed = custom_preprocess(text)
    vec = vectorizer.transform([processed])
    pred = random_forest_model.predict(vec)[0]
    conf = random_forest_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_gradient_boosting(text):
    processed = custom_preprocess(text)
    vec = vectorizer.transform([processed])
    pred = gradient_boosting_model.predict(vec)[0]
    conf = gradient_boosting_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_bert(text, max_length=128):
    tokens = bert_tokenizer(
        [text],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    outputs = bert_model(tokens)
    logits = outputs.logits.numpy()[0]
    prob = tf.nn.softmax(logits).numpy()
    pred = np.argmax(prob)
    conf = np.max(prob)
    return pred, conf

#########################################
# FLASK ROUTES
#########################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')
    model_name = request.form.get('model')

    if not input_text or not model_name:
        return jsonify({'error': 'Text or model not provided'}), 400

    model_map = {
        'logistic': get_prediction_logistic,
        'random_forest': get_prediction_random_forest,
        'gradient_boosting': get_prediction_gradient_boosting,
        'bert': get_prediction_bert
    }

    if model_name not in model_map:
        return jsonify({'error': f'Model "{model_name}" not found'}), 400

    pred_func = model_map[model_name]
    prediction, confidence = pred_func(input_text)

    result = {
        'model': model_name,
        'prediction': 'Hateful' if prediction == 1 else 'Non-Hateful',
        'confidence': f'{confidence:.2f}'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
