from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import re
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification

app = FastAPI()

# Serve static files (JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

#########################################
# LOAD ALL MODELS ON STARTUP
#########################################

@app.on_event("startup")
async def load_models():
    app.state.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    app.state.logistic_model = joblib.load('models/logictic_model.pkl')
    app.state.random_forest_model = joblib.load('models/RandomForest_model.pkl')
    app.state.gradient_boosting_model = joblib.load('models/GradientBoosting_model.pkl')
    app.state.bert_tokenizer = BertTokenizerFast.from_pretrained('models/bert_tokenizer')
    app.state.bert_model = TFBertForSequenceClassification.from_pretrained(
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

def get_prediction_logistic(text, app):
    processed = custom_preprocess(text)
    vec = app.state.vectorizer.transform([processed])
    pred = app.state.logistic_model.predict(vec)[0]
    conf = app.state.logistic_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_random_forest(text, app):
    processed = custom_preprocess(text)
    vec = app.state.vectorizer.transform([processed])
    pred = app.state.random_forest_model.predict(vec)[0]
    conf = app.state.random_forest_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_gradient_boosting(text, app):
    processed = custom_preprocess(text)
    vec = app.state.vectorizer.transform([processed])
    pred = app.state.gradient_boosting_model.predict(vec)[0]
    conf = app.state.gradient_boosting_model.predict_proba(vec).max()
    return pred, conf

def get_prediction_bert(text, app, max_length=128):
    tokens = app.state.bert_tokenizer(
        [text],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    outputs = app.state.bert_model(tokens)
    logits = outputs.logits.numpy()[0]
    prob = tf.nn.softmax(logits).numpy()
    pred = np.argmax(prob)
    conf = np.max(prob)
    return pred, conf

#########################################
# FASTAPI ROUTES
#########################################

class PredictRequest(BaseModel):
    text: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, text: str = Form(...), model: str = Form(...)):
    if not text or not model:
        raise HTTPException(status_code=400, detail="Text or model not provided")

    model_map = {
        'logistic': get_prediction_logistic,
        'random_forest': get_prediction_random_forest,
        'gradient_boosting': get_prediction_gradient_boosting,
        'bert': get_prediction_bert
    }

    if model not in model_map:
        raise HTTPException(status_code=400, detail=f'Model \"{model}\" not found')

    pred_func = model_map[model]
    prediction, confidence = pred_func(text, request.app)

    result = {
        'model': model,
        'prediction': 'Hateful' if prediction == 1 else 'Non-Hateful',
        'confidence': f'{confidence:.2f}'
    }

    return JSONResponse(content=result)

@app.post("/predict_json")
async def predict_json(request: Request, payload: PredictRequest):
    text = payload.text
    model = payload.model

    model_map = {
        'logistic': get_prediction_logistic,
        'random_forest': get_prediction_random_forest,
        'gradient_boosting': get_prediction_gradient_boosting,
        'bert': get_prediction_bert
    }

    if model not in model_map:
        raise HTTPException(status_code=400, detail=f'Model \"{model}\" not found')

    pred_func = model_map[model]
    prediction, confidence = pred_func(text, request.app)

    result = {
        'model': model,
        'prediction': 'Hateful' if prediction == 1 else 'Non-Hateful',
        'confidence': f'{confidence:.2f}'
    }

    return JSONResponse(content=result)

# To run: uvicorn app:app --reload
