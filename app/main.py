import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from dotenv import load_dotenv
from loguru import logger

from .model import Model, load_model


load_dotenv()

app = FastAPI(
    title='Endpoint for ML communication',
    description='Here you got simple API interface to use your ML model',
    version='0.1.0'
)
app.mount("/static", StaticFiles(directory="static"),name="static")
templates = Jinja2Templates(directory='templates')
model = load_model('./models/model_v1.joblib')


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse(
        name = 'home.html',
        context = {'request':request}
    )

@app.get('/status')
def status():
    if not model:
        return HTTPException(status_code=404, detail='Model not loaded')
    return {'message':'Everything is set up'}


@app.post('/predict')
def predict_from_json(data: dict):
    try:
        X = pd.DataFrame.from_dict(data)
        y = list(model.predict(X))
    except ValueError as err:
        logger.error(f'Error. {err}')
    return {'results': y}
