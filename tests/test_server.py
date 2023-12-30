import pandas as pd

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_root():
    response = client.get('/')
    assert response.status_code == 200
    
def test_status():
    response = client.get('/status')
    assert response.status_code == 200
    
def test_model_predict_from_json():
    json = pd.read_csv('data/test_data.csv', index_col=0).to_json()
    response = client.post('/predict/', data=json)
    assert response.status_code == 200
    