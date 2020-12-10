import pytest
from flask_app.application import app
import requests


@pytest.fixture(scope='session')
def client():
    with app.test_client() as client:
        yield client


def test_http(client):
    response = client.get('/')
    assert response.status_code == 200


def test_main(client):
    response = client.get('/')
    assert 'Movie Recommender' in response.data.decode('utf-8')
