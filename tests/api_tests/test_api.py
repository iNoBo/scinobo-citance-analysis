""" 

This test script will be used to test the API endpoints for the SciNoBo-RAA.

"""
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #
from fastapi.testclient import TestClient
from citance_analysis.server.api import app

client = TestClient(app)

def test_infer_citance():
    request_data = {
        "citance": "We used the BERT model [15] to create our dataset. The results were very bad compared to other studies [12].",
        "citation_mark": "[15]"
    }
    response = client.post("/infer_citance", json=request_data)
    assert response.status_code == 200
    assert "citance" in response.json()
    assert "output" in response.json()

    request_data = {
        "citance": "We used the BERT model [15] to create our dataset. The results were very bad compared to other studies [12].",
    }
    response = client.post("/infer_citance", json=request_data)
    assert response.status_code == 200
    assert "citance" in response.json()
    assert "output" in response.json()

def test_infer_citance_invalid_request():
    request_data = {
        "citance": None,
    }
    response = client.post("/infer_citance", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "citance": "We used the BERT model [15] to create our dataset. The results were very bad compared to other studies [12].",
        "citation_mark": None
    }
    response = client.post("/infer_citance", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_infer_citance_empty_request():
    request_data = {
    }
    response = client.post("/infer_citance", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_infer_citances():
    request_data = {
        "citances":
            [
                {
                    "citance": "We used the BERT model [15] to create our dataset. The results were very bad compared to other studies [12]."
                },
                {
                    "citance": "The results from [30] seem just wrong.",
                    "citation_mark": "[30]"
                }
            ]
    }
    response = client.post("/infer_citances", json=request_data)
    assert response.status_code == 200
    assert "output" in response.json()

def test_infer_citances_empty_request():
    request_data = {
    }
    response = client.post("/infer_citances", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_infer_citances_empty_list_request():
    request_data = {
        "citances": []
    }
    response = client.post("/infer_citances", json=request_data)
    assert response.status_code == 200
    assert "output" in response.json()

def test_infer_citances_invalid_request():
    request_data = {
        "citances": None
    }
    response = client.post("/infer_citances", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()

    request_data = {
        "citances":
            [
                {
                    "citance": "We used the BERT model [15] to create our dataset. The results were very bad compared to other studies [12]."
                },
                {
                    "citance": None,
                    "citation_mark": "[30]"
                }
            ]
    }
    response = client.post("/infer_citances", json=request_data)
    assert response.status_code == 422
    assert "detail" in response.json()
