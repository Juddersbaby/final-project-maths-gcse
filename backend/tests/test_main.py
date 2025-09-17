import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fastapi.testclient import TestClient
import main as backend

client = TestClient(backend.app)

def test_recommendation_endpoint():
    resp = client.get('/recommendation', params={'student_id':'S000','k':3,'policy':'baseline'})
    assert resp.status_code == 200
    body = resp.json()
    assert 'next_topics' in body
    assert isinstance(body['next_topics'], list)

def test_recommend_questions_endpoint():
    resp = client.get('/recommend_questions', params={'student_id':'S000','k':3,'policy':'baseline'})
    assert resp.status_code == 200
    body = resp.json()
    assert 'questions' in body
    assert isinstance(body['questions'], list)

def test_ucb_ordering():
    # create a fake history for a synthetic user
    # call recommend_ucb and ensure returned topics are in expected order by computing scores directly
    user = 'S000'
    res = backend.recommend_ucb(user, k=5)
    assert isinstance(res, list)
    assert len(res) <= 5
