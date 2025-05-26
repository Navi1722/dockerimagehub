import io
import os
import json
import pytest
from app import app, IncidentReporter

# Test CSV file path
TEST_CSV_PATH = r"E:\genai-cat project\resources\test.csv"

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test GET / returns 200 and correct content-type."""
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'text/html' in resp.content_type

def test_upload_file_no_file(client):
    """POST /upload with no file should return error 400."""
    resp = client.post('/upload', data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'No file uploaded'

def test_upload_file_wrong_extension(client):
    """POST /upload with a non-CSV file should return error 400."""
    data = {
        'file': (io.BytesIO(b"not,a,csv"), 'test.txt')
    }
    resp = client.post('/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'Only CSV files are allowed'

def test_upload_file_success(client):
    """POST /upload with a valid CSV file should return session info."""
    with open(TEST_CSV_PATH, 'rb') as f:
        data = {
            'file': (f, 'test.csv')
        }
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert 'session_id' in data
    assert isinstance(data['incident_ids'], list)
    assert isinstance(data['total_incidents'], int)
    # Store session_id for next tests if needed
    return data['session_id'], data['incident_ids']

@pytest.mark.skip(reason="Background thread; test only basic functionality here")
def test_generate_report_endpoint(client):
    """Test POST /generate_report with valid session and incident ID."""
    # Upload first to get session and incident ID
    with open(TEST_CSV_PATH, 'rb') as f:
        data = {'file': (f, 'test.csv')}
        upload_resp = client.post('/upload', data=data, content_type='multipart/form-data')
    upload_data = upload_resp.get_json()
    session_id = upload_data['session_id']
    incident_ids = upload_data['incident_ids']
    incident_id = incident_ids[0]

    # Call generate_report endpoint
    post_data = {
        'session_id': session_id,
        'incident_id': incident_id
    }
    resp = client.post('/generate_report', json=post_data)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert 'Report generation started' in data['message']

def test_generate_report_invalid_session(client):
    """POST /generate_report with invalid session should return 400."""
    resp = client.post('/generate_report', json={
        'session_id': 'invalid_session',
        'incident_id': '123'
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data

def test_parse_csv_function():
    """Test the CSV parsing logic with the test CSV file."""
    incidents = IncidentReporter.parse_csv(TEST_CSV_PATH)
    assert isinstance(incidents, dict)
    for incident_id, data in incidents.items():
        assert 'incident_id' in data
        assert 'logs' in data
        assert isinstance(data['logs'], list)
        # Check a log entry
        if data['logs']:
            log = data['logs'][0]
            assert 'timestamp' in log
            assert 'component' in log
            assert 'log_level' in log
            assert 'message' in log

def test_clean_gemini_response_simple():
    """Test that markdown headers and bold/italic are converted to HTML."""
    from app import clean_gemini_response
    md_text = "# Header\n## Subheader\n**bold** and *italic*"
    html = clean_gemini_response(md_text)
    assert '<h1>Header</h1>' in html
    assert '<h2>Subheader</h2>' in html
    assert ('<b>bold</b>' in html or '<strong>bold</strong>' in html)
    assert ('<i>italic</i>' in html or '<em>italic</em>' in html)

# If you want to test chat endpoints or the get_report endpoint,
# you'd need to simulate the report generation or mock the model calls,
# which is complex for unit tests and may require mocking genai calls.

