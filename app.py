import re
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import pandas as pd
import os
import uuid
import threading
from werkzeug.utils import secure_filename
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.0-flash"  # or "gemini-1.0-pro" if unavailable

# In-memory storage
reports = {}
chats = {}

def clean_gemini_response(text):
    """
    Cleans Gemini API response text by converting markdown to HTML tags.
    Handles all common markdown syntax including headers, bold, italics, tables, etc.
    
    Args:
        text (str): The raw response text from Gemini API
        
    Returns:
        str: Text with proper HTML formatting
    """
    # Headers
    cleaned_text = re.sub(r'^#\s+(.*)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^##\s+(.*)$', r'<h2>\1</h2>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^###\s+(.*)$', r'<h3>\1</h3>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^####\s+(.*)$', r'<h4>\1</h4>', cleaned_text, flags=re.MULTILINE)
    
    # Bold and italics
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cleaned_text)
    cleaned_text = re.sub(r'__(.*?)__', r'<b>\1</b>', cleaned_text)
    cleaned_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cleaned_text)
    cleaned_text = re.sub(r'_(.*?)_', r'<i>\1</i>', cleaned_text)
    
    # Code blocks
    cleaned_text = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'`(.*?)`', r'<code>\1</code>', cleaned_text)
    
    # Lists
    cleaned_text = re.sub(r'^\*\s+(.*)$', r'<li>\1</li>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^-\s+(.*)$', r'<li>\1</li>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^(\s*<li>.*<\/li>\s*)+$', r'<ul>\g<0></ul>', cleaned_text, flags=re.MULTILINE)
    
    # Tables
    def process_tables(md_text):
        table_pattern = re.compile(
            r'^(\|?.+\|.+\n\|?[\s]*[-:]+[-|\s:]+\n)((?:\|?.+\|.+\n)*)',
            re.MULTILINE
        )
        
        def table_replacer(match):
            full_table = match.group(0)
            lines = [line.strip() for line in full_table.split('\n') if line.strip()]
            
            # Skip if doesn't look like a proper table
            if len(lines) < 2:
                return full_table
                
            # Process header
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]
            
            # Process alignment row (second line)
            alignments = []
            alignment_line = lines[1].split('|')
            for cell in alignment_line:
                cell = cell.strip()
                if cell.startswith(':') and cell.endswith(':'):
                    alignments.append('center')
                elif cell.startswith(':'):
                    alignments.append('left')
                elif cell.endswith(':'):
                    alignments.append('right')
                else:
                    alignments.append('left')
            
            # Build HTML table
            html_table = ['<table class="markdown-table">']
            
            # Add header
            html_table.append('<thead><tr>')
            for i, header in enumerate(headers):
                align = f' style="text-align:{alignments[i]}"' if i < len(alignments) else ''
                html_table.append(f'<th{align}>{header}</th>')
            html_table.append('</tr></thead>')
            
            # Add body rows
            html_table.append('<tbody>')
            for line in lines[2:]:
                if not line.strip() or all(c in ['-', '|', ':'] for c in line.replace(' ', '')):
                    continue
                cells = [c.strip() for c in line.split('|') if c.strip()]
                html_table.append('<tr>')
                for i, cell in enumerate(cells):
                    align = f' style="text-align:{alignments[i]}"' if i < len(alignments) else ''
                    html_table.append(f'<td{align}>{cell}</td>')
                html_table.append('</tr>')
            html_table.append('</tbody></table>')
            
            return '\n'.join(html_table)
        
        return table_pattern.sub(table_replacer, md_text)
    
    # Process tables first
    cleaned_text = process_tables(text)
    
    # Then process other markdown elements
    # Headers
    cleaned_text = re.sub(r'^#\s+(.*)$', r'<h1>\1</h1>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^##\s+(.*)$', r'<h2>\1</h2>', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^###\s+(.*)$', r'<h3>\1</h3>', cleaned_text, flags=re.MULTILINE)
    
    # Bold/Italic
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cleaned_text)
    cleaned_text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', cleaned_text)
    cleaned_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cleaned_text)
    cleaned_text = re.sub(r'_(.*?)_', r'<em>\1</em>', cleaned_text)
    
    # Code
    cleaned_text = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code class="\1">\2</code></pre>', 
cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'`(.*?)`', r'<code>\1</code>', cleaned_text)
    
    # Lists
    cleaned_text = re.sub(r'^(\s*)[-*+]\s+(.*)$', lambda m: f'<ul><li>{"&nbsp;"*len(m.group(1))}{m.group(2)}</li></ul>', cleaned_text, flags=re.MULTILINE)
    
    # Horizontal rule
    cleaned_text = re.sub(r'^[-*_]{3,}$', r'<hr>', cleaned_text, flags=re.MULTILINE)
    
    # Links and images
    cleaned_text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1">', cleaned_text)
    cleaned_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', cleaned_text)
    
    # Blockquotes
    cleaned_text = re.sub(r'^>\s*(.*)$', r'<blockquote>\1</blockquote>', cleaned_text, flags=re.MULTILINE)
    
    # Clean up nested lists
    cleaned_text = re.sub(r'<\/ul>\s*<ul>', '', cleaned_text)
    
    # Remove empty paragraphs
    cleaned_text = re.sub(r'<p>\s*<\/p>', '', cleaned_text)
    
    return cleaned_text


class IncidentReporter:
    @staticmethod
    def parse_csv(file_path):
        df = pd.read_csv(file_path)
        incidents = {}
        
        for _, row in df.iterrows():
            incident_id = row['incident_id']
            if incident_id not in incidents:
                incidents[incident_id] = {
                    'incident_id': incident_id,
                    'root_cause': row['root_cause'],
                    'affected_services': row['affected_services'],
                    'impact': row['impact'],
                    'severity': row.get('severity', 'medium'),
                    'status': row.get('status', 'open'),
                    'logs': []
                }
            
            try:
                timestamp = datetime.strptime(str(row['timestamp']).strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    timestamp = datetime.strptime(str(row['timestamp']).strip(), '%d-%m-%Y %H:%M')
                except ValueError:
                    timestamp = datetime.now()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'component': row['component'],
                'log_level': row['log_level'].upper(),
                'message': row['message']
            }
            incidents[incident_id]['logs'].append(log_entry)
        
        return incidents

    @staticmethod
    def generate_report(incident_data):
        prompt = f"""
        Generate a comprehensive incident report with these sections:
        1. Incident Summary (ID, Severity, Status)
        2. Detailed Timeline (Chronological events)
        3. Root Cause Analysis
        4. Impact Assessment (Affected services, duration)
        5. Resolution Steps
        6. Preventive Recommendations

        Incident Details:
        - ID: {incident_data['incident_id']}
        - Severity: {incident_data.get('severity', 'medium')}
        - Status: {incident_data.get('status', 'open')}
        - Root Cause: {incident_data['root_cause']}
        - Affected Services: {incident_data['affected_services']}
        - Impact: {incident_data['impact']}

        Log Entries:
        {"".join(
            f"{log['timestamp']} [{log['log_level']}] {log['component']}: {log['message']}"
            for log in sorted(incident_data['logs'], key=lambda x: x['timestamp'])
        )}

        Provide the report in markdown format with clear section headings.
        Include a timeline visualization suggestion at the top.
        """
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        return clean_gemini_response(response.text)

    @staticmethod
    def generate_chat_response(question, incident_id, chat_history):
        prompt = f"""
        You are an incident report assistant. Here's the conversation history:
        {chat_history}

        Current question: {question}
        
        Provide a concise, helpful answer focusing on the incident details.
        If asked about timeline or sequence of events, mention that a visual timeline is available.
        """
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return clean_gemini_response(response.text)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        incidents = IncidentReporter.parse_csv(filepath)
        session_id = str(uuid.uuid4())
        reports[session_id] = {
            'incidents': incidents, 
            'generated_reports': {},
            'upload_time': datetime.now().isoformat()
        }
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'incident_ids': list(incidents.keys()),
            'total_incidents': len(incidents)
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    session_id = data.get('session_id')
    incident_id = data.get('incident_id')
    
    if session_id not in reports or incident_id not in reports[session_id]['incidents']:
        return jsonify({'error': 'Invalid session or incident ID'}), 400
    
    def generate_in_background():
        incident_data = reports[session_id]['incidents'][incident_id]
        report = IncidentReporter.generate_report(incident_data)
        reports[session_id]['generated_reports'][incident_id] = {
            'report': report,
            'generated_at': datetime.now().isoformat()
        }
        
        if session_id not in chats:
            chats[session_id] = {}
        if incident_id not in chats[session_id]:
            chats[session_id][incident_id] = [
                {
                    "role": "assistant", 
                    "content": f"Hello! I'm your assistant for Incident {incident_id}. How can I help?",
                    "timestamp": datetime.now().isoformat()
                }
            ]
    
    thread = threading.Thread(target=generate_in_background)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Report generation started'})

@app.route('/get_report', methods=['GET'])
def get_report():
    session_id = request.args.get('session_id')
    incident_id = request.args.get('incident_id')
    
    if session_id not in reports or incident_id not in reports[session_id]['incidents']:
        return jsonify({'error': 'Invalid session or incident ID'}), 400
    
    report_data = reports[session_id]['generated_reports'].get(incident_id)
    if not report_data:
        return jsonify({'status': 'pending'})
    
    return jsonify({
        'status': 'completed',
        'report': report_data['report'],
        'generated_at': report_data['generated_at']
    })

@app.route('/get_logs_for_timeline', methods=['GET'])
def get_logs_for_timeline():
    session_id = request.args.get('session_id')
    incident_id = request.args.get('incident_id')
    
    if session_id not in reports or incident_id not in reports[session_id]['incidents']:
        return jsonify({'error': 'Invalid session or incident ID'}), 400
    
    incident_data = reports[session_id]['incidents'][incident_id]
    logs = incident_data['logs']
    
    # Sort logs by timestamp and format for frontend
    sorted_logs = sorted(logs, key=lambda x: x['timestamp'])
    formatted_logs = []
    
    for log in sorted_logs:
        formatted_logs.append({
            'timestamp': log['timestamp'],
            'component': log['component'],
            'log_level': log['log_level'],
            'message': log['message'],
            'color': get_log_level_color(log['log_level'])
        })
    
    return jsonify({
        'success': True,
        'logs': formatted_logs,
        'incident_id': incident_id,
        'root_cause': incident_data['root_cause']
    })

def get_log_level_color(log_level):
    colors = {
        'DEBUG': '#5b8ff9',
        'INFO': '#5ad8a6',
        'WARNING': '#f6bd16',
        'ERROR': '#e8684a',
        'CRITICAL': '#a01c5a'
    }
    return colors.get(log_level.upper(), '#999999')

@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.json
    session_id = data.get('session_id')
    incident_id = data.get('incident_id')
    message = data.get('message')
    
    if not all([session_id, incident_id, message]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if session_id not in chats or incident_id not in chats[session_id]:
        return jsonify({'error': 'Invalid session or incident ID'}), 400
    
    # Add user message
    chats[session_id][incident_id].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Prepare prompt with context
    report_content = reports[session_id]['generated_reports'].get(incident_id, {}).get('report', "")
    chat_history = "\n".join(
        f"{msg['role']} ({msg['timestamp']}): {msg['content']}" 
        for msg in chats[session_id][incident_id]
    )
    
    try:
        response = IncidentReporter.generate_chat_response(
            message,
            incident_id,
            f"Report Content:\n{report_content}\n\n{chat_history}"
        )
        
        # Add assistant response
        chats[session_id][incident_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id')
    incident_id = request.args.get('incident_id')
    
    if session_id not in chats or incident_id not in chats[session_id]:
        return jsonify({'error': 'Invalid session or incident ID'}), 400
    
    return jsonify({
        'success': True,
        'messages': chats[session_id][incident_id]
    })

@app.route('/api/kpi/dashboard', methods=['GET'])
def get_kpi_dashboard():
    session_id = request.args.get('session_id')
    
    if session_id not in reports:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    incidents = reports[session_id]['incidents']
    
    # Calculate KPIs
    total_incidents = len(incidents)
    
    resolution_times = []
    detection_times = []
    root_causes = {}
    severity_counts = {
        'critical': 0,
        'high': 0,
        'medium': 0,
        'low': 0
    }
    
    for incident_id, incident_data in incidents.items():
        logs = incident_data['logs']
        
        if not logs:
            continue
            
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x['timestamp'])
        
        # Get timestamps
        first_log_time = datetime.fromisoformat(sorted_logs[0]['timestamp'])
        last_log_time = datetime.fromisoformat(sorted_logs[-1]['timestamp'])
        
        # Calculate detection time (time from first log to first ERROR log)
        error_logs = [log for log in sorted_logs if log['log_level'].upper() in ['ERROR', 'CRITICAL']]
        if error_logs:
            first_error_time = datetime.fromisoformat(error_logs[0]['timestamp'])
            detection_time = (first_error_time - first_log_time).total_seconds() / 60  # in minutes
            detection_times.append(detection_time)
        
        # Calculate resolution time (time from first log to last log)
        resolution_time = (last_log_time - first_log_time).total_seconds() / 60  # in minutes
        resolution_times.append(resolution_time)
        
        # Count root causes
        root_cause = incident_data['root_cause']
        root_causes[root_cause] = root_causes.get(root_cause, 0) + 1
        
        # Count severity
        severity = incident_data.get('severity', 'medium').lower()
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    # Calculate averages
    avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    avg_detection = sum(detection_times) / len(detection_times) if detection_times else 0
    
    # Get top 5 root causes
    sorted_root_causes = sorted(root_causes.items(), key=lambda x: x[1], reverse=True)[:5]
    top_root_causes = [{'cause': cause, 'count': count} for cause, count in sorted_root_causes]
    
    return jsonify({
        'total_incidents': total_incidents,
        'avg_resolution_time_minutes': round(avg_resolution, 2),
        'avg_detection_time_minutes': round(avg_detection, 2),
        'top_root_causes': top_root_causes,
        'severity_distribution': severity_counts,
        'mttd_minutes': round(avg_detection, 2) if detection_times else 0,
        'mttr_minutes': round(avg_resolution, 2) if resolution_times else 0,
        'session_created': reports[session_id]['upload_time']
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)