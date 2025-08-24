# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from anomaly_detector import MultivariateTimeSeriesAnomalyDetector
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_file(file_path):
    """Load CSV or Excel file."""
    try:
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
        else:
            df = pd.read_excel(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and anomaly detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Please upload CSV or Excel files.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Get form parameters
        timestamp_col = request.form.get('timestamp_col', '').strip()
        exclude_cols = request.form.get('exclude_cols', '').strip()
        contamination = float(request.form.get('contamination', 0.1))
        normal_start = request.form.get('normal_start', '').strip()
        normal_end = request.form.get('normal_end', '').strip()

        if timestamp_col == '':
            timestamp_col = None

        # Convert empty strings to None
        if normal_start == '':
            normal_start = None
        if normal_end == '':
            normal_end = None
        
        exclude_cols_list = [col.strip() for col in exclude_cols.split(',') if col.strip()] if exclude_cols else []
        
        # Load data
        df, error = load_file(file_path)
        if error:
            return jsonify({'error': f'Error loading file: {error}'}), 400
        
        if len(df) < 5:
            return jsonify({'error': 'Dataset too small. Need at least 5 rows.'}), 400
        
        # Initialize detector
        detector = MultivariateTimeSeriesAnomalyDetector(contamination=contamination)
        
        # Run anomaly detection
        results_df = detector.fit_predict(
            df,
            timestamp_col=timestamp_col,
            exclude_cols=exclude_cols_list,
            normal_start=normal_start,
            normal_end=normal_end
        )
        
        # Save results
        output_filename = f"anomaly_results_{timestamp}.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        results_df.to_csv(output_path, index=False)
        
        # Optional: quick summary
        summary_stats = {
            "total_rows": len(results_df),
            "anomalies_detected": int(results_df["is_anomaly"].sum()),
            "anomaly_rate": round((results_df["is_anomaly"].mean() * 100), 2)
        }
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'summary': summary_stats,
            'preview': results_df.head(5).to_dict('records'),
            'columns': list(results_df.columns)
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download results file."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
