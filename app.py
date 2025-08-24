from flask import Flask, request, render_template, send_file
import pandas as pd
from io import BytesIO
from anomaly_detector import MultivariateTimeSeriesAnomalyDetector  # your class in separate file
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read file
        if file.filename.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file.filename.endswith(".xlsx"):
            data = pd.read_excel(file_path, sheet_name="Sheet4")
        else:
            return "Invalid file type. Upload CSV or XLSX.", 400

        # Run anomaly detection
        detector = MultivariateTimeSeriesAnomalyDetector(contamination=0.1)
        result = detector.fit_predict(data, timestamp_col="timestamp")

        # Prepare CSV output in memory
        output = BytesIO()
        result.to_csv(output, index=False)
        output.seek(0)

        return send_file(output,
                         mimetype="text/csv",
                         download_name="anomaly_results.csv",
                         as_attachment=True)
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
