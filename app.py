import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from anomaly_detector import MultivariateTimeSeriesAnomalyDetector  # import your class

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file is in request
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                # Load file
                if filename.endswith(".csv"):
                    data = pd.read_csv(filepath)
                else:
                    data = pd.read_excel(filepath, sheet_name="Sheet4")

                # Run anomaly detection
                detector = MultivariateTimeSeriesAnomalyDetector(contamination=0.1)
                result = detector.fit_predict(data, timestamp_col="timestamp")

                # Save results
                result_file = os.path.join(app.config["RESULT_FOLDER"], "anomaly_results.csv")
                result.to_csv(result_file, index=False)

                return send_file(result_file, as_attachment=True)

            except Exception as e:
                flash(f"Error processing file: {e}")
                return redirect(request.url)
        else:
            flash("Allowed file types are CSV or XLSX")
            return redirect(request.url)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
