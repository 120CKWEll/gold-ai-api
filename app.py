from flask import Flask, jsonify
from flask_cors import CORS  # 🟢 เปลี่ยนเป็นตัวพิมพ์ใหญ่ CORS ตรงนี้
import run_predict
import traceback

app = Flask(__name__)
CORS(app)  # 🟢 เปลี่ยนเป็นตัวพิมพ์ใหญ่ CORS ตรงนี้

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running! Go to /predict to get the forecast."})

@app.route('/predict', methods=['GET'])
def get_prediction():
    try:
        results = run_predict.generate_forecast() 
        return jsonify(results)
    except Exception as e:
        print("Error during prediction:", str(e))
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
