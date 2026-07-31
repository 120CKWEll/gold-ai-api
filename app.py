import os
from flask import Flask, jsonify
from flask_cors import CORS
import run_predict
import traceback

app = Flask(__name__)
CORS(app)

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
    # 🟢 ดึงค่า Port จากระบบของ Render ถ้าไม่มีให้ใช้ 5000 แทน
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
