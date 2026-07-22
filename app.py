from flask import Flask, jsonify
import run_predict

app = Flask(__name__)

# เพิ่ม Route หน้าแรกเพื่อใช้ทดสอบว่าเซิร์ฟเวอร์รันติดหรือไม่
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running! Go to /predict to get the forecast."})

@app.route('/predict', methods=['GET'])
def get_prediction():
    # เรียกใช้ฟังก์ชันทำนายผล
    results = run_predict.generate_forecast() 
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
