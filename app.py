from flask import Flask, jsonify
import run_predict # นำเข้าโค้ดทำนายของคุณ

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_prediction():
    # เรียกใช้ฟังก์ชันในไฟล์ run_predict.py ของคุณ
    results = run_predict.generate_forecast() 
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)