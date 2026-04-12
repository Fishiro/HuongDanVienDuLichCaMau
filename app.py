from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from sys import version_info
print(version_info)

app = Flask(__name__)
# Cho phép Frontend ở localhost khác port gọi được vào Backend này
CORS(app)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Cú pháp MỚI: Khởi tạo Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Cấu hình hướng dẫn cho AI theo 2 ngôn ngữ
SYSTEM_INSTRUCTIONS = {
    'vi': "Bạn là một hướng dẫn viên du lịch người bản địa tại Cà Mau. Trả lời thân thiện, ngắn gọn, cung cấp thông tin chính xác về đặc sản, văn hóa, và các điểm đến như Đất Mũi, Rừng U Minh Hạ.",
    'en': "You are a local tour guide in Ca Mau, Vietnam. Answer in a friendly, concise manner in English. Provide accurate information about local specialties, culture, and destinations like Mui Ca Mau, U Minh Ha Forest."
}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        # Nhận ngôn ngữ từ Frontend truyền xuống (mặc định là 'vi' nếu không có)
        language = data.get('language', 'vi') 
        
        if not user_message:
            return jsonify({'error': 'Vui lòng nhập câu hỏi / Please enter a question'}), 400

        # Lấy instruction tương ứng với ngôn ngữ đã chọn
        sys_inst = SYSTEM_INSTRUCTIONS.get(language, SYSTEM_INSTRUCTIONS['vi'])

        # Gọi API Gemini bằng cú pháp MỚI
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
            )
        )
        
        return jsonify({'reply': response.text})
    
    except Exception as e:
        print("LỖI:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy server ở port 5000
    print("🚀 Máy chủ AI Cà Mau đang chạy tại http://localhost:5000")
    app.run(debug=True, port=5000)