from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from sys import version_info

# --- THÊM PHẦN IMPORT CHO RAG ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print(version_info)

app = Flask(__name__)
CORS(app)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- THÊM PHẦN NẠP DỮ LIỆU TRI THỨC FAISS ---
# Sử dụng đúng model local bạn đã dùng để train
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
# Nạp "não bộ" từ thư mục đã tạo, cho phép giải tuần tự hóa an toàn
vector_db = FAISS.load_local("camau_faiss_index", embeddings, allow_dangerous_deserialization=True)

SYSTEM_INSTRUCTIONS = {
    'vi': "Bạn là một hướng dẫn viên du lịch người bản địa tại Cà Mau. Trả lời thân thiện, ngắn gọn, cung cấp thông tin chính xác về đặc sản, văn hóa, và các điểm đến như Đất Mũi, Rừng U Minh Hạ.",
    'en': "You are a local tour guide in Ca Mau, Vietnam. Answer in a friendly, concise manner in English. Provide accurate information about local specialties, culture, and destinations like Mui Ca Mau, U Minh Ha Forest."
}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        language = data.get('language', 'vi') 
        
        if not user_message:
            return jsonify({'error': 'Vui lòng nhập câu hỏi / Please enter a question'}), 400

        # --- BƯỚC THÊM: TRA CỨU TRI THỨC TỪ FAISS ---
        # Tìm 3 đoạn thông tin liên quan nhất đến câu hỏi
        related_docs = vector_db.similarity_search(user_message, k=3)
        context_text = "\n".join([doc.page_content for doc in related_docs])

        # Lấy instruction gốc
        base_inst = SYSTEM_INSTRUCTIONS.get(language, SYSTEM_INSTRUCTIONS['vi'])
        
        # Nhúng thêm tri thức vào Instruction để AI không quên "gốc" hướng dẫn viên
        full_sys_inst = f"{base_inst}\n\nDỮ LIỆU TRI THỨC BỔ SUNG:\n{context_text}\n\nHãy ưu tiên sử dụng dữ liệu tri thức trên để trả lời khách hàng một cách chính xác nhất."

        # Gọi API Gemini với instruction đã được làm giàu tri thức
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_sys_inst, # Gửi instruction mới chứa RAG
            )
        )
        
        return jsonify({'reply': response.text})
    
    except Exception as e:
        print("LỖI:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Máy chủ AI Cà Mau (RAG Enabled) đang chạy tại http://localhost:5000")
    app.run(debug=True, port=5000)