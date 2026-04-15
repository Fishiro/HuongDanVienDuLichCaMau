from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Dùng Hugging Face Inference API để nhúng (Đọc đúng file FAISS cũ, siêu nhẹ)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
CORS(app)

load_dotenv()
# Khai báo các khóa API từ file .env (ở local) hoặc Environment Variables (trên Vercel)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Khởi tạo model nhúng thông qua API đám mây của Hugging Face
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Nạp database tri thức cũ (Cam kết không bị lỗi chiều dữ liệu Vector)
vector_db = FAISS.load_local("camau_faiss_index", embeddings, allow_dangerous_deserialization=True)

SYSTEM_INSTRUCTIONS = {
    'vi': "Bạn là một hướng dẫn viên du lịch người bản địa tại Cà Mau. Trả lời thân thiện, ngắn gọn, cung cấp thông tin chính xác về đặc sản, văn hóa, và các điểm đến như Đất Mũi, Rừng U Minh Hạ.",
    'en': "You are a local tour guide in Ca Mau. Answer friendly and concisely, providing accurate information about specialties, culture, and destinations like Dat Mui, U Minh Ha Forest."
}

# Giữ nguyên route /api/chat để khớp với file vercel.json
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        language = data.get('language', 'vi') 
        
        if not user_message:
            return jsonify({'error': 'Vui lòng nhập câu hỏi'}), 400

        # Tra cứu tri thức từ FAISS
        related_docs = vector_db.similarity_search(user_message, k=3)
        context_text = "\n".join([doc.page_content for doc in related_docs])

        base_inst = SYSTEM_INSTRUCTIONS.get(language, SYSTEM_INSTRUCTIONS['vi'])
        full_sys_inst = f"{base_inst}\n\nDỮ LIỆU TRI THỨC BỔ SUNG:\n{context_text}\n\nHãy ưu tiên sử dụng dữ liệu tri thức trên."

        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=full_sys_inst,
            )
        )
        
        return jsonify({'reply': response.text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route kiểm tra trạng thái
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Cà Mau API is running'})

if __name__ == '__main__':
    app.run(debug=True)