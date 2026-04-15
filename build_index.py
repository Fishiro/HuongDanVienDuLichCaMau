import os
import pandas as pd
import glob
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

data_folder = "rag_data"
all_docs = []

print("1. Đang gom dữ liệu từ các file CSV và Excel...")
# Lấy danh sách cả file .csv và .xlsx
all_files = glob.glob(os.path.join(data_folder, "*.csv")) + glob.glob(os.path.join(data_folder, "*.xlsx"))

for file in all_files:
    try:
        ext = os.path.splitext(file)[1].lower()
        
        # Xử lý file CSV
        if ext == '.csv':
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
                if row_text.strip():
                    all_docs.append(Document(page_content=row_text, metadata={"source": os.path.basename(file)}))
            print(f"✔️ Đã đọc xong CSV: {os.path.basename(file)}")
            
        # Xử lý file Excel
        elif ext == '.xlsx':
            xls = pd.read_excel(file, sheet_name=None) # Đọc toàn bộ các sheet
            for sheet_name, df in xls.items():
                for index, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
                    if row_text.strip():
                        # Ghi chú rõ nguồn là từ file nào, sheet nào
                        all_docs.append(Document(page_content=row_text, metadata={"source": f"{os.path.basename(file)} - {sheet_name}"}))
            print(f"✔️ Đã đọc xong Excel: {os.path.basename(file)}")
            
    except Exception as e:
        print(f"❌ Lỗi khi đọc {file}: {e}")

print(f"\nTổng cộng thu thập được {len(all_docs)} dòng dữ liệu.")

print("\n2. Đang chia nhỏ văn bản (Chunking)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_docs)
print(f" -> Tổng cộng đã chia thành {len(chunks)} đoạn nhỏ.")

print("\n3. Đang tải/chạy mô hình AI Local...")
# Chạy nội bộ, không tốn Quota, không cần sleep!
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("\n4. Đang tạo Vector DB...")
vectorstore = FAISS.from_documents(chunks, embeddings)

print("\n5. Đang lưu kho dữ liệu vào thư mục nội bộ...")
vectorstore.save_local("camau_faiss_index")
print("🎉 Xuất sắc! Toàn bộ file CSV và Excel đã được nạp thành công vào não bộ AI.")