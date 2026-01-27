# Setup Paravibe AI

## Các bước cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/LuanDHV/paravibe-ai.git
cd paravibe-ai
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

```bash
cp .env.example .env.local
```

Chỉnh sửa `.env.local` với thông tin database của bạn

### 5. Khởi chạy ứng dụng

```bash
uvicorn app.main:app --reload --port 8000
```

Ứng dụng sẽ chạy tại `http://localhost:8000`
