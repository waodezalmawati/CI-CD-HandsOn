# Menggunakan image dasar dari Python
FROM python:3.9-slim

# Menyiapkan direktori kerja di dalam container
WORKDIR /app

# Menyalin requirements dan menginstal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin file script
COPY training_model.py .

# Menjalankan service pada port 5001
EXPOSE 5001

# Menjalankan aplikasi
CMD ["python", "training_model.py"]
