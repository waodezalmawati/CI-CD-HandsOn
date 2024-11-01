# CI-CD-HandsOn

#### 1. Buat `requirements.txt`
File `requirements.txt` ini akan memastikan semua library yang dibutuhkan terinstal otomatis di lingkungan Docker. Berikut adalah contoh isinya:

```plaintext
tensorflow
pandas
transformers
torch
datasets
```

Simpan file ini dengan nama `requirements.txt` di direktori proyek Anda.

#### 2. Buat Dockerfile
Dockerfile ini akan mengatur lingkungan untuk menjalankan model dan proses training.

```dockerfile
# Gunakan image dasar dengan Python dan dependencies dasar
FROM python:3.9-slim

# Set direktori kerja
WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam image
COPY . .

# Menjalankan training script
CMD ["python", "training_script.py"]
```

**Catatan**: Ubah `training_script.py` menjadi nama file script Python yang Anda gunakan untuk training.

#### 3. Buat File `training_script.py`
```python
import tensorflow as tf
import pandas as pd
from datasets import Dataset
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Inisialisasi MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Tentukan batch size
BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

# Load tokenizer dan tambahkan token padding
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Baca data dari CSV
df = pd.read_csv("/app/sample_data/Week3NikeProductDescriptionsGenerator.csv")  # Pastikan path file sudah benar
descriptions = df['Product Description'].tolist()

# Tokenisasi data
def preprocess(desc):
    encodings = tokenizer(desc, truncation=True, padding=True)
    return Dataset.from_dict(encodings)

train_dataset = preprocess(descriptions)

# Inisialisasi data collator untuk Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training loop dengan strategy.scope()
with strategy.scope():
    # Load model GPT2
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=model.compute_loss)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=GLOBAL_BATCH_SIZE,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )

    # Inisialisasi Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # Jalankan fine-tuning model
    trainer.train()
```

#### 4. Build dan Push Docker Image ke Docker Hub
Pastikan Anda sudah login ke Docker Hub terlebih dahulu.

```bash
docker build -t <dockerhub_username>/model-training:latest .
docker push <dockerhub_username>/model-training:latest
```

Gantilah `<dockerhub_username>` dengan username Docker Hub Anda.

#### 5. Setup CI/CD dengan GitHub Actions

1. **Buat Folder Workflow**: Di dalam repository GitHub Anda, buat folder `.github/workflows`.
2. **Buat File Workflow CI/CD**: Buat file bernama `ci-cd.yml` di dalam folder `workflows` tersebut.

Isi file `ci-cd.yml` dengan konfigurasi berikut:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Build Docker image
      run: docker build -t <dockerhub_username>/model-training:latest .

    - name: Log in to Docker Hub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Push Docker image to Docker Hub
      run: docker push <dockerhub_username>/model-training:latest
```

**Catatan**:
- Gantilah `<dockerhub_username>` dengan username Docker Hub Anda.
- Tambahkan `DOCKER_USERNAME` dan `DOCKER_PASSWORD` di GitHub Secrets agar pipeline dapat login ke Docker Hub.

#### 6. Buat Deployment YAML untuk Kubernetes
Jika Anda ingin *deploy* model ke Kubernetes, buat file YAML untuk Kubernetes, misalnya `model-deployment.yaml`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-app
  template:
    metadata:
      labels:
        app: model-app
    spec:
      containers:
      - name: model-container
        image: <dockerhub_username>/model-training:latest
        ports:
        - containerPort: 8080
```

Apply file ini ke Kubernetes:

```bash
kubectl apply -f model-deployment.yaml
```

### Cara Kerja CI/CD Pipeline

1. **Commit dan Push Perubahan**: Setiap kali ada perubahan di repository GitHub, GitHub Actions akan menjalankan *workflow* CI yang secara otomatis membangun dan mendorong image ke Docker Hub.
2. **Deployment ke Kubernetes**: Setelah CI berjalan sukses, Anda bisa melanjutkan ke tahap CD dengan men-*deploy* image terbaru ke Kubernetes.
