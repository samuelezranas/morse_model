# morse_model

Streamlit app untuk decode audio Morse (dot/dash) menggunakan model TFLite.

## Deploy ke Hugging Face Spaces (Docker)

Gunakan opsi berikut saat membuat Space:
- SDK: Docker
- Hardware: CPU Basic
- Visibility: Public (atau sesuai kebutuhan)

File yang dipakai untuk deploy:
- `Dockerfile`
- `requirements.txt`
- `app.py`
- `morse_model.tflite`

### Cara deploy

1. Buat Space baru di Hugging Face dengan SDK Docker.
2. Push isi repository ini ke repo Space kamu.
3. Tunggu proses build selesai (image akan install dependency Python + ffmpeg).
4. Setelah running, app akan tersedia di URL Space.

### Menjalankan lokal pakai Docker

```bash
docker build -t morse-model .
docker run -p 7860:7860 morse-model
```

Lalu buka `http://localhost:7860`.