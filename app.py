import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import importlib.util
import importlib
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Import TFLite secara aman
if importlib.util.find_spec("tflite_runtime.interpreter") is not None:
    tflite = importlib.import_module("tflite_runtime.interpreter")
else:
    from tensorflow import lite as tflite

# --- KONFIGURASI & KAMUS ---
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', 
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', 
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', 
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', 
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', 
    '--..': 'Z', ' ': ' '
}

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_segment(interpreter, audio_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing Audio
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0).astype(np.float32).reshape(1, 40)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # 0 = Dot, 1 = Dash
    return "." if np.argmax(output) == 0 else "-"

# --- UI STREAMLIT ---
st.set_page_config(page_title="Morse Audio Decoder", page_icon="📟")

st.title("📟 Morse Audio to Text Decoder")
st.info("Aplikasi ini mendeteksi titik (.) dan garis (-) dari audio lalu menerjemahkannya.")

uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("🚀 Mulai Dekode"):
        with st.spinner("Menganalisis sinyal audio..."):
            temp_input = None
            try:
                # Simpan file upload sebagai file temporer unik per-request.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_input = tmp.name

                # Load model dengan path absolut relatif ke file app.py.
                model_path = os.path.join(os.path.dirname(__file__), "morse_model.tflite")
                try:
                    interp = load_model(model_path)
                except Exception as e:
                    st.error(f"Gagal memuat model: {e}. Pastikan file 'morse_model.tflite' ada di folder yang sama.")
                    st.stop()

                # 1. Segmentasi Audio (Memisahkan beep berdasarkan silence)
                audio = AudioSegment.from_wav(temp_input)
                # min_silence_len: durasi diam minimal antar ketukan (ms)
                # silence_thresh: ambang batas kebisingan (dBFS)
                chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-40)

                if not chunks:
                    st.warning("Tidak ditemukan sinyal Morse. Coba sesuaikan volume audio atau silence threshold.")
                else:
                    simbol_hasil = ""
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for i, chunk in enumerate(chunks):
                            chunk_name = os.path.join(temp_dir, f"chunk_{i}.wav")
                            chunk.export(chunk_name, format="wav")

                            # Prediksi tiap segment
                            res = predict_segment(interp, chunk_name)
                            simbol_hasil += res

                    # Menampilkan hasil
                    st.subheader("Hasil Deteksi Simbol")
                    st.success(f"Sandi Morse: `{simbol_hasil}`")

                    # Terjemahan ke huruf (logika sederhana)
                    # Catatan: mengasumsikan input adalah satu karakter/kata tanpa spasi antar huruf
                    translation = MORSE_CODE_DICT.get(simbol_hasil, "Kombinasi tidak dikenal")

                    st.subheader("Terjemahan Teks")
                    st.info(f"Karakter: **{translation}**")
            finally:
                if temp_input and os.path.exists(temp_input):
                    os.remove(temp_input)