# start.py
# pip install waitress==3.0.0
# pip install transformers==4.39.3

# 1. Referans ses dosyasını hazırlayın
# Bu klasörün içine (coqui-TTS/), 'speaker_ref.wav' adında bir ses dosyası koyun.
# Bu dosya, ses klonlama için kullanılacaktır.

# 2. Sunucuyu başlatın
# python start.py

import os
import torch
from flask import Flask, request, send_file, jsonify
from waitress import serve
import io
import soundfile as sf
import tempfile
from TTS.api import TTS

# PyTorch güvenlik ayarları
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
except ImportError:
    print("Uyarı: TTS kütüphanesinin eski bir sürümü kullanılıyor olabilir, güvenlik ayarları atlanıyor.")


# --- Sunucu ve Model Yapılandırması ---
HOST = "0.0.0.0"
PORT = 5002
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# --- Model Yükleme ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Cihaz: {device.upper()} kullanılacak.")
print(f"'{MODEL_NAME}' modeli yükleniyor...")
try:
    tts = TTS(MODEL_NAME).to(device)
    print("✅✅✅ TTS modeli başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenirken kritik hata: {e}")
    exit()

# --- Web Sunucusu ---
app = Flask(__name__)

# YENİ: Healthcheck endpoint'i
@app.route('/health', methods=['GET'])
def health_check():
    # Modelin yüklü ve kullanılabilir olduğunu kontrol et
    if tts:
        return jsonify({"status": "healthy", "model_loaded": True, "device": device}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False, "reason": "TTS model not loaded"}), 503

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    try:
        # POST isteğinden verileri al
        text = request.form.get('text', '')
        language = request.form.get('language', 'tr')
        speed = float(request.form.get('speed', 1.50))
        speaker_wav_file = request.files.get('speaker_ref_wav')

        if not text:
            return jsonify({"error": "'text' parametresi gerekli."}), 400
        if not speaker_wav_file:
            return jsonify({"error": "POST isteği için 'speaker_ref_wav' dosyası gerekli."}), 400

        # Gelen referans ses dosyasını geçici bir dosyaya kaydet
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as speaker_fp:
            speaker_wav_path = speaker_fp.name
            speaker_wav_file.save(speaker_wav_path)

        print(f"İstek alındı: Hız={speed}, Metin='{text[:30]}...'")
        
        # Sesi üret
        wav_chunks = tts.tts(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            speed=speed
        )
        
        # Üretilen sesi hafızada bir tampona yaz
        buffer = io.BytesIO()
        sf.write(buffer, wav_chunks, tts.synthesizer.output_sample_rate, format='WAV')
        buffer.seek(0)

        print(f"✅ Ses başarıyla hafızada üretildi.")
        
        # Hafızadaki ses verisini direkt olarak gönder
        return send_file(
            buffer, 
            mimetype='audio/wav',
            as_attachment=False,
            download_name='output.wav'
        )

    except Exception as e:
        print(f"❌ Ses üretimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Ses üretimi sırasında bir sunucu hatası oluştu."}), 500
    
    finally:
        # Geçici referans ses dosyasını her durumda sil
        if 'speaker_wav_path' in locals() and os.path.exists(speaker_wav_path):
            os.remove(speaker_wav_path)

if __name__ == '__main__':
    print(f"🚀 Dağıtık Mimarili XTTS Sunucusu http://{HOST}:{PORT} adresinde çalışmaya hazır.")
    serve(app, host=HOST, port=PORT)