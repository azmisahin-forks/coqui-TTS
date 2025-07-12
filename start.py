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
# tempfile'a artık ihtiyacımız yok
from TTS.api import TTS

# --- Sunucu ve Model Yapılandırması ---
HOST = "0.0.0.0"
PORT = 5002
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# --- Referans Ses Dosyası Yolu ---
# Buraya, kaliteli ve uzun referans ses dosyanızın yolunu belirtin.
# Bu dosya, sunucunun çalıştırıldığı dizine göre göreceli veya mutlak bir yol olabilir.
# ÖNEMLİ: Bu dosya sunucuyu çalıştırdığınız makinede/konteynerde olmalı.
REFERENCE_SPEAKER_WAV_PATH = "audio/reference_001.wav" # Örneğin, `start.py`'nin yanındaki `audio` klasöründe.

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

# --- Referans Sesi Kontrol Etme ve Yükleme ---
# Sunucu başladığında referans ses dosyasının varlığını kontrol et
if not os.path.exists(REFERENCE_SPEAKER_WAV_PATH):
    print(f"❌ Hata: Referans ses dosyası bulunamadı: {REFERENCE_SPEAKER_WAV_PATH}")
    print("Lütfen bu yola yüksek kaliteli ve uzun bir WAV dosyası yerleştirin (örn: 'audio/ana_ses.wav').")
    exit()
else:
    print(f"✅ Referans ses dosyası yüklendi: {REFERENCE_SPEAKER_WAV_PATH}")


# --- Web Sunucusu ---
app = Flask(__name__)

# YENİ: Healthcheck endpoint'i (mevcut)
@app.route('/health', methods=['GET'])
def health_check():
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
        # Hız parametresini deneyerek en doğal olanı bulun. 1.0 başlangıç için iyidir.
        # Bu değer, sizin klonladığınız sesin doğal konuşma hızına göre ayarlanmalı.
        speed = float(request.form.get('speed', 1.0)) # Varsayılan 1.0'a çekildi.

        if not text:
            return jsonify({"error": "'text' parametresi gerekli."}), 400
        
        # Kaldırılan kısım: speaker_ref_wav_file yükleme
        # Artık her istekte referans ses dosyası yüklemeyeceğiz.
        # Bunun yerine, yukarıda tanımladığımız sabit REFERENCE_SPEAKER_WAV_PATH'i kullanacağız.

        print(f"İstek alındı: Hız={speed}, Metin='{text[:50]}...'")
        
        # Sesi üretirken sabit referans sesi kullan
        wav_chunks = tts.tts(
            text=text,
            speaker_wav=REFERENCE_SPEAKER_WAV_PATH, # BURAYI DEĞİŞTİRDİK!
            language=language,
            speed=speed
        )
        
        # Üretilen sesi hafızada bir tampona yaz
        buffer = io.BytesIO()
        sf.write(buffer, wav_chunks, tts.synthesizer.output_sample_rate, format='WAV')
        buffer.seek(0)

        print(f"✅ Ses başarıyla hafızada üretildi.")
        
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
        # Geçici referans ses dosyasını silmeye artık gerek yok.
        pass # `if 'speaker_wav_path'` bloğu kaldırıldı.

if __name__ == '__main__':
    print(f"🚀 Dağıtık Mimarili XTTS Sunucusu http://{HOST}:{PORT} adresinde çalışmaya hazır.")
    serve(app, host=HOST, port=PORT)