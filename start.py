# start.py
# pip install waitress==3.0.0
# pip install transformers==4.39.3
# pip install soundfile

import os
import torch
from flask import Flask, request, send_file, jsonify
from waitress import serve
import io
import soundfile as sf
import time # Zamanlama için eklendi
from TTS.api import TTS

# --- Sunucu ve Model Yapılandırması ---
HOST = "0.0.0.0"
PORT = 5002
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# --- ⭐️ KRİTİK DEĞİŞİKLİK: SABİT REFERANS SES DOSYASI ⭐️ ---
# Bu dosya, tüm konuşmaların ana sesi olacak. 
# Yüksek kaliteli uzun (2-5 dk) ve gürültüsüz bir WAV dosyası

REFERENCE_SPEAKER_WAV_PATH = "audio/reference_002.wav" # # 2 dakika uzunluğunda.

# --- Model Yükleme ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Cihaz: {device.upper()} kullanılacak.")

# --- Referans Sesi Kontrol Etme ---
if not os.path.exists(REFERENCE_SPEAKER_WAV_PATH):
    print(f"❌ KRİTİK HATA: Ana referans ses dosyası bulunamadı: '{REFERENCE_SPEAKER_WAV_PATH}'")
    print("Lütfen bu yola yüksek kaliteli ana ses dosyanızı yerleştirin ve sunucuyu yeniden başlatın.")
    exit()
else:
    print(f"✅ Ana referans ses dosyası başarıyla bulundu: {REFERENCE_SPEAKER_WAV_PATH}")

print(f"'{MODEL_NAME}' modeli yükleniyor... (Bu işlem biraz sürebilir)")
try:
    tts = TTS(MODEL_NAME).to(device)
    print("✅✅✅ TTS modeli başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenirken kritik hata: {e}")
    exit()

# --- Web Sunucusu ---
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    # Modelin yüklenip yüklenmediğini ve referans sesin varlığını kontrol et
    model_ok = 'tts' in globals() and tts is not None
    ref_ok = os.path.exists(REFERENCE_SPEAKER_WAV_PATH)
    if model_ok and ref_ok:
        return jsonify({"status": "healthy", "model_loaded": True, "reference_audio_found": True, "device": device}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": model_ok, "reference_audio_found": ref_ok}), 503

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    start_time = time.time()
    try:
        text = request.form.get('text', '')
        language = request.form.get('language', 'tr')
        speed = float(request.form.get('speed', 1.0))

        if not text:
            return jsonify({"error": "'text' parametresi gerekli."}), 400

        print(f"İstek alındı: Hız={speed}, Dil={language}, Metin='{text[:60]}...'")
        
        # ⭐️ KRİTİK DEĞİŞİKLİK: SABİT REFERANS SESİ KULLANILIYOR ⭐️
        # Artık her istekte dosya yüklemesi yapılmıyor.
        wav_chunks = tts.tts(
            text=text,
            speaker_wav=REFERENCE_SPEAKER_WAV_PATH, # Sunucu başlarken yüklenen sabit dosya kullanılıyor
            language=language,
            speed=speed
        )
        
        # Üretilen sesi hafızada bir tampona yaz
        buffer = io.BytesIO()
        output_sample_rate = tts.synthesizer.output_sample_rate
        sf.write(buffer, wav_chunks, output_sample_rate, format='WAV')
        buffer.seek(0)

        processing_time = time.time() - start_time
        audio_duration = len(wav_chunks) / output_sample_rate
        real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"✅ Ses başarıyla üretildi. İşlem Süresi: {processing_time:.2f}s, Ses Süresi: {audio_duration:.2f}s, RTF: {real_time_factor:.2f}")
        
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

if __name__ == '__main__':
    print(f"🚀 Sentiric Voice Engine (XTTS) http://{HOST}:{PORT} adresinde çalışmaya hazır.")
    print(f"⚠️ Not: İlk istek, model önbelleğe alındığı için biraz daha yavaş olabilir.")
    serve(app, host=HOST, port=PORT)