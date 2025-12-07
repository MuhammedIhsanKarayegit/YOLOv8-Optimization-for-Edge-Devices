# Gerekli kütüphaneleri yüklüyoruz
#!pip install ultralytics
import os
from ultralytics import YOLO
import time
import numpy as np

# YOLOv8 Nano modelini (en hafif versiyon) indir
model = YOLO('yolov8n.pt')

# Modelin diskteki boyutunu ölçelim (Megabyte cinsinden)
orijinal_boyut = os.path.getsize('yolov8n.pt') / (1024 * 1024)
print(f"Orijinal Model Boyutu (PyTorch): {orijinal_boyut:.2f} MB")

print("Dönüşüm başlıyor... Bu işlem biraz sürebilir.")

# 1. TFLite Formatına Dönüştür (Float32 - Standart Dönüşüm)
# Bu format Android/Raspberry Pi için uygundur ama tam sıkıştırılmamıştır.
model.export(format='tflite')

# 2. INT8 Quantization ile Dönüştür (Mühendislik dokunuşu)
# int8=True parametresi ağırlıkları 4 kat küçültür.
model.export(format='tflite', int8=True, data='coco128.yaml') # data parametresi kalibrasyon için gereklidir

# Boyutları Kontrol Et
f32_boyut = os.path.getsize('yolov8n_saved_model/yolov8n_float32.tflite') / (1024 * 1024)
int8_boyut = os.path.getsize('yolov8n_saved_model/yolov8n_integer_quant.tflite') / (1024 * 1024) # Dosya adı sürüme göre değişebilir, çıktıdan kontrol et

print(f"\n--- SONUÇLAR ---")
print(f"Orijinal (PyTorch): {orijinal_boyut:.2f} MB")
print(f"TFLite (Float32):   {f32_boyut:.2f} MB")
print(f"TFLite (INT8):      {int8_boyut:.2f} MB")

reduction = (1 - (int8_boyut / orijinal_boyut)) * 100
print(f"📉 Boyut Kazancı: %{reduction:.1f} daha küçük!")

import tensorflow as tf

def run_tflite_inference(model_path, image_path='bus.jpg'):
    # TFLite yorumlayıcısını yükle
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Rastgele bir veri ile test et (Sadece hız ölçüyoruz)
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # Eğer model INT8 ise input'u da dönüştürmek gerekebilir (basitleştirilmiş test için float bırakıyoruz)
    # Gerçek dünya senaryosunda preprocess gerekir.

    # Isınma turu (ilk işlem her zaman yavaştır)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Hız Testi (100 döngü)
    start_time = time.time()
    for _ in range(100):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    end_time = time.time()

    avg_ms = ((end_time - start_time) / 100) * 1000
    fps = 1000 / avg_ms
    return fps, avg_ms

# Örnek bir resim indirelim (Ultralytics içinde gelir ama garanti olsun)
#!yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg' save=False

# Testi Çalıştır (Dosya yollarını export çıktısına göre güncellemelisin)
# Genellikle yolov8n_saved_model klasörü içinde oluşur
tflite_model_path = 'yolov8n_saved_model/yolov8n_float32.tflite' # Burayı kontrol et

try:
    fps, ms = run_tflite_inference(tflite_model_path)
    print(f"\nTFLite Model Hızı (CPU): {fps:.2f} FPS ({ms:.2f} ms)")
except Exception as e:
    print(f"Test hatası (Dosya yolunu kontrol et): {e}")