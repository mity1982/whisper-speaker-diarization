import os
import sys
import argparse
import whisper
import math
import ssl
import urllib.request
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_audio_features(audio_segment, sample_rate=16000):
    """Извлекает признаки аудио для определения спикера"""
    # Основные признаки: средняя амплитуда, основная частота, спектральные характеристики
    
    # 1. Энергия сигнала
    energy = np.mean(audio_segment ** 2)
    
    # 2. Zero crossing rate (скорость пересечения нуля)
    zero_crossings = np.sum(np.diff(np.sign(audio_segment)) != 0)
    zcr = zero_crossings / len(audio_segment)
    
    # 3. Спектральные признаки (простые)
    # Применяем FFT для получения частотного спектра
    fft = np.fft.fft(audio_segment)
    magnitude = np.abs(fft)[:len(fft)//2]
    
    # Спектральный центроид (центр тяжести спектра)
    freqs = np.fft.fftfreq(len(audio_segment), 1/sample_rate)[:len(fft)//2]
    if np.sum(magnitude) > 0:
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    else:
        spectral_centroid = 0
    
    # 4. Спектральная полоса пропускания
    if np.sum(magnitude) > 0:
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
    else:
        spectral_bandwidth = 0
    
    # 5. Мел-частотные коэффициенты (упрощенная версия)
    mel_bands = 13
    mel_filters = np.linspace(0, sample_rate//2, mel_bands)
    mel_energies = []
    for i in range(len(mel_filters)-1):
        start_idx = int(mel_filters[i] * len(magnitude) / (sample_rate//2))
        end_idx = int(mel_filters[i+1] * len(magnitude) / (sample_rate//2))
        if end_idx > start_idx:
            mel_energy = np.mean(magnitude[start_idx:end_idx])
        else:
            mel_energy = 0
        mel_energies.append(mel_energy)
    
    # Возвращаем вектор признаков
    features = [energy, zcr, spectral_centroid, spectral_bandwidth] + mel_energies
    return np.array(features)

def cluster_speakers(segments_features, n_speakers=2):
    """Кластеризует сегменты по спикерам"""
    if len(segments_features) < n_speakers:
        # Если сегментов меньше чем спикеров, назначаем каждому свой кластер
        return list(range(len(segments_features)))
    
    # Нормализуем признаки
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(segments_features)
    
    # Применяем K-means кластеризацию
    kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    speaker_labels = kmeans.fit_predict(features_normalized)
    
    return speaker_labels

def get_audio_path():
    """Получает путь к аудиофайлу из аргументов командной строки или запрашивает у пользователя"""
    parser = argparse.ArgumentParser(description='Транскрипция аудио с помощью Whisper')
    parser.add_argument('audio_file', nargs='?', help='Путь к аудиофайлу')
    args = parser.parse_args()
    
    if args.audio_file:
        return args.audio_file
    else:
        # Если аргумент не передан, запрашиваем у пользователя
        audio_path = input("Введите путь к аудиофайлу: ").strip()
        if not audio_path:
            print("Ошибка: путь к файлу не указан")
            sys.exit(1)
        return audio_path

# === Настройки ===
AUDIO_PATH = get_audio_path()

# Проверяем, что файл существует
if not os.path.exists(AUDIO_PATH):
    print(f"Ошибка: файл '{AUDIO_PATH}' не найден")
    sys.exit(1)

CHUNK_LENGTH_MIN = 10  # Длина кусков в минутах
MODEL_NAME = "base"  # Можно поменять на 'small', 'medium', 'large'
OUTPUT_FILE = "transcription.txt"
NUM_SPEAKERS = 2  # Количество ожидаемых спикеров (можно изменить)

# === Инициализация ===
print(f"[INFO] Загружаю модель Whisper: {MODEL_NAME}...")

# Временно отключаем проверку SSL сертификатов для загрузки модели
try:
    model = whisper.load_model(MODEL_NAME)
except ssl.SSLError:
    print("[INFO] Проблема с SSL сертификатами, пробую альтернативный способ...")
    # Создаем неверифицированный SSL контекст
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Временно заменяем SSL контекст
    original_context = ssl._create_default_https_context
    ssl._create_default_https_context = lambda: ssl_context
    
    try:
        model = whisper.load_model(MODEL_NAME)
    finally:
        # Восстанавливаем оригинальный SSL контекст
        ssl._create_default_https_context = original_context

print(f"[INFO] Загружаю аудиофайл: {AUDIO_PATH}...")
# Whisper может работать напрямую с аудиофайлами без pydub
try:
    audio = whisper.load_audio(AUDIO_PATH)
except FileNotFoundError as e:
    if 'ffmpeg' in str(e):
        print("[ERROR] FFmpeg не найден!")
        print("Для работы с аудиофайлами Whisper требует FFmpeg.")
        print("Установите FFmpeg одним из способов:")
        print("1. Через Homebrew: brew install ffmpeg")
        print("2. Или скачайте с https://ffmpeg.org/download.html")
        print("\nПосле установки FFmpeg перезапустите скрипт.")
        sys.exit(1)
    else:
        print(f"[ERROR] Ошибка загрузки аудиофайла: {e}")
        sys.exit(1)

# Получаем длительность аудио в секундах
audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
chunk_length_sec = CHUNK_LENGTH_MIN * 60

# Разбиваем аудио на куски
chunks = []
for start_time in range(0, int(audio_duration), chunk_length_sec):
    end_time = min(start_time + chunk_length_sec, audio_duration)
    start_sample = int(start_time * whisper.audio.SAMPLE_RATE)
    end_sample = int(end_time * whisper.audio.SAMPLE_RATE)
    chunk_audio = audio[start_sample:end_sample]
    chunks.append((chunk_audio, start_time))

os.makedirs("chunks", exist_ok=True)
full_transcript = ""

print(f"[INFO] Обработка {len(chunks)} кусков...")

# Собираем все сегменты и их признаки для дальнейшей кластеризации
all_segments = []
all_features = []

# Сначала обрабатываем все куски и извлекаем сегменты
for idx, (chunk_audio, chunk_start_time) in enumerate(chunks):
    print(f"[INFO] Распознаю часть {idx+1}/{len(chunks)}...")
    
    # Whisper может работать напрямую с массивом аудио
    result = model.transcribe(chunk_audio, language="ru", verbose=False)
    
    for segment in result['segments']:
        # Корректируем время с учетом смещения куска
        start = segment['start'] + chunk_start_time
        end = segment['end'] + chunk_start_time
        text = segment['text'].strip()
        
        # Извлекаем аудио сегмент для анализа спикера
        start_sample = int(segment['start'] * whisper.audio.SAMPLE_RATE)
        end_sample = int(segment['end'] * whisper.audio.SAMPLE_RATE)
        
        # Убеждаемся, что индексы в пределах chunk_audio
        start_sample = max(0, min(start_sample, len(chunk_audio)-1))
        end_sample = max(start_sample+1, min(end_sample, len(chunk_audio)))
        
        segment_audio = chunk_audio[start_sample:end_sample]
        
        # Извлекаем признаки для определения спикера
        if len(segment_audio) > 0:
            features = extract_audio_features(segment_audio)
            all_features.append(features)
        else:
            # Если сегмент пустой, используем нулевые признаки
            all_features.append(np.zeros(16))  # 16 - размер вектора признаков
        
        # Сохраняем информацию о сегменте
        segment_info = {
            'start': start,
            'end': end,
            'text': text,
            'chunk_idx': idx
        }
        all_segments.append(segment_info)

print(f"[INFO] Определяю спикеров для {len(all_segments)} сегментов...")

# Кластеризуем сегменты по спикерам
if len(all_features) > 0:
    speaker_labels = cluster_speakers(all_features, NUM_SPEAKERS)
else:
    speaker_labels = []

# Формируем финальную транскрипцию с определенными спикерами
for i, (segment_info, speaker_id) in enumerate(zip(all_segments, speaker_labels)):
    start_time = f"{int(segment_info['start']//60):02d}:{int(segment_info['start']%60):02d}"
    end_time = f"{int(segment_info['end']//60):02d}:{int(segment_info['end']%60):02d}"
    
    # Присваиваем спикера (добавляем 1 чтобы начинать с "Спикер 1")
    speaker = f"Спикер {speaker_id + 1}"
    
    full_transcript += f"[{start_time} - {end_time}] {speaker}: {segment_info['text']}\n"

# Сохраняем в файл
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_transcript)

print(f"[DONE] Расшифровка сохранена в: {OUTPUT_FILE}")
