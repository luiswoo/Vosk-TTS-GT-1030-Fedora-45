#!/home/username/vosk-env/bin/python ## username должно быть актуальным пользователем
import os
import sys

# Автоматически прописываем пути к библиотекам CUDA 11 и cuDNN 8 до загрузки onnxruntime
if "VOSK_CUDA_READY" not in os.environ:
    import nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand
    cuda_paths = [list(m.__path__)[0] + "/lib" for m in (nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["VOSK_CUDA_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import io
import re
import wave
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from vosk_tts import Model, Synth

# Импортируем нормализатор
from ru_normalizr import Normalizer, NormalizeOptions

# ===== НАСТРОЙКИ СЕРВЕРА =====
HOST = "127.0.0.1"
PORT = 5555
MODEL_NAME = "vosk-model-tts-ru-0.7-multi"

# ===== НАСТРОЙКИ РЕЧИ =====
LENGTH_SCALE = 1.3        # Базовая скорость: < 1.0 быстрее, > 1.0 медленнее
NOISE_SCALE = 0.667       # "Волнистость" голоса (noise_level)
NOISE_W_SCALE = 0.8       # Вариативность длительности фонем (duration_noise_level)
VOLUME = 1.0              # Громкость (множитель амплитуды)
SENTENCE_SILENCE = 0.3   # Пауза между предложениями, секунды

# Список дикторов модели vosk-model-tts-ru-0.7-multi
SPEAKERS = {
    1: (0, "Женский голос 1 — Ирина (speaker_id=0)"),
    2: (1, "Женский голос 2 — Наташа (speaker_id=1)"),
    3: (2, "Женский голос 3 (speaker_id=2)"),
    4: (3, "Мужской голос 1 (speaker_id=3)"),
    5: (4, "Мужской голос 2 (speaker_id=4)"),
}

# Глобальное состояние и блокировка потоков
current_model = None
current_synth = None
current_sample_rate = 22050
current_speaker_id = 0
current_voice_name = SPEAKERS[1][1]
voice_lock = threading.Lock()

# ВАЖНО: Инициализируем нормализатор ОДИН РАЗ при старте скрипта!
# Это уберет задержки на загрузку словарей при каждом запросе.
print("Инициализация ru-normalizr (загрузка словарей в ОЗУ)...")
normalizer = Normalizer(NormalizeOptions.tts())

# ==========================================
# ФУНКЦИИ ОБРАБОТКИ ТЕКСТА
# ==========================================

def clean_text(text: str) -> str:
    """Нормализует текст для TTS с помощью ru-normalizr."""
    # normalizer сам переведет числа, даты, латиницу и сокращения
    text = normalizer.normalize(text)
    
    # Финальная чистка от специфичных символов, если модель Vosk их не любит
    # (ru-normalizr уже убирает большинство мусора, но оставляем только кириллицу и знаки препинания)
    text = re.sub(r'[^а-яА-ЯёЁ\s.,!?;:\-—()]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def split_sentences(text: str):
    """Разбивает текст на предложения для вставки пауз SENTENCE_SILENCE."""
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def calc_speech_rate(sd_rate: float) -> float:
    """Переводит sd_rate (-100..+100 из Speech Dispatcher) + LENGTH_SCALE в speech_rate для Vosk."""
    base_rate = 1.0 / LENGTH_SCALE if LENGTH_SCALE > 0 else 1.0
    sd_rate = max(-90.0, min(200.0, sd_rate))
    if sd_rate >= 0:
        factor = 1.0 + (sd_rate / 100.0)
    else:
        factor = 1.0 / (1.0 + abs(sd_rate) / 100.0)
    return base_rate * factor


def synthesize_wav(text: str, sd_rate: float = 0.0) -> bytes:
    """Генерирует WAV-байты из текста на GPU с учетом настроек речи."""
    # Нормализуем весь текст сразу ДО разбивки на предложения
    text = clean_text(text)
    if not text:
        text = " "

    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]

    speech_rate = calc_speech_rate(sd_rate)
    silence_samples = int(current_sample_rate * SENTENCE_SILENCE)
    silence_chunk = np.zeros(silence_samples, dtype=np.float32)

    audio_chunks = []

    with voice_lock:
        spk_id = current_speaker_id
        for idx, sentence in enumerate(sentences):
            # Повторно вызывать clean_text внутри цикла уже не нужно!
            if not sentence.strip():
                continue
            raw = current_synth.synth_audio(
                sentence,
                speaker_id=spk_id,
                noise_level=NOISE_SCALE,
                speech_rate=speech_rate,
                duration_noise_level=NOISE_W_SCALE,
            )
            audio_chunks.append(raw.astype(np.float32))
            if idx < len(sentences) - 1 and silence_samples > 0:
                audio_chunks.append(silence_chunk)

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(100, dtype=np.float32)

    if VOLUME != 1.0:
        full_audio = np.clip(full_audio * VOLUME, -32768.0, 32767.0)

    pcm_data = full_audio.astype(np.int16).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(current_sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class SilentThreadingHTTPServer(ThreadingHTTPServer):
    """Сервер, который тихо игнорирует разрывы соединения со стороны клиента."""
    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


class TTSRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        text = params.get("text", [""])[0]
        output_path = params.get("output", [""])[0]
        try:
            sd_rate = float(params.get("sd_rate", ["0"])[0])
        except ValueError:
            sd_rate = 0.0

        if not text or not text.strip():
            try:
                self.send_response(400)
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        try:
            wav_bytes = synthesize_wav(text, sd_rate=sd_rate)

            if output_path:
                tmp_path = output_path + ".part"
                with open(tmp_path, "wb") as f:
                    f.write(wav_bytes)
                os.replace(tmp_path, output_path)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"\n[Ошибка синтеза]: {e}")
            try:
                self.send_response(500)
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass


def print_menu():
    print("\n" + "=" * 45)
    print(f" Сервер Vosk TTS запущен на http://{HOST}:{PORT}")
    print(f" Активный голос: {current_voice_name}")
    print("=" * 45)
    for num, (_, desc) in SPEAKERS.items():
        print(f" [{num}] {desc}")
    print(" [0] Выход")
    print("=" * 45)


def console_loop():
    global current_speaker_id, current_voice_name
    while True:
        print_menu()
        try:
            choice = int(input("Выберите голос: ").strip())
        except ValueError:
            print("Введите число.")
            continue
        except (EOFError, KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            break

        if choice == 0:
            os.kill(os.getpid(), signal.SIGINT)
            break
        elif choice in SPEAKERS:
            with voice_lock:
                current_speaker_id, current_voice_name = SPEAKERS[choice]
            print(f"\n>>> Переключено на: {current_voice_name}")
        else:
            print(f"Неверный выбор. Доступны пункты: 0–{max(SPEAKERS.keys())}")

def main():
    global current_model, current_synth, current_speaker_id, current_voice_name

    print(f"Загрузка модели {MODEL_NAME} на GPU (GT 1030)...")
    current_model = Model(model_name=MODEL_NAME)
    current_synth = Synth(current_model)

    print("Прогрев CUDA/cuDNN...")
    _ = current_synth.synth_audio("Тест", speaker_id=current_speaker_id)

    server = SilentThreadingHTTPServer((HOST, PORT), TTSRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    def shutdown(sig, frame):
        print("\nОстановка сервера и освобождение памяти GPU...")
        server.shutdown()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        print_menu()
        try:
            choice = input("> ").strip()
            if choice == "0":
                shutdown(None, None)
            elif choice.isdigit() and int(choice) in SPEAKERS:
                num = int(choice)
                with voice_lock:
                    current_speaker_id, current_voice_name = SPEAKERS[num]
                print(f"--> Голос переключен на: {current_voice_name}")
            else:
                print(f"--> Неверный ввод. Введите цифру от 1 до {len(SPEAKERS)} или 0 для выхода.")
        except EOFError:
            shutdown(None, None)

if __name__ == "__main__":
    main()
