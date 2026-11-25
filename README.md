#  VOSK TTS НА GPU (GT 1030, FEDORA) 

1. Системные пакеты (драйвер CUDA для GT 1030, Python 3.12 и ffmpeg):
```
sudo dnf install -y \
  https://rpmfusion.org(rpm -E %fedora).noarch.rpm \
  https://rpmfusion.org(rpm -E %fedora).noarch.rpm

sudo dnf install -y xorg-x11-drv-nvidia-580xx-cuda python3.12 ffmpeg
```
### Проверка видеокарты:
```nvidia-smi```

2. Создание изолированного окружения Python 3.12:
```python3.12 -m venv ~/vosk-env```

3. Установка пакетов в окружение (строго эти версии для Pascal / GT 1030):
### Устанавливаем vosk-tts, num2words (для озвучки чисел) и удаляем CPU-версию onnxruntime
```
proxychains4 ~/vosk-env/bin/pip install --retries 10 --timeout 60 vosk-tts num2words
~/vosk-env/bin/pip uninstall -y onnxruntime
```

### Устанавливаем GPU-рантайм (cuDNN 8), NumPy 1.x и библиотеки CUDA 11.8
```
proxychains4 ~/vosk-env/bin/pip install --retries 10 --timeout 60 \
  "onnxruntime-gpu==1.18.0" \
  "numpy<2" \
  nvidia-cuda-runtime-cu11 \
  "nvidia-cudnn-cu11<9" \
  nvidia-cublas-cu11 \
  nvidia-cufft-cu11 \
  nvidia-curand-cu11
```

4. Прописать пути CUDA в activate (для ручного запуска команд в терминале):
```
echo 'export LD_LIBRARY_PATH=$(python -c '\''import nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand; print(":".join([list(m.__path__)[0] + "/lib" for m in (nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand)]))'\''):$LD_LIBRARY_PATH' >> ~/vosk-env/bin/activate
```

5. Выдача прав и первый запуск (с прокси для скачивания модели в ~/.cache/vosk):
```
chmod +x "/home/username/vosk_tts_server.py"
chmod +x "/home/username/book_to_audio.py"
```
6. Повседневное использование (уже БЕЗ прокси и БЕЗ source):
```
/home/username/vosk_tts_server.py
```

### Запуск HTTP-сервера для Speech Dispatcher (порт 5555, меню выбора 4 голосов):
```/home/username/vosk_tts_server.py```

### Пакетная озвучка книги (FB2, EPUB, TXT) в MP3 по главам:
```"/home/username/book_to_audio.py" /путь/к/книге.fb2```

### Озвучка книги с выбором диктора (-s 0..3) и ускорением (-r 0.85):
```"/home/username/book_to_audio.py" /путь/к/книге.fb2 -s 2 -r 0.85``` подробней в ```book_to_audio.py --help```

### Примечание
Исползование утилиты proxychains4 обусловлено только Чебурнетом и заблокированными репозиториями Федоры.

username должно сооствествовать актуальному имени пользователя
