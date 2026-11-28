#!/home/username/vosk-env/bin/python ##username должно быть актуальным пользователем
import os
import sys

# Автоматически подключаем библиотеки CUDA 11 и cuDNN 8 для GT 1030
if "VOSK_CUDA_READY" not in os.environ:
    import nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand
    cuda_paths = [list(m.__path__)[0] + "/lib" for m in (nvidia.cublas, nvidia.cudnn, nvidia.cuda_runtime, nvidia.cufft, nvidia.curand)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["VOSK_CUDA_READY"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import re
import time
import wave
import zipfile
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from vosk_tts import Model, Synth
from ru_normalizr import Normalizer, NormalizeOptions

# ===== НАСТРОЙКИ РЕЧИ ПО УМОЛЧАНИЮ =====
MODEL_NAME = "vosk-model-tts-ru-0.7-multi"
SAMPLE_RATE = 22050
LENGTH_SCALE = 1.3
NOISE_SCALE = 0.667
NOISE_W_SCALE = 0.8
VOLUME = 1.0
SENTENCE_SILENCE = 0.35
PARAGRAPH_SILENCE = 0.55
MAX_CHUNK_CHARS = 240

SPEAKERS = {
    1: (0, "Женский голос 1 — Ирина (speaker_id=0)"),
    2: (1, "Женский голос 2 — Наташа (speaker_id=1)"),
    3: (2, "Женский голос 3 (speaker_id=2)"),
    4: (3, "Мужской голос 1 (speaker_id=3)"),
    5: (4, "Мужской голос 2 (speaker_id=4)"),
}

# ===== ИНИЦИАЛИЗАЦИЯ НОРМАЛИЗАТОРА =====
NORMALIZER = Normalizer(NormalizeOptions.tts())

# Регулярное выражение для санитизации (кавычки ИСКЛЮЧЕНЫ)
_TTS_ALLOWED_PATTERN = re.compile(r'[^а-яА-ЯёЁ\s.,!?;:()\-]')


def clean_text(text: str) -> str:
    """Нормализует текст через ru-normalizr и санитизирует для Vosk TTS."""
    # 1. Нормализация
    normalized = NORMALIZER.normalize(text)
    
    # 2. Санитизация
    replacements = [
        ('…', '...'),
        ('–', '-'),
        ('—', '-'),
        ('„', ''),
        ('"', ''),
        ('«', ''),
        ('»', ''),
        ('"', ''),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    
    sanitized = _TTS_ALLOWED_PATTERN.sub(' ', normalized)
    return re.sub(r'\s+', ' ', sanitized).strip()


def split_into_safe_chunks(text: str, max_len: int = MAX_CHUNK_CHARS):
    """Разбивает текст на чанки для GPU синтеза."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            chunks.append(sent)
        else:
            words = sent.split(' ')
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 <= max_len:
                    buf = f"{buf} {w}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)
    return chunks


def sanitize_filename(name: str) -> str:
    """Очищает имя файла от недопустимых символов."""
    name = re.sub(r'[\\/*?:"<>|]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    return name[:60] or "Глава"


# ---------- FB2 helpers ----------

def _fb2_ns(root):
    return {'fb': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}


def _fb2_findall(elem, path, ns):
    if not ns:
        return elem.findall(path)
    parts = []
    for p in path.split('/'):
        if p == '' or p == '.' or p == '..':
            parts.append(p)
        else:
            parts.append(f"fb:{p}")
    return elem.findall('/'.join(parts), ns)


def _fb2_text(elem) -> str:
    return " ".join("".join(elem.itertext()).split())


def _load_fb2(file_path: Path):
    """Загружает FB2 из обычного файла или из .fb2.zip архива."""
    if file_path.suffix.lower() == '.zip':
        # Это fb2.zip — нужно распаковать
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Ищем XML-файл внутри архива
            fb2_files = [f for f in zf.namelist() if f.endswith('.fb2')]
            if not fb2_files:
                raise ValueError(f"В архиве {file_path.name} не найден файл .fb2")
            # Читаем первый найденный FB2
            with zf.open(fb2_files[0]) as fb2_fh:
                return ET.parse(fb2_fh)
    else:
        # Обычный .fb2 файл
        with open(file_path, "rb") as fh:
            return ET.parse(fh)

def parse_fb2_meta(file_path: Path) -> dict:
    meta = {
        "author": "",
        "book_title": "",
        "series": "",
        "series_number": "",
        "annotation": "",
    }
    try:
        tree = _load_fb2(file_path)
        root = tree.getroot()
        ns = _fb2_ns(root)

        def findall(elem, path):
            return _fb2_findall(elem, path, ns)

        authors = findall(root, './/title-info/author') or findall(root, './/author')
        names = []
        for auth in authors:
            fn = findall(auth, 'first-name')
            mn = findall(auth, 'middle-name')
            ln = findall(auth, 'last-name')
            nick = findall(auth, 'nickname')
            parts = []
            for lst in (fn, mn, ln):
                if lst:
                    txt = _fb2_text(lst[0])
                    if txt:
                        parts.append(txt)
            if parts:
                names.append(" ".join(parts))
            elif nick:
                names.append(_fb2_text(nick[0]))
        meta["author"] = ", ".join([n for n in names if n])

        bt = findall(root, './/title-info/book-title') or findall(root, './/book-title')
        if bt:
            meta["book_title"] = _fb2_text(bt[0])

        seq = findall(root, './/title-info/sequence') or findall(root, './/sequence')
        if seq:
            meta["series"] = seq[0].attrib.get("name", "")
            meta["series_number"] = seq[0].attrib.get("number", "")

        ann = findall(root, './/title-info/annotation') or findall(root, './/annotation')
        if ann:
            meta["annotation"] = _fb2_text(ann[0])
    except Exception as e:
        print(f"Предупреждение: не удалось прочитать метаданные FB2: {e}")
    return meta


def _collect_sections(sec, findall, text_of, out):
    nested = findall(sec, 'section')
    if nested:
        for sub in nested:
            _collect_sections(sub, findall, text_of, out)
        return

    titles = findall(sec, 'title')
    if titles:
        parts = [text_of(p) for p in findall(titles[0], 'p')]
        if not parts:
            parts = [text_of(titles[0])]
        title = " ".join(" ".join(parts).split())
    else:
        title = ""
    if not title:
        title = f"Глава {len(out) + 1}"

    paras = [title + "."]

    title_texts = set()
    for te in findall(sec, 'title'):
        for p in findall(te, './/p'):
            title_texts.add(text_of(p))
        title_texts.add(text_of(te))

    for p in findall(sec, './/p'):
        txt = text_of(p)
        if txt and txt not in title_texts:
            paras.append(txt)

    if len(paras) > 1:
        out.append((title, paras))


def parse_fb2(file_path: Path):
    try:
        tree = _load_fb2(file_path)
    except Exception as e:
        print(f"Ошибка чтения FB2: {e}")
        return []
    root = tree.getroot()
    ns = _fb2_ns(root)

    def findall(elem, path):
        return _fb2_findall(elem, path, ns)

    def text_of(elem):
        return _fb2_text(elem)

    bodies = findall(root, 'body')
    if not bodies:
        txt = text_of(root)
        return [("Книга", [txt])] if txt else []

    main_body = bodies[0]
    sections = findall(main_body, 'section')
    chapters = []

    if not sections:
        paras = [text_of(p) for p in findall(main_body, './/p') if text_of(p)]
        body_title_texts = set()
        for te in findall(main_body, 'title'):
            for p in findall(te, './/p'):
                body_title_texts.add(text_of(p))
        paras = [p for p in paras if p not in body_title_texts]
        return [("Книга", paras)] if paras else []

    for sec in sections:
        _collect_sections(sec, findall, text_of, chapters)

    return chapters


# ---------- EPUB ----------

def parse_epub(file_path: Path):
    chapters = []
    with zipfile.ZipFile(file_path, 'r') as zf:
        html_files = sorted([f for f in zf.namelist() if f.endswith(('.xhtml', '.html', '.htm'))])
        for idx, fname in enumerate(html_files, 1):
            raw_html = zf.read(fname).decode('utf-8', errors='ignore')
            raw_html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw_html, flags=re.S | re.I)

            heading = ""
            m = re.search(r'<h[1-3][^>]*>(.*?)</h[1-3]>', raw_html, flags=re.S | re.I)
            if m:
                heading = " ".join(re.sub(r'<[^>]+>', ' ', m.group(1)).split())

            lines = re.split(r'</(?:p|div|h[1-6]|li|br)>', raw_html, flags=re.I)
            paras = []
            for line in lines:
                clean_line = re.sub(r'<[^>]+>', ' ', line)
                clean_line = " ".join(clean_line.split())
                if clean_line:
                    paras.append(clean_line)
            if paras:
                title = heading if heading else (paras[0][:50] if len(paras[0]) < 80 else f"Глава {idx}")
                chapters.append((title, paras))
    return chapters


def convert_wav_to_mp3(wav_path: Path, qscale: str = "4"):
    """VBR MP3 через libmp3lame."""
    mp3_path = wav_path.with_suffix(".mp3")
    res = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-q:a", qscale, str(mp3_path)],
        capture_output=True
    )
    if res.returncode == 0 and mp3_path.exists():
        wav_path.unlink()
        return mp3_path
    return wav_path


def main():
    parser = argparse.ArgumentParser(description="Пакетная озвучка книг (FB2, EPUB) на GT 1030 через Vosk TTS")
    parser.add_argument("book", help="Путь к файлу книги (.fb2, .epub)")
    parser.add_argument(
        "-s", "--speaker", type=int, default=1,
        choices=list(SPEAKERS.keys()),
        help="Номер голоса: 1,2,3 (жен), 4,5 (муж)"
    )
    parser.add_argument("-r", "--rate", type=float, default=LENGTH_SCALE, help="LENGTH_SCALE")
    parser.add_argument("-o", "--outdir", default=None, help="Папка для сохранения аудиокниги")
    parser.add_argument("--no-mp3", action="store_true", help="Не сжимать в MP3, оставить WAV")
    args = parser.parse_args()

    book_path = Path(args.book).expanduser().resolve()
    if not book_path.exists():
        print(f"Ошибка: файл не найден: {book_path}")
        sys.exit(1)

    ext = book_path.suffix.lower()
    if ext == '.zip':
        # Проверяем, это fb2.zip или что-то другое
        with zipfile.ZipFile(book_path, 'r') as zf:
            if any(f.endswith('.fb2') for f in zf.namelist()):
                ext = '.fb2.zip'
            else:
                print(f"Ошибка: архив {book_path.name} не содержит .fb2 файлов")
                sys.exit(1)

    print(f"Чтение книги: {book_path.name}...")

    fb2_meta = {}
    if ext in ('.fb2', '.fb2.zip'):
        fb2_meta = parse_fb2_meta(book_path)
        chapters = parse_fb2(book_path)
    elif ext == '.epub':
        chapters = parse_epub(book_path)
    else:
        print(f"Ошибка: формат '{ext}' не поддерживается. Используйте .fb2, .fb2.zip или .epub.")
        sys.exit(1)

    if not chapters:
        print("Ошибка: не удалось извлечь ни одной главы.")
        sys.exit(2)

    if fb2_meta.get("book_title") or fb2_meta.get("annotation") or fb2_meta.get("author"):
        intro_parts = []
        if fb2_meta.get("author"):
            intro_parts.append(f"Автор: {fb2_meta['author']}.")
        if fb2_meta.get("book_title"):
            intro_parts.append(f"Книга: {fb2_meta['book_title']}.")
        if fb2_meta.get("series"):
            num = f" номер {fb2_meta['series_number']}" if fb2_meta.get("series_number") else ""
            intro_parts.append(f"Серия: {fb2_meta['series']}{num}.")
        if fb2_meta.get("annotation"):
            intro_parts.append(f"Аннотация. {fb2_meta['annotation']}")
        if intro_parts:
            chapters.insert(0, ("00_Описание", [" ".join(intro_parts)]))

    if fb2_meta.get("author") and fb2_meta.get("book_title"):
        folder_name = sanitize_filename(f"{fb2_meta['author']} - {fb2_meta['book_title']}")
    else:
        folder_name = f"{book_path.stem}_audio"

    if args.outdir:
        base_dir = Path(args.outdir).expanduser().resolve()
    else:
        base_dir = book_path.parent

    out_dir = base_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_id, speaker_desc = SPEAKERS[args.speaker]

    total_chars = sum(sum(len(p) for p in paras) for _, paras in chapters)
    print(f"Найдено глав: {len(chapters)} | Общий объём: {total_chars:,} знаков (~{total_chars / 40000:.2f} а.л.)")
    if fb2_meta.get("book_title"):
        print(f"Книга: {fb2_meta['book_title']}")
    if fb2_meta.get("author"):
        print(f"Автор: {fb2_meta['author']}")
    if fb2_meta.get("series"):
        num = f" #{fb2_meta['series_number']}" if fb2_meta.get("series_number") else ""
        print(f"Серия: {fb2_meta['series']}{num}")
    if fb2_meta.get("annotation"):
        ann = fb2_meta["annotation"]
        print(f"Аннотация: {ann[:200]}{'…' if len(ann) > 200 else ''}")
    print(f"Диктор: {args.speaker} ({speaker_desc}) | Папка вывода: {out_dir}")
    print(f"Нормализатор: ru-normalizr (режим TTS) | Python: {sys.version.split()[0]}")

    print(f"\nЗагрузка модели {MODEL_NAME} на GPU (GT 1030)...")
    model = Model(model_name=MODEL_NAME)
    synth = Synth(model)
    
    print("Прогрев GPU...")
    _ = synth.synth_audio("Прогрев видеокарты.", speaker_id=speaker_id)
    
    print("Прогрев нормализатора...")
    _ = NORMALIZER.normalize("Тест нормализации: 123, 45.6%, 2024 год.")
    
    speech_rate = 1.0 / args.rate if args.rate > 0 else 1.0
    sent_silence = np.zeros(int(SAMPLE_RATE * SENTENCE_SILENCE), dtype=np.int16).tobytes()
    para_silence = np.zeros(int(SAMPLE_RATE * PARAGRAPH_SILENCE), dtype=np.int16).tobytes()

    has_ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
    has_lame = False
    if has_ffmpeg:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        has_lame = "libmp3lame" in (enc.stdout or "")
    use_mp3 = has_ffmpeg and has_lame and not args.no_mp3
    if args.no_mp3:
        print("Примечание: --no-mp3 указан, файлы будут в WAV.")
    elif not has_ffmpeg:
        print("Примечание: ffmpeg не найден, файлы будут сохранены в WAV.")
    elif not has_lame:
        print("Примечание: ffmpeg без libmp3lame, файлы будут сохранены в WAV.")
        print("           Установите: sudo dnf swap ffmpeg-free ffmpeg --allowerasing")

    t_start = time.time()
    processed_chars = 0
    for ch_idx, (title, paras) in enumerate(chapters, 1):
        safe_title = sanitize_filename(title)
        wav_path = out_dir / f"{ch_idx:03d}_{safe_title}.wav"
        mp3_path = wav_path.with_suffix(".mp3")
        if wav_path.exists() or mp3_path.exists():
            existing = mp3_path if mp3_path.exists() else wav_path
            print(f"[{ch_idx}/{len(chapters)}] пропуск (уже есть): {existing.name}")
            processed_chars += sum(len(p) for p in paras)
            continue
        ch_t0 = time.time()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            for p_idx, para in enumerate(paras):
                processed_chars += len(para)
                cleaned = clean_text(para)
                if not cleaned:
                    continue
                chunks = split_into_safe_chunks(cleaned)
                for c_idx, chunk in enumerate(chunks):
                    raw = synth.synth_audio(
                        chunk,
                        speaker_id=speaker_id,
                        noise_level=NOISE_SCALE,
                        speech_rate=speech_rate,
                        duration_noise_level=NOISE_W_SCALE,
                    )
                    if VOLUME != 1.0:
                        raw = np.clip(raw.astype(np.float32) * VOLUME, -32768.0, 32767.0)
                    wf.writeframes(raw.astype(np.int16).tobytes())
                    if c_idx < len(chunks) - 1:
                        wf.writeframes(sent_silence)
                if p_idx < len(paras) - 1:
                    wf.writeframes(para_silence)
        final_file = convert_wav_to_mp3(wav_path) if use_mp3 else wav_path
        elapsed = time.time() - t_start
        pct = (processed_chars / total_chars * 100) if total_chars else 100
        speed = processed_chars / elapsed if elapsed > 0 else 1
        eta = (total_chars - processed_chars) / speed if speed > 0 else 0
        print(f"[{ch_idx}/{len(chapters)}] {final_file.name} ({time.time() - ch_t0:.1f}с) | Прогресс: {pct:.1f}% | Осталось: ~{eta / 60:.1f} мин")
    print(f"\nГотово! Вся книга озвучена за {(time.time() - t_start) / 60:.1f} мин.")


if __name__ == "__main__":
    main()
