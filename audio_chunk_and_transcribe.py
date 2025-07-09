import os 
import time
import io
import base64
import json
import re
from datetime import datetime
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()  
LLM_API_KEY = os.getenv("LLM_API_KEY") 
LLM_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash"

# Client LLM
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE_URL)

# parser les timestamps
ts_ms_re = re.compile(r"^(?:(?P<mn>\d+)m)?(?P<sec>\d+)s(?P<ms>\d+)ms$")
ts_re = re.compile(r"^(?:(?P<mn>\d+)mn)?(?P<sec>\d+(?:\.\d+)?)s$")


def parse_timestamp(ts: str) -> float | None:
    """Convertit un timestamp en float (secondes)."""
    m = ts_ms_re.match(ts)
    if m:
        mn = int(m.group('mn') or 0)
        sec = int(m.group('sec'))
        ms = int(m.group('ms'))
        return mn * 60 + sec + ms / 1000.0
    m = ts_re.match(ts)
    if m:
        mn = int(m.group('mn') or 0)
        sec = float(m.group('sec'))
        return mn * 60 + sec
    return None


def format_timestamp(seconds: float) -> str:
    """Formate un float (secondes) en 'XmnY.ZZZs'."""
    mn = int(seconds // 60)
    sc = seconds - mn * 60
    return f"{mn}mn{sc:.3f}s"


def chunk_audio(file_path: str, chunk_minutes: int = 5) -> list[tuple[str, int]]:
    """
    Découpe l'audio en chunks; retourne [(path, offset_ms), ...].
    """
    audio = AudioSegment.from_file(file_path)
    ms_chunk = chunk_minutes * 60_000
    base = os.path.join(os.getcwd(), "processed_chunks")
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(base, f"chunks_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    chunks = []
    for idx, start_ms in enumerate(range(0, len(audio), ms_chunk), start=1):
        seg = audio[start_ms:start_ms + ms_chunk]
        path = os.path.join(out_dir, f"chunk_{idx}.wav")
        seg.export(path, format="wav")
        chunks.append((path, start_ms))
    print(f"[chunk_audio] créé {len(chunks)} chunks dans {out_dir}")
    return chunks


def transcribe_audio(path: str) -> list[dict]:
    """
    Transcrit un chunk via l'API; renvoie [{'speaker', 'start_sec', 'end_sec', 'text'}, ...].
    """
    audio = AudioSegment.from_file(path)
    buf = io.BytesIO(); audio.export(buf, format="wav")
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "Transcribe the following audio conversation. "
        "Return JSON: array of {speaker, start, end, text}. "
        "Timestamps relative to the chunk, format 'XmnY.ZZZs' or 'XmYsZms'."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Please transcribe this segment."},
            {"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{b64}"}}
        ]}
    ]
    resp = llm_client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0, max_tokens=800_000
    )
    raw = resp.choices[0].message.content.strip()
    # Strip code fences
    json_str = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)
    # Remove trailing commas before closing brackets/braces
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[transcribe_audio] Échec JSON pour {path}: {e}\nContenu nettoyé:\n{json_str}")
        return []

    segments = []
    for item in data:
        start_sec = parse_timestamp(item.get('start', ''))
        end_sec = parse_timestamp(item.get('end', ''))
        if start_sec is None or end_sec is None:
            print(f"[transcribe_audio] Skip mal formaté: {item.get('start')}, {item.get('end')}")
            continue
        segments.append({
            'speaker': item.get('speaker'),
            'start_sec': start_sec,
            'end_sec': end_sec,
            'text': item.get('text') or item.get('text_content') or item.get('text_text_normalized')
        })
    print(f"[transcribe_audio] {len(segments)} segments parsés pour {os.path.basename(path)}")
    return segments


def run_pipeline(input_file: str, chunk_minutes: int = 5, save_json: bool = True) -> dict:
    """
    Pipeline complet: découpe, transcription, ajustement timestamp.
    """
    all_transcripts = {}
    chunks = chunk_audio(input_file, chunk_minutes)
    if not chunks:
        print("Aucun chunk généré, vérifiez le fichier audio.")
        return {}

    for path, offset_ms in chunks:
        offset_s = offset_ms / 1000.0
        print(f"[run_pipeline] Transcribing {os.path.basename(path)} (offset {offset_s}s)")
        parts = transcribe_audio(path)
        if not parts:
            print(f"[run_pipeline] Aucune transcription pour {path}")
        for seg in parts:
            seg['start'] = format_timestamp(seg['start_sec'] + offset_s)
            seg['end']   = format_timestamp(seg['end_sec']   + offset_s)
            del seg['start_sec'], seg['end_sec']
        all_transcripts[os.path.basename(path)] = parts

    if save_json:
        
        output_dir = os.path.join(os.getcwd(), "Transcription_json")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"transcription_full_{int(time.time())}.json"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_transcripts, f, ensure_ascii=False, indent=2)
        print(f"[run_pipeline] Résultats sauvegardés dans {out_path}")
    return all_transcripts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Découpe & transcrit un audio avec timestamps continus.")
    parser.add_argument('file', help="Chemin vers fichier audio")
    parser.add_argument('--minutes', type=int, default=5, help="Durée de chaque segment (min)")
    parser.add_argument('--no-save', dest='save_json', action='store_false', help="Ne pas sauver JSON")
    args = parser.parse_args()
    run_pipeline(args.file, chunk_minutes=args.minutes, save_json=args.save_json)
