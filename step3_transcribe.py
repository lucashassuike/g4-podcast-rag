"""
Step 3 — Transcrever áudios usando Azure OpenAI gpt-4o-transcribe-2.
Suporta: split de arquivos grandes (>25MB) via ffmpeg, resumo, progresso.

Uso: python step3_transcribe.py [--max N] [--workers N]
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI
from tqdm import tqdm

import config

# Limite de arquivo da API (reduzido para evitar "too many tokens")
MAX_FILE_SIZE_MB = 10


def get_ffmpeg_path() -> str:
    """Encontra o executável do ffmpeg (imageio-ffmpeg ou sistema)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # Tenta do PATH do sistema


def get_client() -> AzureOpenAI:
    """Cria cliente Azure OpenAI para transcrição."""
    return AzureOpenAI(
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
    )


def get_audio_duration_seconds(filepath: str) -> float:
    """Obtém duração do áudio em segundos via ffprobe/ffmpeg."""
    ffmpeg = get_ffmpeg_path()
    # Usa ffmpeg para obter duração
    try:
        result = subprocess.run(
            [ffmpeg, "-i", filepath, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60
        )
        # Parse duration from stderr (ffmpeg outputs info to stderr)
        for line in result.stderr.split("\n"):
            if "Duration:" in line:
                # Format: Duration: HH:MM:SS.xx
                dur_str = line.split("Duration:")[1].split(",")[0].strip()
                parts = dur_str.split(":")
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    return 0


def split_audio(filepath: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> list[str]:
    """
    Divide áudio grande em chunks menores usando ffmpeg.
    Retorna lista de caminhos (originais ou temporários).
    """
    file_size = Path(filepath).stat().st_size
    if file_size <= max_size_mb * 1024 * 1024:
        return [filepath]

    ffmpeg = get_ffmpeg_path()
    print(f"   [SPLIT] Arquivo grande ({file_size / 1_048_576:.0f}MB), dividindo...")

    # Calcula quantas partes
    num_chunks = math.ceil(file_size / (max_size_mb * 1024 * 1024))

    # Obtém duração total
    duration = get_audio_duration_seconds(filepath)
    if duration <= 0:
        # Fallback: estima ~1MB por minuto de áudio comprimido
        duration = (file_size / 1_048_576) * 60

    chunk_duration = duration / num_chunks
    chunks = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()

        cmd = [
            ffmpeg,
            "-y",                        # Overwrite
            "-i", filepath,
            "-ss", str(start_time),      # Start time
            "-t", str(chunk_duration),   # Duration
            "-acodec", "libmp3lame",
            "-ab", "64k",               # Bitrate baixo para ficar < 25MB
            "-ar", "16000",             # 16kHz suficiente para speech
            "-ac", "1",                 # Mono
            "-loglevel", "error",
            tmp.name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and Path(tmp.name).stat().st_size > 1000:
            chunks.append(tmp.name)
        else:
            Path(tmp.name).unlink(missing_ok=True)

    print(f"   [SPLIT] Dividido em {len(chunks)} partes")
    return chunks


def transcribe_file(client: AzureOpenAI, filepath: str, use_diarize: bool = False) -> dict:
    """
    Transcreve um único arquivo de áudio.
    Se use_diarize=True, usa o modelo diarize que segmenta por turnos de fala.
    Retorna dict com 'text' e 'turns' (lista de turnos de fala).
    """
    if use_diarize:
        return _transcribe_diarize(filepath)

    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model=config.AZURE_TRANSCRIPTION_MODEL,
            file=f,
            language="pt",
        )

    text = result.text if hasattr(result, "text") else str(result)
    return {"text": text, "turns": []}


def _transcribe_diarize(filepath: str) -> dict:
    """Transcreve via REST API com modelo diarize (turnos de fala)."""
    import requests as req

    url = (
        f"{config.AZURE_ENDPOINT}/openai/deployments/{config.AZURE_DIARIZE_MODEL}"
        f"/audio/transcriptions?api-version={config.AZURE_API_VERSION}"
    )

    with open(filepath, "rb") as f:
        resp = req.post(
            url,
            headers={"api-key": config.AZURE_API_KEY},
            files={"file": f},
            data={
                "language": "pt",
                "response_format": "json",
                "chunking_strategy": json.dumps({"type": "server_vad"}),
            },
            timeout=600,
        )

    resp.raise_for_status()
    data = resp.json()
    text = data.get("text", "")

    # Cada \n é uma mudança de speaker/turno
    turns = [t.strip() for t in text.split("\n") if t.strip()]
    plain_text = " ".join(turns)

    return {"text": plain_text, "turns": turns}


def _format_ts(seconds: float) -> str:
    """Formata segundos em HH:MM:SS ou MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def estimate_segments(text: str, duration_seconds: int) -> list[dict]:
    """
    Estima timestamps baseado na posição das sentenças no texto
    e na duração total do episódio.
    """
    if not text or duration_seconds <= 0:
        return []

    # Divide em sentenças
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return []

    segments = []
    char_pos = 0
    for sent in sentences:
        start = (char_pos / total_chars) * duration_seconds
        char_pos += len(sent)
        end = (char_pos / total_chars) * duration_seconds
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "timestamp": _format_ts(start),
            "text": sent,
        })

    return segments


def transcribe_episode(episode: dict) -> dict:
    """Transcreve um episódio completo (com split se necessário)."""
    ep_id = episode["id"]
    title = episode["title"]
    local_file = episode.get("local_file", "")
    transcript_file = config.TRANSCRIPTS_DIR / f"{ep_id}.json"

    # Pula se já transcrito com segments; enriquece se tem texto mas sem segments
    if transcript_file.exists():
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("segments"):
                return {"id": ep_id, "status": "skipped"}
            # Tem texto mas sem segments — enriquece com timestamps estimados
            if existing.get("text"):
                duration = existing.get("duration", episode.get("duration", 0))
                segments = estimate_segments(existing["text"], duration)
                existing["segments"] = segments
                existing["segment_count"] = len(segments)
                existing["text_timestamped"] = "\n".join(
                    f"[{seg['timestamp']}] {seg['text']}" for seg in segments
                ) if segments else existing["text"]
                existing["timestamps_estimated"] = True
                with open(transcript_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                return {"id": ep_id, "status": "enriched", "segments": len(segments)}
        except (json.JSONDecodeError, KeyError):
            pass

    if not local_file or not Path(local_file).exists():
        return {"id": ep_id, "status": "error", "error": "Arquivo de audio nao encontrado"}

    try:
        client = get_client()
        # Diarize desabilitado por ora — usar modelo padrão com split para compatibilidade
        use_diarize = False

        # Divide se necessário
        chunks = split_audio(local_file)

        # Transcreve cada parte
        texts = []
        all_turns = []
        for i, chunk_path in enumerate(chunks):
            result = transcribe_file(client, chunk_path, use_diarize=use_diarize)
            texts.append(result["text"])
            all_turns.extend(result["turns"])

            # Limpa arquivos temporários (não o original)
            if chunk_path != local_file:
                Path(chunk_path).unlink(missing_ok=True)

        full_text = " ".join(texts)

        # Estima timestamps baseado na duração do episódio
        duration = episode.get("duration", 0)
        segments = estimate_segments(full_text, duration)

        # Texto com timestamps estimados
        text_with_ts = "\n".join(
            f"[{seg['timestamp']}] {seg['text']}" for seg in segments
        ) if segments else full_text

        # Texto com turnos de fala (se diarize)
        text_with_turns = "\n\n".join(
            f"[Turno {i+1}] {turn}" for i, turn in enumerate(all_turns)
        ) if all_turns else ""

        # Salva transcrição com metadados
        transcript_data = {
            "id": ep_id,
            "title": title,
            "guest": episode.get("guest", ""),
            "date_published": episode.get("date_published", 0),
            "duration": duration,
            "text": full_text,
            "text_timestamped": text_with_ts,
            "text_turns": text_with_turns,
            "segments": segments,
            "turns": all_turns,
            "word_count": len(full_text.split()),
            "segment_count": len(segments),
            "turn_count": len(all_turns),
            "chunks_used": len(chunks),
            "timestamps_estimated": True,
            "diarized": use_diarize,
        }

        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        return {
            "id": ep_id,
            "status": "ok",
            "words": transcript_data["word_count"],
            "segments": len(segments),
            "turns": len(all_turns),
        }

    except Exception as e:
        return {"id": ep_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Transcrever episodios do G4 Podcast")
    parser.add_argument("--max", type=int, default=0, help="Maximo de episodios (0=todos)")
    parser.add_argument("--workers", type=int, default=2, help="Transcricoes simultaneas")
    parser.add_argument("--retranscribe", action="store_true",
                        help="Re-transcreve episodios sem timestamps (segments)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3 -- Transcricao com Azure OpenAI gpt-4o-transcribe-2")
    print("=" * 60)

    if not config.EPISODES_FILE.exists():
        print("ERRO: episodes.json nao encontrado. Rode step1 primeiro.")
        sys.exit(1)

    with open(config.EPISODES_FILE, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    # Filtra só os que têm arquivo local
    episodes = [e for e in episodes if e.get("local_file")]
    if args.max > 0:
        episodes = episodes[:args.max]

    print(f"  {len(episodes)} episodios para transcrever")

    # Conta já transcritos (com segments/timestamps)
    already = 0
    for e in episodes:
        tf = config.TRANSCRIPTS_DIR / f"{e['id']}.json"
        if tf.exists():
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("segments"):
                    already += 1
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"  {already} ja transcritos (serao pulados)")
    print()

    results = {"ok": 0, "skipped": 0, "enriched": 0, "error": 0}
    errors = []

    # Transcrição com paralelismo limitado (API tem rate limits)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(transcribe_episode, ep): ep
            for ep in episodes
        }

        with tqdm(total=len(futures), desc="Transcrevendo", unit="ep") as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                results[status] = results.get(status, 0) + 1
                if status == "error":
                    errors.append(result)
                pbar.update(1)

    print(f"\nTranscricao concluida:")
    print(f"   Novos:       {results['ok']}")
    print(f"   Enriquecidos:{results['enriched']}")
    print(f"   Pulados:     {results['skipped']}")
    print(f"   Erros:       {results['error']}")

    if errors:
        print("\n-- Erros --")
        for e in errors[:10]:
            print(f"  ID {e['id']}: {e.get('error', 'desconhecido')[:100]}")

    # Estatísticas
    total_words = 0
    for tf in config.TRANSCRIPTS_DIR.glob("*.json"):
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_words += data.get("word_count", 0)

    total_done = results["ok"] + already
    print(f"\nTotal de palavras transcritas: {total_words:,}")
    if total_done > 0:
        print(f"Media por episodio: {total_words // total_done:,} palavras")


if __name__ == "__main__":
    main()
