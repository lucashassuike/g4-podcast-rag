"""
Step 3b — Transcrição com diarização (separação de speakers + timestamps).
Usa Azure Speech SDK ConversationTranscriber.

Requer: pip install azure-cognitiveservices-speech imageio-ffmpeg

Uso: python step3_transcribe_diarized.py [--max N] [--force]
  --max N    Máximo de episódios a processar (0=todos)
  --force    Re-transcreve mesmo se já existe transcrição diarizada
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

import config


def get_ffmpeg_path() -> str:
    """Encontra o executável do ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def convert_mp3_to_wav(mp3_path: str) -> str:
    """Converte MP3 para WAV mono 16kHz 16-bit (requisito do ConversationTranscriber)."""
    ffmpeg = get_ffmpeg_path()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        ffmpeg,
        "-y",
        "-i", mp3_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg falhou: {result.stderr[:200]}")

    return tmp.name


def ticks_to_seconds(offset_ticks: int) -> float:
    """Converte ticks (100ns) para segundos."""
    return offset_ticks / 10_000_000


def format_timestamp(seconds: float) -> str:
    """Formata segundos em HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcribe_with_diarization(wav_path: str) -> list[dict]:
    """
    Transcreve áudio WAV usando ConversationTranscriber.
    Retorna lista de segmentos: {speaker, text, offset_seconds, timestamp}.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY,
        region=config.AZURE_SPEECH_REGION,
    )
    speech_config.speech_recognition_language = "pt-BR"

    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)

    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    segments = []
    done_event = threading.Event()
    error_msg = None

    def on_transcribed(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            offset_s = ticks_to_seconds(evt.result.offset)
            segments.append({
                "speaker": evt.result.speaker_id,
                "text": evt.result.text,
                "offset_seconds": round(offset_s, 2),
                "timestamp": format_timestamp(offset_s),
            })

    def on_canceled(evt):
        nonlocal error_msg
        if evt.reason == speechsdk.CancellationReason.Error:
            error_msg = f"Erro: {evt.error_details}"
        done_event.set()

    def on_stopped(evt):
        done_event.set()

    transcriber.transcribed.connect(on_transcribed)
    transcriber.canceled.connect(on_canceled)
    transcriber.session_stopped.connect(on_stopped)

    transcriber.start_transcribing_async().get()

    # Aguarda finalização (timeout = duração estimada + margem)
    # Para episódios longos, pode levar bastante tempo
    if not done_event.wait(timeout=7200):  # 2 horas max
        transcriber.stop_transcribing_async().get()
        raise TimeoutError("Transcrição excedeu timeout de 2 horas")

    transcriber.stop_transcribing_async().get()

    if error_msg:
        raise RuntimeError(error_msg)

    return segments


def identify_speakers(segments: list[dict]) -> dict[str, str]:
    """
    Tenta identificar quem é host e quem é convidado.
    Heurística: Guest-1 geralmente é quem fala primeiro (host).
    """
    if not segments:
        return {}

    speaker_counts = {}
    for seg in segments:
        sp = seg["speaker"]
        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1

    # Ordena por primeira aparição
    first_appearance = {}
    for seg in segments:
        sp = seg["speaker"]
        if sp not in first_appearance:
            first_appearance[sp] = seg["offset_seconds"]

    sorted_speakers = sorted(first_appearance.items(), key=lambda x: x[1])

    labels = {}
    for i, (speaker_id, _) in enumerate(sorted_speakers):
        if i == 0:
            labels[speaker_id] = "Host"
        else:
            labels[speaker_id] = f"Convidado-{i}"

    return labels


def build_transcript_text(segments: list[dict], speaker_labels: dict[str, str]) -> str:
    """Monta texto completo com marcações de speaker e timestamp."""
    lines = []
    current_speaker = None

    for seg in segments:
        label = speaker_labels.get(seg["speaker"], seg["speaker"])
        ts = seg["timestamp"]

        if seg["speaker"] != current_speaker:
            current_speaker = seg["speaker"]
            lines.append(f"\n[{ts}] {label}:")

        lines.append(seg["text"])

    return "\n".join(lines).strip()


def process_episode(episode: dict, force: bool = False) -> dict:
    """Processa um episódio: converte para WAV, transcreve com diarização."""
    ep_id = episode["id"]
    title = episode["title"]
    local_file = episode.get("local_file", "")
    transcript_file = config.TRANSCRIPTS_DIR / f"{ep_id}.json"

    # Verifica se já tem transcrição diarizada
    if transcript_file.exists() and not force:
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("segments"):
                return {"id": ep_id, "status": "skipped", "reason": "ja diarizado"}
        except (json.JSONDecodeError, KeyError):
            pass

    if not local_file or not Path(local_file).exists():
        return {"id": ep_id, "status": "error", "error": "Arquivo de audio nao encontrado"}

    wav_path = None
    try:
        print(f"  [{ep_id}] Convertendo para WAV: {title[:60]}...", flush=True)
        wav_path = convert_mp3_to_wav(local_file)

        print(f"  [{ep_id}] Transcrevendo com diarizacao...", flush=True)
        segments = transcribe_with_diarization(wav_path)

        if not segments:
            return {"id": ep_id, "status": "error", "error": "Nenhum segmento retornado"}

        # Identifica speakers
        speaker_labels = identify_speakers(segments)

        # Texto formatado com speakers e timestamps
        formatted_text = build_transcript_text(segments, speaker_labels)

        # Texto plano (compatível com RAG existente)
        plain_text = " ".join(seg["text"] for seg in segments)

        # Salva transcrição enriquecida
        transcript_data = {
            "id": ep_id,
            "title": title,
            "guest": episode.get("guest", ""),
            "date_published": episode.get("date_published", 0),
            "duration": episode.get("duration", 0),
            "text": plain_text,
            "text_diarized": formatted_text,
            "segments": segments,
            "speaker_labels": speaker_labels,
            "word_count": len(plain_text.split()),
            "segment_count": len(segments),
            "speaker_count": len(speaker_labels),
        }

        with open(transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        print(
            f"  [{ep_id}] OK - {len(segments)} segmentos, "
            f"{len(speaker_labels)} speakers, "
            f"{transcript_data['word_count']} palavras",
            flush=True,
        )

        return {
            "id": ep_id,
            "status": "ok",
            "segments": len(segments),
            "speakers": len(speaker_labels),
            "words": transcript_data["word_count"],
        }

    except Exception as e:
        return {"id": ep_id, "status": "error", "error": str(e)[:300]}

    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Transcrever episodios com diarizacao (Azure Speech SDK)"
    )
    parser.add_argument("--max", type=int, default=0, help="Maximo de episodios (0=todos)")
    parser.add_argument("--force", action="store_true", help="Re-transcreve mesmo se ja existe")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 3b -- Transcricao com Diarizacao (Azure Speech SDK)")
    print("=" * 60)

    if not config.EPISODES_FILE.exists():
        print("ERRO: episodes.json nao encontrado. Rode step1 primeiro.")
        sys.exit(1)

    with open(config.EPISODES_FILE, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    # Filtra episódios com arquivo local
    episodes = [e for e in episodes if e.get("local_file")]
    if args.max > 0:
        episodes = episodes[:args.max]

    print(f"  {len(episodes)} episodios com audio")

    # Conta já diarizados
    already = 0
    if not args.force:
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

    print(f"  {already} ja diarizados (serao pulados)")
    print()

    results = {"ok": 0, "skipped": 0, "error": 0}
    errors = []

    # Processa sequencialmente (Speech SDK usa streaming, 1 por vez é mais estável)
    for i, ep in enumerate(episodes, 1):
        print(f"\n--- [{i}/{len(episodes)}] ---", flush=True)
        result = process_episode(ep, force=args.force)
        results[result["status"]] += 1

        if result["status"] == "error":
            errors.append(result)
            print(f"  ERRO: {result.get('error', '?')[:150]}", flush=True)

    print(f"\n{'=' * 60}")
    print(f"Diarizacao concluida:")
    print(f"   Novos:   {results['ok']}")
    print(f"   Pulados: {results['skipped']}")
    print(f"   Erros:   {results['error']}")

    if errors:
        print(f"\n-- Erros ({len(errors)}) --")
        for e in errors[:10]:
            print(f"  ID {e['id']}: {e.get('error', '?')[:120]}")

    # Estatísticas
    total_segments = 0
    total_words = 0
    for tf in config.TRANSCRIPTS_DIR.glob("*.json"):
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_segments += data.get("segment_count", 0)
            total_words += data.get("word_count", 0)
        except Exception:
            pass

    print(f"\nTotal: {total_words:,} palavras em {total_segments:,} segmentos")


if __name__ == "__main__":
    main()
