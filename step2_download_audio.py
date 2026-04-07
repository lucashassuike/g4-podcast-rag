"""
Step 2 — Download de todos os MP3s dos episódios.
Suporta resumo (pula arquivos já baixados) e mostra progresso.

Uso: python step2_download_audio.py [--max N]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

import config


def sanitize_filename(title: str, ep_id: int) -> str:
    """Cria nome de arquivo seguro a partir do título."""
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe = safe.strip()[:80]
    return f"{ep_id}_{safe}.mp3"


def download_episode(episode: dict, output_dir: Path) -> dict:
    """Baixa um episódio. Retorna status."""
    url = episode["audio_url"]
    filename = sanitize_filename(episode["title"], episode["id"])
    filepath = output_dir / filename

    # Pula se já existe e tem tamanho razoável (> 100KB)
    if filepath.exists() and filepath.stat().st_size > 100_000:
        return {"id": episode["id"], "status": "skipped", "file": str(filepath)}

    try:
        resp = requests.get(url, stream=True, timeout=300, allow_redirects=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))

        with open(filepath, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        # Atualiza o episódio com o caminho local
        episode["local_file"] = str(filepath)

        return {
            "id": episode["id"],
            "status": "ok",
            "file": str(filepath),
            "size_mb": round(downloaded / 1_048_576, 1),
        }

    except Exception as e:
        return {"id": episode["id"], "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Download episódios do G4 Podcast")
    parser.add_argument("--max", type=int, default=0, help="Máximo de episódios (0=todos)")
    parser.add_argument("--workers", type=int, default=4, help="Downloads simultâneos")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 2 — Download dos áudios")
    print("=" * 60)

    if not config.EPISODES_FILE.exists():
        print("❌ Arquivo episodes.json não encontrado. Rode step1 primeiro.")
        sys.exit(1)

    with open(config.EPISODES_FILE, "r", encoding="utf-8") as f:
        all_episodes = json.load(f)

    episodes = all_episodes
    if args.max > 0:
        episodes = all_episodes[:args.max]
        print(f"🔢 Limitado a {args.max} episódios")

    print(f"📥 {len(episodes)} episódios para processar")
    print(f"👷 {args.workers} downloads simultâneos")
    print()

    results = {"ok": 0, "skipped": 0, "error": 0}
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_episode, ep, config.AUDIO_DIR): ep
            for ep in episodes
        }

        with tqdm(total=len(futures), desc="Baixando", unit="ep") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results[result["status"]] += 1
                if result["status"] == "error":
                    errors.append(result)
                pbar.update(1)
                pbar.set_postfix(ok=results["ok"], skip=results["skipped"], err=results["error"])

    # Atualiza episodes.json com caminhos locais (sempre salva todos)
    for ep in all_episodes:
        filename = sanitize_filename(ep["title"], ep["id"])
        filepath = config.AUDIO_DIR / filename
        if filepath.exists():
            ep["local_file"] = str(filepath)

    with open(config.EPISODES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_episodes, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Download concluído:")
    print(f"   ✓ Novos:   {results['ok']}")
    print(f"   ⏭ Pulados: {results['skipped']}")
    print(f"   ✗ Erros:   {results['error']}")

    if errors:
        print("\n── Erros ──")
        for e in errors:
            print(f"  ID {e['id']}: {e.get('error', 'desconhecido')}")

    # Espaço usado
    total_size = sum(
        f.stat().st_size for f in config.AUDIO_DIR.glob("*.mp3")
    )
    print(f"\n💾 Espaço total em disco: {total_size / 1_073_741_824:.1f} GB")


if __name__ == "__main__":
    main()
