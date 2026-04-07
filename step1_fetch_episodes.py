"""
Step 1 — Buscar todos os episódios via Podcast Index API.
Salva em data/episodes.json com metadados completos.

Uso: python step1_fetch_episodes.py
"""

import hashlib
import json
import time

import requests

import config


def get_auth_headers() -> dict:
    """Gera headers de autenticação para a Podcast Index API."""
    if not config.PI_API_KEY or not config.PI_API_SECRET:
        raise ValueError(
            "⚠️  Defina PODCAST_INDEX_API_KEY e PODCAST_INDEX_API_SECRET no .env\n"
            "   Pegue ambos em: https://api.podcastindex.org"
        )

    epoch = str(int(time.time()))
    data_to_hash = config.PI_API_KEY + config.PI_API_SECRET + epoch
    sha1 = hashlib.sha1(data_to_hash.encode("utf-8")).hexdigest()

    return {
        "User-Agent": "G4PodcastRAG/1.0",
        "X-Auth-Key": config.PI_API_KEY,
        "X-Auth-Date": epoch,
        "Authorization": sha1,
    }


def fetch_all_episodes() -> list[dict]:
    """
    Busca todos os episódios do feed via Podcast Index API.
    A API suporta max=1000 por chamada. Se houver mais,
    faz chamadas adicionais usando o campo 'since'.
    """
    all_episodes = []
    since = 0  # epoch — 0 = desde o início
    page = 1

    while True:
        print(f"📡 Buscando página {page} (desde epoch {since})...")
        url = f"{config.PI_BASE_URL}/episodes/byfeedid"
        params = {
            "id": config.PI_FEED_ID,
            "max": 1000,
            "fulltext": "",
            "since": since,
        }

        resp = requests.get(url, headers=get_auth_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            break

        all_episodes.extend(items)
        print(f"   ↳ {len(items)} episódios nesta página (total: {len(all_episodes)})")

        # Se retornou menos que 1000, acabou
        if len(items) < 1000:
            break

        # Próxima página: usar o datePublished mais antigo como 'since'
        # Na verdade, o Podcast Index retorna em ordem cronológica reversa
        # Então pegamos o mais antigo e buscamos antes dele
        oldest = min(items, key=lambda x: x.get("datePublished", 0))
        since = oldest.get("datePublished", 0) - 1
        page += 1

    return all_episodes


def normalize_episodes(raw_episodes: list[dict]) -> list[dict]:
    """Normaliza campos para formato consistente."""
    episodes = []
    for ep in raw_episodes:
        title = ep.get("title", "Sem título")

        # Tenta extrair o convidado do título (padrão "TEMA COM/- CONVIDADO | SÉRIE")
        guest = ""
        for sep in [" COM ", " - ", " | "]:
            if sep in title.upper():
                parts = title.upper().split(sep)
                if len(parts) >= 2:
                    # Heurística: o convidado geralmente está na segunda parte
                    candidate = title.split(sep if sep != title.upper().split(sep)[0] else sep)
                    break

        # Extrai guest name de campos persons se disponível
        persons = ep.get("persons", [])
        if persons:
            guest = ", ".join(p.get("name", "") for p in persons if p.get("name"))

        episodes.append({
            "id": ep.get("id", 0),
            "title": title,
            "guest": guest,
            "description": ep.get("description", ""),
            "date_published": ep.get("datePublished", 0),
            "duration": ep.get("duration", 0),
            "audio_url": ep.get("enclosureUrl", ""),
            "audio_type": ep.get("enclosureType", "audio/mpeg"),
            "audio_size": ep.get("enclosureLength", 0),
            "episode_url": ep.get("link", ""),
            "image": ep.get("image", "") or ep.get("feedImage", ""),
        })

    # Ordena por data (mais antigo primeiro)
    episodes.sort(key=lambda x: x["date_published"])
    return episodes


def main():
    print("=" * 60)
    print("STEP 1 — Buscar episódios do G4 Podcasts")
    print("=" * 60)

    raw = fetch_all_episodes()
    print(f"\n✅ Total bruto: {len(raw)} episódios encontrados")

    episodes = normalize_episodes(raw)

    # Filtra episódios sem URL de áudio
    valid = [e for e in episodes if e["audio_url"]]
    skipped = len(episodes) - len(valid)
    if skipped:
        print(f"⚠️  {skipped} episódios sem URL de áudio (ignorados)")

    # Salva
    config.EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.EPISODES_FILE, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Salvo em: {config.EPISODES_FILE}")
    print(f"📊 {len(valid)} episódios com áudio disponível")

    # Preview
    print("\n── Primeiros 5 episódios ──")
    for ep in valid[:5]:
        print(f"  • [{ep['date_published']}] {ep['title'][:70]}")

    print("\n── Últimos 5 episódios ──")
    for ep in valid[-5:]:
        print(f"  • [{ep['date_published']}] {ep['title'][:70]}")


if __name__ == "__main__":
    main()
