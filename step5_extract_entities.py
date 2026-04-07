"""
Step 5 -- Extrair entidades e relacionamentos das transcricoes usando GPT-4o.
Produz um JSON por episodio com pessoas, empresas, conceitos, livros, metricas
e relacionamentos.

Uso: python step5_extract_entities.py [--max N] [--force] [--workers N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import tiktoken
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config

# ── Hosts conhecidos do G4 ─────────────────────────────────
KNOWN_HOSTS = {"Tallis Gomes", "Alfredo Soares", "Bruno Nardon", "Tony Celestino"}

# ── Prompt de extracao ─────────────────────────────────────

SYSTEM_PROMPT = """\
Voce e um sistema de extracao de entidades para o G4 Podcast, um podcast brasileiro de negocios.
Extraia TODAS as entidades e relacionamentos do trecho de transcricao fornecido.

Regras:
- Nomes completos sempre que possivel (ex: "Tallis Gomes", nao apenas "Tallis")
- Empresas: nome completo/comum (ex: "XP Inc", "G4 Educacao", "Buser")
- Conceitos: frameworks, estrategias, termos de negocios (ex: "OKR", "flywheel", "product-market fit", "cultura organizacional")
- Livros: titulo e autor quando mencionados
- Metricas: numeros notaveis com contexto (faturamento, crescimento, tamanho de equipe)
- Relacionamentos: conecte entidades onde o texto indica conexao
- Role de pessoas: "host" (Tallis Gomes, Alfredo Soares, Bruno Nardon, Tony Celestino), "guest" (convidado principal), "mentioned" (citado na conversa)
- Extraia APENAS entidades explicitamente mencionadas no texto; nao invente

Tipos de relacionamento permitidos:
- works_at, founded, invested_in, mentored_by
- mentioned, discussed, recommends
- operates_in, related_to, competed_with

Responda APENAS com JSON valido no schema especificado. Sem texto extra."""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["host", "guest", "mentioned"]},
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["name", "role", "company", "title"],
                "additionalProperties": False,
            },
        },
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["name", "industry", "context"],
                "additionalProperties": False,
            },
        },
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["name", "category"],
                "additionalProperties": False,
            },
        },
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                },
                "required": ["title", "author"],
                "additionalProperties": False,
            },
        },
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["value", "context"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["person", "company", "concept", "book"]},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                    "target_type": {"type": "string", "enum": ["person", "company", "concept", "book"]},
                },
                "required": ["source", "source_type", "relation", "target", "target_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["people", "companies", "concepts", "books", "metrics", "relationships"],
    "additionalProperties": False,
}


# ── Limpeza de texto ───────────────────────────────────────

# Padroes de ads/intros do G4 que nao agregam valor para entidades
AD_PATTERNS = [
    r"Escaneie o QR Code.*?(?:episodio|Skills|G4)",
    r"clique no link na descri[cç][aã]o.*?(?:episodio|Skills)",
    r"teste gratuitamente o G4 Skills.*?(?:time|empresa)",
    r"G4 imers[aã]o e mentoria.*?(?:inscri[cç][aã]o|link|processo)",
    r"g4educacao\.com.*?(?:mentoria|link)",
    r"Aumentar a produtividade.*?G4 Skills",
    r"Desenvolva o seu time com o G4 Skills",
    r"Libere o seu acesso de 7 dias gr[aá]tis",
]


def clean_text(text: str) -> str:
    """Limpa texto da transcricao: remove ads, normaliza."""
    if not text:
        return ""
    # Remove padroes de propaganda
    for pattern in AD_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Normaliza whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chunking ───────────────────────────────────────────────

def chunk_for_extraction(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """Divide texto em chunks para extracao de entidades."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start = end - overlap

    return [c for c in chunks if len(c) > 50]


# ── API Call ───────────────────────────────────────────────

def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_CHAT_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
    )


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(config.KG_MAX_RETRIES))
def extract_from_chunk(client: AzureOpenAI, title: str, chunk_text: str) -> dict:
    """Extrai entidades de um chunk via GPT-4o."""
    user_msg = f'Episodio: "{title}"\n\nTranscricao:\n---\n{chunk_text}\n---'

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chunk_entities",
                "strict": True,
                "schema": EXTRACTION_SCHEMA,
            },
        },
        temperature=0.1,
        max_tokens=4096,
    )

    result = json.loads(response.choices[0].message.content)
    time.sleep(config.KG_BATCH_DELAY)
    return result


# ── Merge de chunks ────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Normaliza nome para deduplicacao."""
    return re.sub(r"\s+", " ", name.strip().title())


def merge_chunk_results(chunks_results: list[dict]) -> dict:
    """Combina resultados de multiplos chunks em um unico resultado."""
    people = {}
    companies = {}
    concepts = {}
    books = {}
    metrics = []
    relationships = set()

    for result in chunks_results:
        for p in result.get("people", []):
            key = normalize_name(p["name"])
            if key in people:
                # Merge: prioriza roles mais especificas
                existing = people[key]
                if p.get("role") == "guest" or (p.get("role") == "host" and existing.get("role") == "mentioned"):
                    existing["role"] = p["role"]
                if p.get("company") and not existing.get("company"):
                    existing["company"] = p["company"]
                if p.get("title") and not existing.get("title"):
                    existing["title"] = p["title"]
            else:
                people[key] = {
                    "name": key,
                    "role": p.get("role", "mentioned"),
                    "company": p.get("company", ""),
                    "title": p.get("title", ""),
                }

        for c in result.get("companies", []):
            key = normalize_name(c["name"])
            if key not in companies:
                companies[key] = {
                    "name": key,
                    "industry": c.get("industry", ""),
                    "context": c.get("context", ""),
                }

        for c in result.get("concepts", []):
            key = c["name"].strip().lower()
            if key not in concepts:
                concepts[key] = {
                    "name": c["name"].strip(),
                    "category": c.get("category", ""),
                }

        for b in result.get("books", []):
            key = b["title"].strip().lower()
            if key not in books:
                books[key] = {
                    "title": b["title"].strip(),
                    "author": b.get("author", ""),
                }

        for m in result.get("metrics", []):
            metrics.append(m)

        for r in result.get("relationships", []):
            rel_key = (
                normalize_name(r["source"]),
                r["relation"],
                normalize_name(r["target"]),
            )
            if rel_key not in relationships:
                relationships.add(rel_key)

    # Converte relationships de volta para lista de dicts
    rel_list = [
        {
            "source": s,
            "source_type": "person",  # simplificado
            "relation": rel,
            "target": t,
            "target_type": "company",
        }
        for s, rel, t in relationships
    ]

    # Corrige source/target types baseado nas entidades conhecidas
    people_names = set(people.keys())
    company_names = set(companies.keys())
    concept_names = {c["name"] for c in concepts.values()}

    for r in rel_list:
        if r["source"] in people_names:
            r["source_type"] = "person"
        elif r["source"] in company_names:
            r["source_type"] = "company"
        elif r["source"].lower() in {c.lower() for c in concept_names}:
            r["source_type"] = "concept"

        if r["target"] in people_names:
            r["target_type"] = "person"
        elif r["target"] in company_names:
            r["target_type"] = "company"
        elif r["target"].lower() in {c.lower() for c in concept_names}:
            r["target_type"] = "concept"

    return {
        "people": list(people.values()),
        "companies": list(companies.values()),
        "concepts": list(concepts.values()),
        "books": list(books.values()),
        "metrics": metrics,
        "relationships": rel_list,
    }


# ── Processamento por episodio ─────────────────────────────

def process_episode(episode_data: dict, client: AzureOpenAI, force: bool = False) -> dict:
    """Processa um episodio: limpa, chunka, extrai entidades."""
    ep_id = str(episode_data["id"])
    title = episode_data["title"]
    output_file = config.KG_ENTITIES_DIR / f"{ep_id}.json"

    # Skip se ja processado
    if output_file.exists() and not force:
        return {"id": ep_id, "status": "skipped"}

    text = episode_data.get("text", "")
    if not text or len(text) < 100:
        return {"id": ep_id, "status": "error", "error": "Texto muito curto"}

    try:
        # Limpa texto
        cleaned = clean_text(text)

        # Chunka
        chunks = chunk_for_extraction(
            cleaned,
            chunk_size=config.KG_CHUNK_SIZE,
            overlap=config.KG_CHUNK_OVERLAP,
        )

        # Extrai entidades de cada chunk
        chunk_results = []
        for chunk in chunks:
            result = extract_from_chunk(client, title, chunk)
            chunk_results.append(result)

        # Merge
        merged = merge_chunk_results(chunk_results)

        # Monta output
        output = {
            "episode_id": ep_id,
            "title": title,
            "date_published": episode_data.get("date_published", 0),
            "duration": episode_data.get("duration", 0),
            "chunks_processed": len(chunks),
            **merged,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        n_entities = (
            len(merged["people"])
            + len(merged["companies"])
            + len(merged["concepts"])
        )

        return {
            "id": ep_id,
            "status": "ok",
            "entities": n_entities,
            "relationships": len(merged["relationships"]),
            "chunks": len(chunks),
        }

    except Exception as e:
        return {"id": ep_id, "status": "error", "error": str(e)[:200]}


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extrair entidades dos episodios")
    parser.add_argument("--max", type=int, default=0, help="Maximo de episodios (0=todos)")
    parser.add_argument("--force", action="store_true", help="Reprocessa ja extraidos")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 5 -- Extracao de Entidades (GPT-4o)")
    print("=" * 60)

    # Carrega transcricoes
    transcript_files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        print("ERRO: Nenhuma transcricao encontrada.")
        sys.exit(1)

    # Carrega dados
    episodes = []
    for tf in transcript_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("text") and len(data["text"]) > 100:
                episodes.append(data)
        except Exception:
            continue

    if args.max > 0:
        episodes = episodes[:args.max]

    # Conta ja processados
    already = sum(
        1 for e in episodes
        if (config.KG_ENTITIES_DIR / f"{e['id']}.json").exists()
    )

    print(f"  {len(episodes)} episodios para processar")
    print(f"  {already} ja extraidos (serao pulados)")
    if not args.force:
        remaining = len(episodes) - already
        print(f"  {remaining} a processar")
        # Estimativa de custo
        avg_chunks = 3
        est_calls = remaining * avg_chunks
        est_cost = est_calls * 0.005
        print(f"  Estimativa: ~{est_calls} chamadas API (~${est_cost:.2f})")
    print()

    client = get_client()
    results = {"ok": 0, "skipped": 0, "error": 0}
    errors = []

    for i, ep in enumerate(tqdm(episodes, desc="Extraindo entidades", unit="ep")):
        result = process_episode(ep, client, force=args.force)
        results[result["status"]] += 1
        if result["status"] == "error":
            errors.append(result)

    print(f"\nExtracao concluida:")
    print(f"   Novos:   {results['ok']}")
    print(f"   Pulados: {results['skipped']}")
    print(f"   Erros:   {results['error']}")

    if errors:
        print(f"\n-- Erros ({len(errors)}) --")
        for e in errors[:10]:
            print(f"  ID {e['id']}: {e.get('error', '?')[:120]}")

    # Stats
    total_people = 0
    total_companies = 0
    total_concepts = 0
    total_rels = 0
    for ef in config.KG_ENTITIES_DIR.glob("*.json"):
        try:
            with open(ef, "r", encoding="utf-8") as f:
                d = json.load(f)
            total_people += len(d.get("people", []))
            total_companies += len(d.get("companies", []))
            total_concepts += len(d.get("concepts", []))
            total_rels += len(d.get("relationships", []))
        except Exception:
            pass

    print(f"\nTotais (antes de deduplicacao global):")
    print(f"  Pessoas:         {total_people:,}")
    print(f"  Empresas:        {total_companies:,}")
    print(f"  Conceitos:       {total_concepts:,}")
    print(f"  Relacionamentos: {total_rels:,}")


if __name__ == "__main__":
    main()
