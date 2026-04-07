"""
Step 4 — Criar índice vetorial (embeddings + ChromaDB).
Chunka transcrições, gera embeddings via Azure OpenAI, armazena em ChromaDB.

Uso: python step4_build_index.py [--rebuild]
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
import tiktoken
from openai import AzureOpenAI
from tqdm import tqdm

import config


def get_embedding_client() -> AzureOpenAI:
    """Cliente Azure OpenAI para embeddings."""
    return AzureOpenAI(
        api_key=config.AZURE_API_KEY,
        api_version=config.AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=config.AZURE_ENDPOINT,
    )


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide texto em chunks de ~chunk_size tokens com overlap.
    Respeita limites de parágrafo quando possível.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Tenta cortar no final de uma frase
        if end < len(tokens):
            last_period = chunk_text.rfind(". ")
            last_newline = chunk_text.rfind("\n")
            cut_point = max(last_period, last_newline)
            if cut_point > len(chunk_text) * 0.5:  # Só corta se não perder muito
                chunk_text = chunk_text[: cut_point + 1]
                # Recalcula end baseado no texto cortado
                end = start + len(enc.encode(chunk_text))

        chunks.append(chunk_text.strip())
        if end >= len(tokens):
            break
        start = end - overlap  # Overlap

    return [c for c in chunks if len(c) > 20]


def batch_embed(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Gera embeddings em batch."""
    # Limpa textos: remove vazios e trunca muito longos (max ~8000 tokens ~ 30000 chars)
    cleaned = [t[:30000] if len(t) > 30000 else t for t in texts]
    cleaned = [t if t.strip() else "vazio" for t in cleaned]
    response = client.embeddings.create(
        model=config.AZURE_EMBEDDING_MODEL,
        input=cleaned,
    )
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(description="Construir índice vetorial")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir do zero")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 4 — Construir índice vetorial (ChromaDB)")
    print("=" * 60)

    # Carregar transcrições
    transcript_files = list(config.TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        print("❌ Nenhuma transcrição encontrada. Rode step3 primeiro.")
        sys.exit(1)

    print(f"📄 {len(transcript_files)} transcrições encontradas")

    # Inicializar ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))

    collection_name = "g4_podcast"
    if args.rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print("🗑️  Coleção anterior deletada")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    print(f"📊 Chunks existentes no índice: {existing_count}")

    # Processar cada transcrição
    embedding_client = get_embedding_client()
    total_chunks = 0
    total_new = 0

    for tf in tqdm(transcript_files, desc="Indexando", unit="ep"):
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)

        ep_id = str(data["id"])

        # Verifica se já indexado (checa se algum doc com esse ep_id existe)
        if not args.rebuild:
            existing = collection.get(where={"episode_id": ep_id}, limit=1)
            if existing and existing["ids"]:
                continue

        # Usa texto diarizado se disponível, senão texto plano
        has_diarization = bool(data.get("segments"))
        source_text = data.get("text_diarized") or data["text"]

        # Chunkar texto
        chunks = chunk_text(
            source_text,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        total_chunks += len(chunks)

        if not chunks:
            continue

        # Gerar embeddings em batches
        all_embeddings = []
        for i in range(0, len(chunks), config.EMBEDDING_BATCH):
            batch = chunks[i : i + config.EMBEDDING_BATCH]
            embeddings = batch_embed(embedding_client, batch)
            all_embeddings.extend(embeddings)

        # Monta info de speakers se houver diarização
        speaker_labels = data.get("speaker_labels", {})
        speakers_str = ", ".join(speaker_labels.values()) if speaker_labels else ""

        # Preparar dados para ChromaDB
        ids = [f"{ep_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "episode_id": ep_id,
                "title": data["title"],
                "guest": data.get("guest", ""),
                "date_published": data.get("date_published", 0),
                "duration": data.get("duration", 0),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "has_diarization": has_diarization,
                "speakers": speakers_str,
            }
            for i in range(len(chunks))
        ]

        # Inserir no ChromaDB
        collection.add(
            ids=ids,
            embeddings=all_embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_new += len(chunks)

    final_count = collection.count()
    print(f"\n✅ Indexação concluída:")
    print(f"   📊 Chunks totais no índice: {final_count}")
    print(f"   🆕 Novos chunks adicionados: {total_new}")
    print(f"   📦 Episódios processados: {len(transcript_files)}")
    print(f"\n💾 ChromaDB salvo em: {config.CHROMADB_DIR}")


if __name__ == "__main__":
    main()
