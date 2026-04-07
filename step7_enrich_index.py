"""
Step 7 -- Enriquecer ChromaDB com metadados de entidades do Knowledge Graph.
Adiciona pessoas, empresas e conceitos como metadata dos chunks, melhorando
a busca por entidade e tema.

Uso: python step7_enrich_index.py
"""

import json
import sys
from pathlib import Path

import chromadb
from tqdm import tqdm

import config


def load_entity_index() -> dict:
    """
    Carrega entidades por episodio a partir dos arquivos individuais.
    Retorna dict: episode_id -> {people: [...], companies: [...], concepts: [...]}
    """
    entity_map = {}
    entity_files = sorted(config.KG_ENTITIES_DIR.glob("*.json"))

    if not entity_files:
        print("AVISO: Nenhum arquivo de entidades encontrado. Rode step5 primeiro.")
        return {}

    for ef in entity_files:
        try:
            with open(ef, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        ep_id = str(data.get("episode_id", ef.stem))

        people_names = sorted(set(
            p["name"] for p in data.get("people", []) if p.get("name")
        ))
        company_names = sorted(set(
            c["name"] for c in data.get("companies", []) if c.get("name")
        ))
        concept_names = sorted(set(
            c["name"] for c in data.get("concepts", []) if c.get("name")
        ))
        book_titles = sorted(set(
            b["title"] for b in data.get("books", []) if b.get("title")
        ))

        entity_map[ep_id] = {
            "people": people_names,
            "companies": company_names,
            "concepts": concept_names,
            "books": book_titles,
        }

    return entity_map


def enrich_index():
    """Atualiza metadata dos chunks no ChromaDB com entidades do KG."""
    # Carrega entidades
    entity_map = load_entity_index()
    if not entity_map:
        print("Nenhuma entidade para enriquecer. Rode step5 + step6 primeiro.")
        sys.exit(1)

    print(f"Entidades carregadas para {len(entity_map)} episodios")

    # Abre ChromaDB
    client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
    collection = client.get_or_create_collection(
        name="g4_podcast",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = collection.count()
    print(f"Chunks no indice: {total_chunks}")

    # Processa em batches (ChromaDB get limit)
    BATCH_SIZE = 500
    updated = 0
    skipped = 0

    for offset in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Enriquecendo"):
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )

        ids_to_update = []
        metas_to_update = []

        for chunk_id, meta in zip(batch["ids"], batch["metadatas"]):
            ep_id = str(meta.get("episode_id", ""))

            if ep_id not in entity_map:
                skipped += 1
                continue

            entities = entity_map[ep_id]

            # Junta entidades como strings separadas por "; "
            new_meta = {**meta}
            new_meta["kg_people"] = "; ".join(entities["people"][:20])  # top 20
            new_meta["kg_companies"] = "; ".join(entities["companies"][:20])
            new_meta["kg_concepts"] = "; ".join(entities["concepts"][:20])
            new_meta["kg_books"] = "; ".join(entities["books"][:10])
            new_meta["kg_enriched"] = True

            ids_to_update.append(chunk_id)
            metas_to_update.append(new_meta)

        if ids_to_update:
            collection.update(
                ids=ids_to_update,
                metadatas=metas_to_update,
            )
            updated += len(ids_to_update)

    print(f"\nEnriquecimento concluido:")
    print(f"  Chunks atualizados: {updated:,}")
    print(f"  Chunks sem entidades: {skipped:,}")
    print(f"  Total: {updated + skipped:,}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7 -- Enriquecer ChromaDB com entidades do KG")
    print("=" * 60)
    enrich_index()
