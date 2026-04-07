"""
Step 6 -- Construir Knowledge Graph consolidado a partir das entidades extraidas.
Deduplica entidades, constroi grafo de relacionamentos, exporta JSON + GraphML.

Uso: python step6_build_knowledge_graph.py
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import config

# ── Aliases conhecidos ─────────────────────────────────────
PERSON_ALIASES = {
    "tallis": "Tallis Gomes",
    "tg": "Tallis Gomes",
    "alfredo": "Alfredo Soares",
    "nardon": "Bruno Nardon",
    "bruno nardon": "Bruno Nardon",
    "tony": "Tony Celestino",
}

COMPANY_ALIASES = {
    "g4": "G4 Educacao",
    "g4 educacao": "G4 Educacao",
    "g4 educação": "G4 Educacao",
    "gestao 4.0": "G4 Educacao",
    "gestão 4.0": "G4 Educacao",
    "xp": "XP Inc",
    "xp investimentos": "XP Inc",
}


def normalize(name: str) -> str:
    """Normaliza string: strip, title case, espaco unico."""
    return re.sub(r"\s+", " ", name.strip().title())


def remove_accents(s: str) -> str:
    """Remove acentos para matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def resolve_person(name: str) -> str:
    """Resolve alias de pessoa para nome canonico."""
    key = name.strip().lower()
    if key in PERSON_ALIASES:
        return PERSON_ALIASES[key]
    # Checa se algum alias eh substring
    for alias, canonical in PERSON_ALIASES.items():
        if alias in key:
            return canonical
    return normalize(name)


def resolve_company(name: str) -> str:
    """Resolve alias de empresa para nome canonico."""
    key = remove_accents(name.strip().lower())
    if key in COMPANY_ALIASES:
        return COMPANY_ALIASES[key]
    for alias, canonical in COMPANY_ALIASES.items():
        if alias in key:
            return canonical
    return normalize(name)


def build_graph():
    """Constroi knowledge graph consolidado."""
    entity_files = sorted(config.KG_ENTITIES_DIR.glob("*.json"))
    if not entity_files:
        print("ERRO: Nenhum arquivo de entidades encontrado. Rode step5 primeiro.")
        sys.exit(1)

    print(f"Processando {len(entity_files)} arquivos de entidades...")

    # Registros globais
    people = {}        # name -> {roles, company, title, episodes, episode_count}
    companies = {}     # name -> {industry, context, episodes, episode_count}
    concepts = {}      # name -> {category, episodes, episode_count}
    books = {}         # title -> {author, episodes, episode_count}
    metrics_all = []   # lista global
    relationships = defaultdict(lambda: {"episodes": set(), "count": 0})

    for ef in entity_files:
        try:
            with open(ef, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        ep_id = str(data.get("episode_id", ef.stem))

        # People
        for p in data.get("people", []):
            name = resolve_person(p["name"])
            if len(name) < 2:
                continue
            if name not in people:
                people[name] = {
                    "name": name,
                    "roles": set(),
                    "company": "",
                    "title": "",
                    "episodes": set(),
                }
            people[name]["roles"].add(p.get("role", "mentioned"))
            if p.get("company"):
                people[name]["company"] = resolve_company(p["company"])
            if p.get("title"):
                people[name]["title"] = p["title"]
            people[name]["episodes"].add(ep_id)

        # Companies
        for c in data.get("companies", []):
            name = resolve_company(c["name"])
            if len(name) < 2:
                continue
            if name not in companies:
                companies[name] = {
                    "name": name,
                    "industry": "",
                    "context": "",
                    "episodes": set(),
                }
            if c.get("industry"):
                companies[name]["industry"] = c["industry"]
            if c.get("context"):
                companies[name]["context"] = c["context"]
            companies[name]["episodes"].add(ep_id)

        # Concepts
        for c in data.get("concepts", []):
            key = c["name"].strip().lower()
            if len(key) < 2:
                continue
            if key not in concepts:
                concepts[key] = {
                    "name": c["name"].strip(),
                    "category": c.get("category", ""),
                    "episodes": set(),
                }
            if c.get("category"):
                concepts[key]["category"] = c["category"]
            concepts[key]["episodes"].add(ep_id)

        # Books
        for b in data.get("books", []):
            key = b["title"].strip().lower()
            if len(key) < 2:
                continue
            if key not in books:
                books[key] = {
                    "name": b["title"].strip(),
                    "author": b.get("author", ""),
                    "episodes": set(),
                }
            if b.get("author"):
                books[key]["author"] = b["author"]
            books[key]["episodes"].add(ep_id)

        # Metrics
        for m in data.get("metrics", []):
            metrics_all.append({**m, "episode_id": ep_id})

        # Relationships
        for r in data.get("relationships", []):
            source = normalize(r["source"])
            target = normalize(r["target"])
            rel_key = (source, r["relation"], target)
            relationships[rel_key]["source_type"] = r.get("source_type", "person")
            relationships[rel_key]["target_type"] = r.get("target_type", "company")
            relationships[rel_key]["episodes"].add(ep_id)
            relationships[rel_key]["count"] += 1

    # Serializa sets para listas
    def finalize(entity_dict):
        result = {}
        for key, val in entity_dict.items():
            entry = {**val}
            if "episodes" in entry:
                entry["episode_count"] = len(entry["episodes"])
                entry["episodes"] = sorted(entry["episodes"])
            if "roles" in entry:
                entry["roles"] = sorted(entry["roles"])
            result[key] = entry
        return result

    people_final = finalize(people)
    companies_final = finalize(companies)
    concepts_final = finalize(concepts)
    books_final = finalize(books)

    rel_list = []
    for (source, relation, target), info in relationships.items():
        rel_list.append({
            "source": source,
            "source_type": info.get("source_type", "person"),
            "relation": relation,
            "target": target,
            "target_type": info.get("target_type", "company"),
            "episode_count": len(info["episodes"]),
            "episodes": sorted(info["episodes"]),
        })

    # Ordena por frequencia
    rel_list.sort(key=lambda r: r["episode_count"], reverse=True)

    # Monta graph.json
    graph = {
        "entities": {
            "people": dict(sorted(people_final.items(), key=lambda x: x[1].get("episode_count", 0), reverse=True)),
            "companies": dict(sorted(companies_final.items(), key=lambda x: x[1].get("episode_count", 0), reverse=True)),
            "concepts": dict(sorted(concepts_final.items(), key=lambda x: x[1].get("episode_count", 0), reverse=True)),
            "books": dict(sorted(books_final.items(), key=lambda x: x[1].get("episode_count", 0), reverse=True)),
        },
        "relationships": rel_list,
        "metrics_sample": metrics_all[:500],  # amostra de metricas
        "stats": {
            "total_people": len(people_final),
            "total_companies": len(companies_final),
            "total_concepts": len(concepts_final),
            "total_books": len(books_final),
            "total_relationships": len(rel_list),
            "total_metrics": len(metrics_all),
            "episodes_processed": len(entity_files),
        },
    }

    # Salva graph.json
    with open(config.KG_GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"\nGrafo salvo em: {config.KG_GRAPH_FILE}")

    # Exporta NetworkX GraphML
    try:
        import networkx as nx

        G = nx.DiGraph()

        # Nodes
        for name, p in people_final.items():
            G.add_node(name, type="person", episode_count=p.get("episode_count", 0),
                       roles=",".join(p.get("roles", [])), company=p.get("company", ""))
        for name, c in companies_final.items():
            G.add_node(name, type="company", episode_count=c.get("episode_count", 0),
                       industry=c.get("industry", ""))
        for key, c in concepts_final.items():
            G.add_node(c["name"], type="concept", episode_count=c.get("episode_count", 0),
                       category=c.get("category", ""))
        for key, b in books_final.items():
            G.add_node(b["name"], type="book", episode_count=b.get("episode_count", 0),
                       author=b.get("author", ""))

        # Edges
        for r in rel_list:
            if G.has_node(r["source"]) and G.has_node(r["target"]):
                G.add_edge(r["source"], r["target"],
                           relation=r["relation"], weight=r["episode_count"])

        nx.write_graphml(G, str(config.KG_NETWORKX_FILE))
        print(f"GraphML salvo em: {config.KG_NETWORKX_FILE}")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    except ImportError:
        print("NetworkX nao instalado, pulando export GraphML")

    # Stats
    print(f"\n{'=' * 50}")
    print(f"Knowledge Graph - Estatisticas")
    print(f"{'=' * 50}")
    print(f"  Pessoas:         {graph['stats']['total_people']:,}")
    print(f"  Empresas:        {graph['stats']['total_companies']:,}")
    print(f"  Conceitos:       {graph['stats']['total_concepts']:,}")
    print(f"  Livros:          {graph['stats']['total_books']:,}")
    print(f"  Relacionamentos: {graph['stats']['total_relationships']:,}")
    print(f"  Metricas:        {graph['stats']['total_metrics']:,}")
    print(f"  Episodios:       {graph['stats']['episodes_processed']}")

    # Top entities
    print(f"\nTop 10 Pessoas:")
    for name, p in list(people_final.items())[:10]:
        print(f"  {name} ({p.get('episode_count',0)} eps) - {','.join(p.get('roles',[]))}")

    print(f"\nTop 10 Empresas:")
    for name, c in list(companies_final.items())[:10]:
        print(f"  {name} ({c.get('episode_count',0)} eps)")

    print(f"\nTop 10 Conceitos:")
    for key, c in list(concepts_final.items())[:10]:
        print(f"  {c['name']} ({c.get('episode_count',0)} eps)")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6 -- Construir Knowledge Graph")
    print("=" * 60)
    build_graph()
