"""
MCP Server — Agente G4 Podcast RAG.
Expõe ferramentas de busca semântica nos episódios do G4 Podcast
para uso no Claude Desktop.

Uso: python mcp_server.py
Ou configure no Claude Desktop (claude_desktop_config.json)
"""

import json
from pathlib import Path
from typing import Any

import chromadb
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openai import AzureOpenAI

import config

# ── Inicialização ───────────────────────────────────────────

app = Server("g4-podcast-rag")

# ChromaDB
chroma_client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
collection = chroma_client.get_or_create_collection(
    name="g4_podcast",
    metadata={"hnsw:space": "cosine"},
)

# Azure OpenAI (para embeddings na busca)
embedding_client = AzureOpenAI(
    api_key=config.AZURE_API_KEY,
    api_version=config.AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=config.AZURE_ENDPOINT,
)

# Azure OpenAI (chat)
chat_client = AzureOpenAI(
    api_key=config.AZURE_API_KEY,
    api_version=config.AZURE_CHAT_API_VERSION,
    azure_endpoint=config.AZURE_ENDPOINT,
)

# Knowledge Graph (carrega se existir)
_kg_graph = None

def _load_kg():
    global _kg_graph
    if _kg_graph is not None:
        return _kg_graph
    if config.KG_GRAPH_FILE.exists():
        with open(config.KG_GRAPH_FILE, "r", encoding="utf-8") as f:
            _kg_graph = json.load(f)
    else:
        _kg_graph = {"entities": {"people": {}, "companies": {}, "concepts": {}, "books": {}}, "relationships": []}
    return _kg_graph


def get_query_embedding(text: str) -> list[float]:
    """Gera embedding para a query de busca."""
    response = embedding_client.embeddings.create(
        model=config.AZURE_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def format_results(results: dict) -> str:
    """Formata resultados do ChromaDB em texto legível."""
    if not results or not results.get("documents") or not results["documents"][0]:
        return "Nenhum resultado encontrado."

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else [0] * len(docs)

    output_parts = []
    seen_episodes = {}

    for doc, meta, dist in zip(docs, metas, distances):
        ep_id = meta.get("episode_id", "?")
        title = meta.get("title", "Sem título")
        guest = meta.get("guest", "")
        relevance = round((1 - dist) * 100, 1)

        speakers = meta.get("speakers", "")

        # Agrupa por episódio
        if ep_id not in seen_episodes:
            seen_episodes[ep_id] = {
                "title": title,
                "guest": guest,
                "speakers": speakers,
                "chunks": [],
                "max_relevance": relevance,
            }

        seen_episodes[ep_id]["chunks"].append(doc)
        seen_episodes[ep_id]["max_relevance"] = max(
            seen_episodes[ep_id]["max_relevance"], relevance
        )

    # Formata saída agrupada por episódio
    for ep_id, data in sorted(
        seen_episodes.items(), key=lambda x: x[1]["max_relevance"], reverse=True
    ):
        guest_str = f" (Convidado: {data['guest']})" if data["guest"] else ""
        speakers_str = ""
        if data.get("speakers"):
            speakers_str = f"\n**Participantes:** {data['speakers']}"
        output_parts.append(
            f"### 🎙️ {data['title']}{guest_str}\n"
            f"**Relevância:** {data['max_relevance']}%{speakers_str}\n"
        )
        for i, chunk in enumerate(data["chunks"], 1):
            output_parts.append(f"**Trecho {i}:**\n{chunk}\n")
        output_parts.append("---\n")

    return "\n".join(output_parts)

def firecrawl_extract(url: str) -> str:
    if not config.FIRECRAWL_API_KEY:
        return ""
    base = config.FIRECRAWL_BASE_URL.rstrip("/")
    endpoint = f"{base}/extract"
    headers = {"Authorization": f"Bearer {config.FIRECRAWL_API_KEY}"}
    payload = {"url": url, "formats": ["markdown", "content"], "autoparse": True}
    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "markdown" in data:
                return data["markdown"] or ""
            if "content" in data:
                return data["content"] or ""
            if "data" in data and isinstance(data["data"], dict):
                md = data["data"].get("markdown") or data["data"].get("content") or ""
                return md or ""
        return ""
    except Exception:
        return ""

def summarize_company(website: str, raw_text: str, extra: str = "") -> dict:
    content = raw_text[:100000]
    extra_info = extra[:4000] if extra else ""
    system = (
        "Você é um consultor de negócios. Extraia um perfil objetivo da empresa com base no site."
    )
    user = (
        f"Website: {website}\n"
        f"Conteúdo:\n{content}\n\n"
        f"Info adicionais: {extra_info}\n\n"
        "Responda em JSON com as chaves: "
        "nome, setor, produtos_servicos, publico_alvo, modelo_de_negocio, proposta_de_valor, canais_de_aquisicao, regioes, estagio, diferenciais."
    )
    try:
        resp = chat_client.chat.completions.create(
            model=config.AZURE_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

def generate_consulting_plan(perfil: dict, desafios: str, context_text: str) -> str:
    perfil_json = json.dumps(perfil, ensure_ascii=False)
    system = "Você é um consultor sênior do G4 Educação. Gere recomendações práticas, passo a passo."
    user = (
        f"Perfil da empresa (JSON): {perfil_json}\n\n"
        f"Desafios: {desafios}\n\n"
        f"Contexto do G4 Podcast (trechos):\n{context_text}\n\n"
        "Entregue:\n"
        "- Diagnóstico\n"
        "- 5-8 ações priorizadas (com 30-60 dias)\n"
        "- Riscos e como mitigar\n"
        "- Indicadores (KPIs)\n"
        "- Referências dos episódios citados quando possível\n"
    )
    try:
        resp = chat_client.chat.completions.create(
            model=config.AZURE_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Não foi possível gerar o plano. Erro: {e}"


# ── Tools ───────────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="buscar_conhecimento",
            description=(
                "Busca semântica na base de conhecimento do G4 Podcast. "
                "Use para encontrar insights, estratégias e conselhos de "
                "empresários e especialistas que participaram do podcast. "
                "Retorna trechos relevantes com contexto do episódio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pergunta": {
                        "type": "string",
                        "description": (
                            "A pergunta ou tema para buscar. "
                            "Ex: 'como escalar vendas B2B', "
                            "'gestão de equipes remotas', "
                            "'estratégias de pricing'"
                        ),
                    },
                    "max_resultados": {
                        "type": "integer",
                        "description": "Número máximo de trechos (padrão: 8)",
                        "default": 8,
                    },
                },
                "required": ["pergunta"],
            },
        ),
        Tool(
            name="buscar_por_convidado",
            description=(
                "Busca episódios e trechos de um convidado específico do G4 Podcast. "
                "Use quando o usuário perguntar sobre uma pessoa específica."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nome_convidado": {
                        "type": "string",
                        "description": "Nome do convidado. Ex: 'Bruno Nardon', 'Tallis Gomes'",
                    },
                    "tema": {
                        "type": "string",
                        "description": "Tema opcional para filtrar. Ex: 'liderança'",
                        "default": "",
                    },
                },
                "required": ["nome_convidado"],
            },
        ),
        Tool(
            name="listar_episodios",
            description=(
                "Lista todos os episódios disponíveis do G4 Podcast com título, "
                "convidado e data. Use para dar visão geral do conteúdo disponível."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limite": {
                        "type": "integer",
                        "description": "Número máximo de episódios (padrão: 20)",
                        "default": 20,
                    },
                    "busca_titulo": {
                        "type": "string",
                        "description": "Filtrar por texto no título (opcional)",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="contexto_empresarial",
            description=(
                "Busca conhecimento do G4 Podcast aplicado ao contexto específico "
                "do empresário. Informe o segmento, desafio e porte da empresa "
                "para receber insights personalizados."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segmento": {
                        "type": "string",
                        "description": "Segmento da empresa. Ex: 'SaaS B2B', 'varejo', 'educação'",
                    },
                    "desafio": {
                        "type": "string",
                        "description": "Principal desafio atual. Ex: 'escalar vendas', 'reter talentos'",
                    },
                    "faturamento": {
                        "type": "string",
                        "description": "Faixa de faturamento. Ex: 'R$1-5M/ano', 'R$50M+'",
                        "default": "",
                    },
                    "num_funcionarios": {
                        "type": "string",
                        "description": "Número de funcionários. Ex: '10-50', '200+'",
                        "default": "",
                    },
                },
                "required": ["segmento", "desafio"],
            },
        ),
        Tool(
            name="buscar_entidade",
            description=(
                "Busca informacoes sobre uma entidade no Knowledge Graph do G4 Podcast. "
                "Retorna dados sobre pessoas, empresas, conceitos ou livros mencionados "
                "nos episodios, incluindo frequencia e episodios relacionados."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome da entidade. Ex: 'Tallis Gomes', 'G4 Educacao', 'OKR'",
                    },
                    "tipo": {
                        "type": "string",
                        "description": "Tipo: 'person', 'company', 'concept', 'book' ou 'all' (padrao: all)",
                        "default": "all",
                    },
                },
                "required": ["nome"],
            },
        ),
        Tool(
            name="explorar_conexoes",
            description=(
                "Explora conexoes e relacionamentos de uma entidade no Knowledge Graph. "
                "Mostra com quem/o que a entidade se conecta, tipo de relacao e frequencia."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": "Nome da entidade para explorar conexoes. Ex: 'Tallis Gomes'",
                    },
                    "max_conexoes": {
                        "type": "integer",
                        "description": "Maximo de conexoes a retornar (padrao: 20)",
                        "default": 20,
                    },
                },
                "required": ["nome"],
            },
        ),
        Tool(
            name="buscar_por_tema",
            description=(
                "Busca conceitos, livros e entidades relacionadas a um tema no Knowledge Graph. "
                "Use para explorar um assunto e descobrir o que o G4 Podcast ja abordou sobre ele."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tema": {
                        "type": "string",
                        "description": "Tema para buscar. Ex: 'lideranca', 'growth', 'cultura organizacional'",
                    },
                    "max_resultados": {
                        "type": "integer",
                        "description": "Maximo de resultados (padrao: 15)",
                        "default": 15,
                    },
                },
                "required": ["tema"],
            },
        ),
        Tool(
            name="consultor_negocios",
            description=(
                "Assistente consultivo: enriquece informações do site da empresa com Firecrawl, "
                "busca aprendizados no G4 Podcast e gera um plano de ação personalizado."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "website": {
                        "type": "string",
                        "description": "URL do site da empresa. Ex: 'https://minhaempresa.com'",
                    },
                    "desafios": {
                        "type": "string",
                        "description": "Descrição dos desafios atuais do negócio.",
                    },
                    "info_adicionais": {
                        "type": "string",
                        "description": "Informações extras sobre a empresa (opcional).",
                        "default": "",
                    },
                },
                "required": ["website", "desafios"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    if name == "buscar_conhecimento":
        pergunta = arguments["pergunta"]
        max_results = arguments.get("max_resultados", config.MAX_RESULTS)

        embedding = get_query_embedding(pergunta)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=max_results,
        )

        text = format_results(results)
        return [TextContent(
            type="text",
            text=f"## Resultados para: \"{pergunta}\"\n\n{text}",
        )]

    elif name == "buscar_por_convidado":
        nome = arguments["nome_convidado"]
        tema = arguments.get("tema", "")

        # Busca por metadado (guest) + semântica se tema fornecido
        query_text = f"{nome} {tema}".strip()
        embedding = get_query_embedding(query_text)

        # Primeiro tenta filtrar por convidado no metadado
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=10,
                where={"guest": {"$contains": nome}},
            )
        except Exception:
            # Fallback: busca semântica pura
            results = collection.query(
                query_embeddings=[embedding],
                n_results=10,
            )

        text = format_results(results)
        return [TextContent(
            type="text",
            text=f"## Episódios com {nome}\n\n{text}",
        )]

    elif name == "listar_episodios":
        limite = arguments.get("limite", 20)
        busca = arguments.get("busca_titulo", "").lower()

        # Carrega episodes.json
        if config.EPISODES_FILE.exists():
            with open(config.EPISODES_FILE, "r", encoding="utf-8") as f:
                episodes = json.load(f)

            if busca:
                episodes = [e for e in episodes if busca in e.get("title", "").lower()]

            episodes = episodes[-limite:]  # Mais recentes

            lines = []
            for ep in reversed(episodes):
                from datetime import datetime, timezone

                date_str = ""
                if ep.get("date_published"):
                    try:
                        dt = datetime.fromtimestamp(ep["date_published"], tz=timezone.utc)
                        date_str = dt.strftime("%d/%m/%Y")
                    except Exception:
                        date_str = str(ep["date_published"])

                guest = f" — {ep['guest']}" if ep.get("guest") else ""
                dur_min = ep.get("duration", 0) // 60
                lines.append(f"- **{ep['title']}**{guest} ({date_str}, {dur_min}min)")

            text = "\n".join(lines) if lines else "Nenhum episódio encontrado."
            return [TextContent(
                type="text",
                text=f"## Episódios do G4 Podcast ({len(lines)} listados)\n\n{text}",
            )]
        else:
            return [TextContent(type="text", text="❌ episodes.json não encontrado.")]

    elif name == "contexto_empresarial":
        segmento = arguments["segmento"]
        desafio = arguments["desafio"]
        faturamento = arguments.get("faturamento", "")
        funcionarios = arguments.get("num_funcionarios", "")

        # Constrói query contextualizada
        query = (
            f"Empresa do segmento {segmento}, "
            f"enfrentando o desafio de {desafio}. "
        )
        if faturamento:
            query += f"Faturamento: {faturamento}. "
        if funcionarios:
            query += f"Equipe: {funcionarios} funcionários. "
        query += "Quais estratégias, conselhos e experiências são relevantes?"

        embedding = get_query_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=12,  # Mais resultados para contexto
        )

        text = format_results(results)
        context = f"**Segmento:** {segmento}\n**Desafio:** {desafio}"
        if faturamento:
            context += f"\n**Faturamento:** {faturamento}"
        if funcionarios:
            context += f"\n**Equipe:** {funcionarios}"

        return [TextContent(
            type="text",
            text=(
                f"## Conhecimento G4 para seu contexto\n\n"
                f"{context}\n\n---\n\n{text}\n\n"
                f"💡 *Use esses trechos como base para gerar conselhos "
                f"personalizados ao contexto do empresário.*"
            ),
        )]

    elif name == "buscar_entidade":
        nome = arguments["nome"].strip()
        tipo = arguments.get("tipo", "all").lower()
        nome_lower = nome.lower()

        kg = _load_kg()
        results = []

        type_map = {
            "person": ("people", "Pessoa"),
            "company": ("companies", "Empresa"),
            "concept": ("concepts", "Conceito"),
            "book": ("books", "Livro"),
        }
        search_types = type_map if tipo == "all" else {tipo: type_map[tipo]} if tipo in type_map else type_map

        for etype, (dict_key, label) in search_types.items():
            entities = kg.get("entities", {}).get(dict_key, {})
            for key, entity in entities.items():
                entity_name = entity.get("name", key)
                if nome_lower in entity_name.lower() or nome_lower in key.lower():
                    ep_count = entity.get("episode_count", 0)
                    info = f"**[{label}] {entity_name}** ({ep_count} episodios)\n"
                    if entity.get("roles"):
                        info += f"  Papeis: {', '.join(entity['roles'])}\n"
                    if entity.get("company"):
                        info += f"  Empresa: {entity['company']}\n"
                    if entity.get("industry"):
                        info += f"  Setor: {entity['industry']}\n"
                    if entity.get("category"):
                        info += f"  Categoria: {entity['category']}\n"
                    if entity.get("author"):
                        info += f"  Autor: {entity['author']}\n"
                    if entity.get("context"):
                        info += f"  Contexto: {entity['context']}\n"
                    eps = entity.get("episodes", [])
                    if eps:
                        info += f"  Episodios: {', '.join(eps[:10])}"
                        if len(eps) > 10:
                            info += f" (+{len(eps)-10} mais)"
                        info += "\n"
                    results.append((ep_count, info))

        results.sort(key=lambda x: x[0], reverse=True)
        text = "\n".join(r[1] for r in results[:20]) if results else f"Nenhuma entidade encontrada para '{nome}'."
        return [TextContent(type="text", text=f"## Entidade: {nome}\n\n{text}")]

    elif name == "explorar_conexoes":
        nome = arguments["nome"].strip()
        max_conn = arguments.get("max_conexoes", 20)
        nome_lower = nome.lower()

        kg = _load_kg()
        connections = []

        for rel in kg.get("relationships", []):
            source = rel.get("source", "")
            target = rel.get("target", "")
            if nome_lower in source.lower() or nome_lower in target.lower():
                other = target if nome_lower in source.lower() else source
                connections.append({
                    "entity": other,
                    "relation": rel.get("relation", ""),
                    "direction": "saida" if nome_lower in source.lower() else "entrada",
                    "source_type": rel.get("source_type", ""),
                    "target_type": rel.get("target_type", ""),
                    "episode_count": rel.get("episode_count", 0),
                    "episodes": rel.get("episodes", []),
                })

        connections.sort(key=lambda x: x["episode_count"], reverse=True)
        connections = connections[:max_conn]

        if connections:
            lines = []
            for c in connections:
                arrow = "->" if c["direction"] == "saida" else "<-"
                lines.append(
                    f"- {arrow} **{c['entity']}** ({c['relation']}) "
                    f"[{c['episode_count']} eps]"
                )
            text = "\n".join(lines)
        else:
            text = f"Nenhuma conexao encontrada para '{nome}'."

        return [TextContent(type="text", text=f"## Conexoes de: {nome}\n\n{text}")]

    elif name == "buscar_por_tema":
        tema = arguments["tema"].strip()
        max_results = arguments.get("max_resultados", 15)
        tema_lower = tema.lower()

        kg = _load_kg()
        results = []

        # Busca em conceitos
        for key, entity in kg.get("entities", {}).get("concepts", {}).items():
            name_str = entity.get("name", key)
            category = entity.get("category", "")
            if tema_lower in name_str.lower() or tema_lower in key.lower() or tema_lower in category.lower():
                results.append((
                    entity.get("episode_count", 0),
                    f"**[Conceito] {name_str}** ({entity.get('episode_count',0)} eps)"
                    + (f" - Categoria: {category}" if category else "")
                ))

        # Busca em livros
        for key, entity in kg.get("entities", {}).get("books", {}).items():
            name_str = entity.get("name", key)
            if tema_lower in name_str.lower() or tema_lower in key.lower():
                author = entity.get("author", "")
                results.append((
                    entity.get("episode_count", 0),
                    f"**[Livro] {name_str}**"
                    + (f" por {author}" if author else "")
                    + f" ({entity.get('episode_count',0)} eps)"
                ))

        # Busca em pessoas pelo role/title
        for key, entity in kg.get("entities", {}).get("people", {}).items():
            roles = entity.get("roles", [])
            title = entity.get("title", "")
            if any(tema_lower in r.lower() for r in roles) or tema_lower in title.lower():
                results.append((
                    entity.get("episode_count", 0),
                    f"**[Pessoa] {entity.get('name', key)}** ({entity.get('episode_count',0)} eps)"
                    + (f" - {', '.join(roles)}" if roles else "")
                ))

        # Busca em empresas pelo industry/context
        for key, entity in kg.get("entities", {}).get("companies", {}).items():
            industry = entity.get("industry", "")
            context = entity.get("context", "")
            if tema_lower in industry.lower() or tema_lower in context.lower():
                results.append((
                    entity.get("episode_count", 0),
                    f"**[Empresa] {entity.get('name', key)}** ({entity.get('episode_count',0)} eps)"
                    + (f" - {industry}" if industry else "")
                ))

        # Busca relacionamentos por tema
        rel_matches = []
        for rel in kg.get("relationships", []):
            if tema_lower in rel.get("relation", "").lower():
                rel_matches.append((
                    rel.get("episode_count", 0),
                    f"  {rel['source']} --[{rel['relation']}]--> {rel['target']} ({rel.get('episode_count',0)} eps)"
                ))
        rel_matches.sort(key=lambda x: x[0], reverse=True)

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:max_results]

        parts = []
        if results:
            parts.append("### Entidades\n" + "\n".join(f"- {r[1]}" for r in results))
        if rel_matches:
            parts.append("### Relacionamentos\n" + "\n".join(r[1] for r in rel_matches[:10]))
        text = "\n\n".join(parts) if parts else f"Nenhum resultado para o tema '{tema}'."

        return [TextContent(type="text", text=f"## Tema: {tema}\n\n{text}")]

    elif name == "consultor_negocios":
        website = arguments["website"].strip()
        desafios = arguments["desafios"].strip()
        extra = arguments.get("info_adicionais", "").strip()

        site_text = firecrawl_extract(website) if website else ""
        perfil = summarize_company(website, site_text, extra) if site_text else {}

        perfil_str = ""
        if perfil:
            resumo = [
                f"Nome: {perfil.get('nome','')}",
                f"Setor: {perfil.get('setor','')}",
                f"Produtos/Serviços: {perfil.get('produtos_servicos','')}",
                f"Público-alvo: {perfil.get('publico_alvo','')}",
                f"Modelo: {perfil.get('modelo_de_negocio','')}",
            ]
            perfil_str = "\n".join([s for s in resumo if s and not s.endswith(": ")])

        query_parts = [desafios]
        if perfil.get("setor"):
            query_parts.append(perfil["setor"])
        if perfil.get("modelo_de_negocio"):
            query_parts.append(perfil["modelo_de_negocio"])
        if perfil.get("produtos_servicos"):
            query_parts.append(perfil["produtos_servicos"])
        query = " | ".join(query_parts)

        embedding = get_query_embedding(query)
        rag_results = collection.query(query_embeddings=[embedding], n_results=12)
        context_text = format_results(rag_results)

        plan = generate_consulting_plan(perfil, desafios, context_text)

        header = "## Consultoria baseada no G4 Podcast\n"
        if perfil_str:
            header += f"\n### Perfil da empresa\n{perfil_str}\n"
        header += f"\n### Desafios\n{desafios}\n"
        header += f"\n### Plano sugerido\n{plan}\n"
        header += f"\n---\n### Contexto utilizado\n{context_text}\n"

        return [TextContent(type="text", text=header)]

    return [TextContent(type="text", text=f"Ferramenta '{name}' nao encontrada.")]


# ── Main ────────────────────────────────────────────────────


async def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    print("G4 Podcast RAG - MCP Server iniciado", file=sys.stderr)
    print(f"Chunks no indice: {collection.count()}", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
