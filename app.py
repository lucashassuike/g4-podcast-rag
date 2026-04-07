"""
G4 Podcast RAG -- Web App (Streamlit)
Agente de Conhecimento do G4 Podcast para empreendedores.

Uso: streamlit run app.py
"""

import json
import os
import streamlit as st
import chromadb
from openai import AzureOpenAI
from pathlib import Path

import config

# ── Page Config ─────────────────────────────────────────────

st.set_page_config(
    page_title="G4 Podcast - Agente de Conhecimento",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers para ler secrets ────────────────────────────────

def _secret(key: str, default: str = "") -> str:
    """Le de st.secrets primeiro, depois os.environ."""
    if key in st.secrets:
        return str(st.secrets[key])
    return os.getenv(key, default)

# ── Init (cached) ───────────────────────────────────────────

@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path=str(config.CHROMADB_DIR))
    return client.get_or_create_collection(
        name="g4_podcast",
        metadata={"hnsw:space": "cosine"},
    )

@st.cache_resource
def init_openai():
    return AzureOpenAI(
        api_key=_secret("AZURE_OPENAI_API_KEY"),
        api_version=_secret("AZURE_EMBEDDING_API_VERSION", "2023-05-15"),
        azure_endpoint=_secret("AZURE_OPENAI_ENDPOINT"),
    )

@st.cache_resource
def init_chat_client():
    return AzureOpenAI(
        api_key=_secret("AZURE_OPENAI_API_KEY"),
        api_version=_secret("AZURE_CHAT_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=_secret("AZURE_OPENAI_ENDPOINT"),
    )

@st.cache_resource
def load_kg():
    if config.KG_GRAPH_FILE.exists():
        with open(config.KG_GRAPH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"entities": {}, "relationships": []}

@st.cache_resource
def load_episodes():
    if config.EPISODES_FILE.exists():
        with open(config.EPISODES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ── Debug: mostra diagnostico na sidebar ────────────────────
_api_key = _secret("AZURE_OPENAI_API_KEY")
_endpoint = _secret("AZURE_OPENAI_ENDPOINT")
_embed_model = _secret("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
_chat_model = _secret("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

with st.sidebar:
    with st.expander("Debug - Config", expanded=False):
        st.text(f"API Key: {_api_key[:8]}...{_api_key[-4:]}" if len(_api_key) > 12 else f"API Key: EMPTY ({len(_api_key)} chars)")
        st.text(f"Endpoint: {_endpoint}")
        st.text(f"Embed model: {_embed_model}")
        st.text(f"Chat model: {_chat_model}")
        st.text(f"Secrets keys: {list(st.secrets.keys())}")

collection = init_chroma()
embedding_client = init_openai()
chat_client = init_chat_client()
kg = load_kg()
episodes = load_episodes()


# ── Helper Functions ────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    response = embedding_client.embeddings.create(
        model=config.AZURE_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def search_knowledge(query: str, n_results: int = 8) -> list[dict]:
    """Busca semantica no ChromaDB."""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )
    if not results or not results["documents"] or not results["documents"][0]:
        return []

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "text": doc,
            "title": meta.get("title", ""),
            "guest": meta.get("guest", ""),
            "episode_id": meta.get("episode_id", ""),
            "relevance": round((1 - dist) * 100, 1),
            "people": meta.get("kg_people", ""),
            "companies": meta.get("kg_companies", ""),
            "concepts": meta.get("kg_concepts", ""),
        })
    return items


def search_entities(query: str, entity_type: str = "all") -> list[dict]:
    """Busca no knowledge graph."""
    query_lower = query.lower()
    results = []

    type_map = {
        "person": ("people", "Pessoa"),
        "company": ("companies", "Empresa"),
        "concept": ("concepts", "Conceito"),
        "book": ("books", "Livro"),
    }
    search_types = type_map if entity_type == "all" else {entity_type: type_map.get(entity_type, ("people", "?"))}

    for etype, (dict_key, label) in search_types.items():
        entities = kg.get("entities", {}).get(dict_key, {})
        for key, entity in entities.items():
            name = entity.get("name", key)
            if query_lower in name.lower() or query_lower in key.lower():
                results.append({
                    "name": name,
                    "type": label,
                    "episode_count": entity.get("episode_count", 0),
                    "roles": entity.get("roles", []),
                    "company": entity.get("company", ""),
                    "industry": entity.get("industry", ""),
                    "category": entity.get("category", ""),
                    "context": entity.get("context", ""),
                    "author": entity.get("author", ""),
                })

    results.sort(key=lambda x: x["episode_count"], reverse=True)
    return results[:20]


def generate_ai_answer(query: str, context_chunks: list[dict]) -> str:
    """Gera resposta contextualizada com GPT-4o usando os chunks encontrados."""
    context = "\n\n---\n\n".join(
        f"[{c['title']}] (Relevancia: {c['relevance']}%)\n{c['text']}"
        for c in context_chunks[:6]
    )

    response = chat_client.chat.completions.create(
        model=config.AZURE_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Voce e um consultor de negocios especialista no conteudo do G4 Podcast. "
                    "Responda a pergunta do empreendedor usando APENAS o contexto fornecido. "
                    "Cite episodios e convidados quando relevante. "
                    "Seja pratico, direto e acionavel. "
                    "Responda em portugues do Brasil. "
                    "Se nao houver informacao suficiente no contexto, diga que nao encontrou."
                ),
            },
            {
                "role": "user",
                "content": f"Contexto dos episodios:\n\n{context}\n\n---\n\nPergunta: {query}",
            },
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content


# ── Lead Gate ───────────────────────────────────────────────

def show_lead_gate():
    """Tela de captura de lead antes de liberar acesso."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 🎙️ Agente G4 Podcast")
        st.markdown(
            "### Acesse o conhecimento de **500+ episodios** do G4 Podcast "
            "com inteligencia artificial"
        )
        st.markdown(
            "- 🔍 Busca semantica em milhares de trechos\n"
            "- 🧠 Knowledge Graph com 9.800+ entidades\n"
            "- 🤖 Respostas contextualizadas por IA\n"
            "- 💼 Consultor de negocios personalizado"
        )
        st.markdown("---")
        st.markdown("**Preencha abaixo para acessar gratuitamente:**")

        with st.form("lead_form"):
            name = st.text_input("Nome completo")
            email = st.text_input("Email profissional")
            company = st.text_input("Nome da empresa")
            role = st.selectbox(
                "Cargo",
                ["CEO / Fundador", "C-Level", "Diretor", "Gerente", "Outro"],
            )
            revenue = st.selectbox(
                "Faturamento anual",
                [
                    "Ate R$500k",
                    "R$500k - R$2M",
                    "R$2M - R$10M",
                    "R$10M - R$50M",
                    "R$50M+",
                ],
            )
            submitted = st.form_submit_button("Acessar Agente G4", use_container_width=True)

            if submitted:
                if not name or not email:
                    st.error("Preencha nome e email para continuar.")
                else:
                    # Salva lead (append em arquivo JSON Lines)
                    lead = {
                        "name": name,
                        "email": email,
                        "company": company,
                        "role": role,
                        "revenue": revenue,
                    }
                    leads_file = Path(config.DATA_DIR) / "leads.jsonl"
                    with open(leads_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(lead, ensure_ascii=False) + "\n")

                    st.session_state["authenticated"] = True
                    st.session_state["user_name"] = name
                    st.session_state["user_company"] = company
                    st.rerun()


# ── Main App ────────────────────────────────────────────────

def show_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Ola, {st.session_state.get('user_name', 'Empreendedor')}! 👋")
        st.markdown(f"**{st.session_state.get('user_company', '')}**")
        st.markdown("---")

        st.markdown("### Estatisticas da Base")
        st.metric("Episodios", f"{len(episodes):,}")
        st.metric("Chunks indexados", f"{collection.count():,}")

        n_people = len(kg.get("entities", {}).get("people", {}))
        n_companies = len(kg.get("entities", {}).get("companies", {}))
        n_concepts = len(kg.get("entities", {}).get("concepts", {}))
        st.metric("Entidades no Grafo", f"{n_people + n_companies + n_concepts:,}")

        st.markdown("---")
        st.markdown(
            "💡 *Feito com IA pelo G4 Podcast*\n\n"
            "Dados de 500+ episodios processados com Azure OpenAI"
        )

    # Header
    st.markdown("# 🎙️ Agente de Conhecimento G4 Podcast")
    st.markdown("Pergunte qualquer coisa sobre negocios, gestao e crescimento.")

    # Tabs
    tab_search, tab_entities, tab_consultant = st.tabs([
        "🔍 Buscar Conhecimento",
        "🧠 Knowledge Graph",
        "💼 Consultor de Negocios",
    ])

    # ── Tab 1: Busca Semantica ──
    with tab_search:
        query = st.text_input(
            "Qual a sua pergunta?",
            placeholder="Ex: Como escalar vendas B2B? / O que o Nardon fala sobre cultura?",
            key="search_query",
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            n_results = st.slider("Resultados", 3, 15, 8, key="n_results")

        if query:
            with st.spinner("Buscando nos 500+ episodios..."):
                chunks = search_knowledge(query, n_results)

            if chunks:
                # Gera resposta IA
                with st.spinner("Gerando resposta com IA..."):
                    ai_answer = generate_ai_answer(query, chunks)

                st.markdown("### 🤖 Resposta do Agente")
                st.markdown(ai_answer)

                st.markdown("---")
                st.markdown("### 📋 Trechos Encontrados")

                for i, chunk in enumerate(chunks):
                    with st.expander(
                        f"**{chunk['title']}** — Relevancia: {chunk['relevance']}%",
                        expanded=(i < 2),
                    ):
                        if chunk["guest"]:
                            st.caption(f"Convidado: {chunk['guest']}")
                        st.markdown(chunk["text"])

                        if chunk["people"]:
                            st.caption(f"👤 Pessoas: {chunk['people']}")
                        if chunk["companies"]:
                            st.caption(f"🏢 Empresas: {chunk['companies']}")
                        if chunk["concepts"]:
                            st.caption(f"💡 Conceitos: {chunk['concepts']}")
            else:
                st.warning("Nenhum resultado encontrado. Tente reformular a pergunta.")

    # ── Tab 2: Knowledge Graph ──
    with tab_entities:
        entity_query = st.text_input(
            "Buscar entidade (pessoa, empresa, conceito, livro)",
            placeholder="Ex: Tallis Gomes, OKR, Product-Market Fit",
            key="entity_query",
        )

        entity_type = st.selectbox(
            "Tipo",
            ["all", "person", "company", "concept", "book"],
            format_func=lambda x: {
                "all": "Todos",
                "person": "Pessoa",
                "company": "Empresa",
                "concept": "Conceito",
                "book": "Livro",
            }.get(x, x),
            key="entity_type",
        )

        if entity_query:
            results = search_entities(entity_query, entity_type)
            if results:
                for r in results:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**[{r['type']}] {r['name']}**")
                            details = []
                            if r.get("roles"):
                                details.append(f"Papeis: {', '.join(r['roles'])}")
                            if r.get("company"):
                                details.append(f"Empresa: {r['company']}")
                            if r.get("industry"):
                                details.append(f"Setor: {r['industry']}")
                            if r.get("category"):
                                details.append(f"Categoria: {r['category']}")
                            if r.get("context"):
                                details.append(f"Contexto: {r['context']}")
                            if r.get("author"):
                                details.append(f"Autor: {r['author']}")
                            if details:
                                st.caption(" | ".join(details))
                        with col2:
                            st.metric("Episodios", r["episode_count"])
                        st.markdown("---")
            else:
                st.warning(f"Nenhuma entidade encontrada para '{entity_query}'.")

    # ── Tab 3: Consultor ──
    with tab_consultant:
        st.markdown(
            "Conte sobre seu negocio e receba insights personalizados "
            "baseados em 500+ episodios do G4 Podcast."
        )

        with st.form("consultant_form"):
            segmento = st.text_input(
                "Segmento da empresa",
                placeholder="Ex: SaaS B2B, varejo, educacao, saude",
            )
            desafio = st.text_area(
                "Principal desafio atual",
                placeholder="Ex: Escalar vendas sem perder margem, reter talentos tech, expandir para novos mercados",
            )
            faturamento = st.selectbox(
                "Faturamento anual",
                ["", "Ate R$500k", "R$500k - R$2M", "R$2M - R$10M", "R$10M - R$50M", "R$50M+"],
            )
            funcionarios = st.selectbox(
                "Tamanho da equipe",
                ["", "1-5", "6-20", "21-50", "51-200", "200+"],
            )
            submit_consultant = st.form_submit_button(
                "Gerar Insights Personalizados",
                use_container_width=True,
            )

        if submit_consultant and segmento and desafio:
            # Monta query contextualizada
            query = (
                f"Empresa do segmento {segmento}, "
                f"enfrentando o desafio de {desafio}. "
            )
            if faturamento:
                query += f"Faturamento: {faturamento}. "
            if funcionarios:
                query += f"Equipe: {funcionarios} funcionarios. "
            query += "Quais estrategias, conselhos e experiencias sao relevantes?"

            with st.spinner("Analisando base de conhecimento..."):
                chunks = search_knowledge(query, 12)

            if chunks:
                with st.spinner("Gerando consultoria personalizada..."):
                    context = "\n\n---\n\n".join(
                        f"[{c['title']}]\n{c['text']}" for c in chunks[:8]
                    )

                    response = chat_client.chat.completions.create(
                        model=config.AZURE_CHAT_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Voce e um consultor de negocios senior do G4 Educacao. "
                                    "Baseado no contexto dos episodios do G4 Podcast, gere um "
                                    "plano de acao personalizado para o empreendedor. "
                                    "Estruture em: 1) Diagnostico, 2) Recomendacoes (3-5 acoes concretas), "
                                    "3) Episodios recomendados para aprofundar. "
                                    "Cite convidados e episodios especificos. "
                                    "Seja pratico e acionavel. Responda em portugues do Brasil."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Contexto dos episodios:\n\n{context}\n\n---\n\n"
                                    f"Perfil do empreendedor:\n"
                                    f"- Segmento: {segmento}\n"
                                    f"- Desafio: {desafio}\n"
                                    f"- Faturamento: {faturamento or 'Nao informado'}\n"
                                    f"- Equipe: {funcionarios or 'Nao informado'}\n\n"
                                    f"Gere um plano de acao personalizado."
                                ),
                            },
                        ],
                        temperature=0.4,
                        max_tokens=3000,
                    )

                    st.markdown("### 📋 Plano de Acao Personalizado")
                    st.markdown(response.choices[0].message.content)

                    st.markdown("---")
                    st.markdown("### 🎙️ Episodios Usados como Base")
                    seen = set()
                    for c in chunks[:8]:
                        title = c["title"]
                        if title not in seen:
                            seen.add(title)
                            guest_str = f" — {c['guest']}" if c["guest"] else ""
                            st.markdown(f"- **{title}**{guest_str} ({c['relevance']}%)")


# ── Router ──────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    show_app()
else:
    show_lead_gate()
