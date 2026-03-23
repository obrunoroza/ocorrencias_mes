# app.py
import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard de Ocorrências por Mês",
    layout="wide"
)

st.title("📊 Dashboard de Ocorrências por Mês")
st.caption(
    "Faça upload de várias planilhas Excel com nome no padrão MM-AA, por exemplo: 01-26, 02-26, 03-26."
)

# ============================================================
# COLUNAS ESPERADAS / ALIASES
# ============================================================
COLUNAS_MAP = {
    "tipo_plano": [
        "Tipo Plano se UNIMED",
        "Tipo Plano UNIMED",
        "Tipo Plano",
    ],
    "data_inclusao": [
        "Inclusão  Req. HBIS",
        "Inclusão Req. HBIS",
        "Data Inclusão",
        "Data",
    ],
    "num_requisicao": [
        "Nº Requisição",
        "No Requisição",
        "Numero Requisição",
    ],
    "num_aviso": [
        "Nº Aviso",
        "No Aviso",
        "Numero Aviso",
    ],
    "justificativa": [
        "Justificativa da Espécie de Req",
        "Justificativa",
    ],
    "ocorrido": [
        "Ocorrido",
        "OCORRIDO",
    ],
    "motivo": [
        "Motivo",
        "MOTIVO",
    ],
    "observacao": [
        "Observação",
        "Observacao",
        "OBSERVAÇÃO",
    ],
}

COLUNAS_ANALISE = ["ocorrido", "motivo", "justificativa"]

# ============================================================
# FUNÇÕES
# ============================================================
def normalizar_nome_coluna(nome: str) -> str:
    return re.sub(r"\s+", " ", str(nome).strip())

def encontrar_coluna(df: pd.DataFrame, aliases: list[str]) -> str | None:
    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    for alias in aliases:
        chave = normalizar_nome_coluna(alias)
        if chave in mapa:
            return mapa[chave]
    return None

def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalizar_nome_coluna(c) for c in df.columns]

    renomear = {}
    for canonica, aliases in COLUNAS_MAP.items():
        encontrada = encontrar_coluna(df, aliases)
        if encontrada is not None:
            renomear[encontrada] = canonica

    df = df.rename(columns=renomear)

    # Garante colunas de análise mesmo que não existam
    for col in ["tipo_plano", "data_inclusao", "num_requisicao", "num_aviso",
                "justificativa", "ocorrido", "motivo", "observacao"]:
        if col not in df.columns:
            df[col] = np.nan

    return df

def limpar_texto_serie(serie: pd.Series) -> pd.Series:
    return (
        serie.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r'^"|"$', "", regex=True)
        .str.strip()
        .replace("", np.nan)
    )

def extrair_mes_arquivo(nome_arquivo: str) -> str | None:
    """
    Espera nomes como:
    01-26.xlsx
    02-26.xls
    base_03-26.xlsx
    """
    match = re.search(r"(?<!\d)(\d{2})-(\d{2})(?!\d)", nome_arquivo)
    if not match:
        return None

    mm, aa = match.groups()
    ano = 2000 + int(aa)
    mes = int(mm)

    if mes < 1 or mes > 12:
        return None

    return f"{ano:04d}-{mes:02d}"

@st.cache_data(show_spinner=False)
def carregar_arquivo(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    erros = []
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        return pd.DataFrame(), [f"Erro ao ler '{uploaded_file.name}': {e}"]

    df = padronizar_colunas(df)

    # Limpeza de texto
    for col in ["tipo_plano", "justificativa", "ocorrido", "motivo", "observacao"]:
        df[col] = limpar_texto_serie(df[col])

    # Data
    df["data_inclusao"] = pd.to_datetime(df["data_inclusao"], errors="coerce")

    # Mês a partir do nome do arquivo
    mes_ref = extrair_mes_arquivo(uploaded_file.name)
    if mes_ref is None:
        erros.append(
            f"Não foi possível identificar o mês no nome do arquivo '{uploaded_file.name}'. "
            "Use o padrão MM-AA, por exemplo: 01-26.xlsx"
        )
        df["mes_ref"] = np.nan
    else:
        df["mes_ref"] = mes_ref

    df["arquivo_origem"] = uploaded_file.name
    return df, erros

def tabela_top(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    base = df[df[coluna].notna()].copy()
    if base.empty:
        return pd.DataFrame(columns=[coluna, "qtd", "percentual"])
    out = base[coluna].value_counts().reset_index()
    out.columns = [coluna, "qtd"]
    out["percentual"] = (out["qtd"] / out["qtd"].sum() * 100).round(2)
    return out

def montar_pareto(df: pd.DataFrame, coluna: str = "motivo") -> pd.DataFrame:
    base = tabela_top(df, coluna)
    if base.empty:
        return pd.DataFrame(columns=[coluna, "qtd", "percentual", "percentual_acumulado"])
    base["percentual_acumulado"] = base["percentual"].cumsum().round(2)
    return base

def grafico_barras_horizontal(df_top: pd.DataFrame, categoria: str, titulo: str):
    if df_top.empty:
        st.info(f"Sem dados para {titulo.lower()}.")
        return

    fig = px.bar(
        df_top,
        x="qtd",
        y=categoria,
        orientation="h",
        text="qtd",
        title=titulo,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, 60 * len(df_top))
    )
    st.plotly_chart(fig, use_container_width=True)

def grafico_pareto(df_pareto: pd.DataFrame, categoria: str = "motivo"):
    if df_pareto.empty:
        st.info("Sem dados para o Pareto dos Motivos.")
        return

    fig_bar = px.bar(
        df_pareto,
        x=categoria,
        y="qtd",
        text="qtd",
        title="Pareto dos Motivos"
    )
    fig_bar.update_layout(xaxis_tickangle=-35)

    # Linha acumulada usando graph_objects para eixo secundário simples
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df_pareto[categoria],
            y=df_pareto["qtd"],
            name="Quantidade",
            text=df_pareto["qtd"],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_pareto[categoria],
            y=df_pareto["percentual_acumulado"],
            name="% acumulado",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Pareto dos Motivos",
        xaxis_tickangle=-35,
        height=500,
    )
    fig.update_yaxes(title_text="Quantidade", secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", range=[0, 105], secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

def grafico_evolucao_mensal(df: pd.DataFrame):
    if df.empty or df["mes_ref"].dropna().empty:
        st.info("Sem dados para evolução mensal.")
        return

    evol = (
        df.dropna(subset=["mes_ref"])
        .groupby("mes_ref", as_index=False)
        .size()
        .rename(columns={"size": "qtd"})
        .sort_values("mes_ref")
    )

    fig = px.bar(
        evol,
        x="mes_ref",
        y="qtd",
        text="qtd",
        title="Evolução por Mês"
    )
    st.plotly_chart(fig, use_container_width=True)

def exportar_excel(df_base: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_base.to_excel(writer, sheet_name="base_consolidada", index=False)

        for col in COLUNAS_ANALISE:
            tabela = (
                df_base[df_base[col].notna()]
                .groupby(["mes_ref", col], as_index=False)
                .size()
                .rename(columns={"size": "qtd"})
                .sort_values(["mes_ref", "qtd"], ascending=[True, False])
            )
            tabela.to_excel(writer, sheet_name=f"{col}_mes"[:31], index=False)

        pareto = (
            df_base[df_base["motivo"].notna()]
            .groupby(["mes_ref", "motivo"], as_index=False)
            .size()
            .rename(columns={"size": "qtd"})
            .sort_values(["mes_ref", "qtd"], ascending=[True, False])
        )
        pareto.to_excel(writer, sheet_name="pareto_motivos", index=False)

    return output.getvalue()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Configurações")

    uploaded_files = st.file_uploader(
        "Upload das planilhas Excel",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="Envie várias planilhas, uma para cada mês, com nome no padrão MM-AA."
    )

    top_n = st.slider(
        "Quantidade padrão exibida no Top",
        min_value=5,
        max_value=30,
        value=5,
        step=1
    )

    st.markdown("---")
    st.markdown("**Colunas esperadas**")
    st.caption("Ocorrido, Motivo, Justificativa e, opcionalmente, demais colunas auxiliares.")

# ============================================================
# CARGA E CONSOLIDAÇÃO
# ============================================================
if not uploaded_files:
    st.info("Envie uma ou mais planilhas para iniciar a análise.")
    st.stop()

frames = []
erros_gerais = []

for file in uploaded_files:
    df_tmp, erros = carregar_arquivo(file)
    if not df_tmp.empty:
        frames.append(df_tmp)
    erros_gerais.extend(erros)

if erros_gerais:
    with st.expander("Avisos de importação", expanded=True):
        for erro in erros_gerais:
            st.warning(erro)

if not frames:
    st.error("Nenhum arquivo válido foi carregado.")
    st.stop()

df = pd.concat(frames, ignore_index=True)

# Remove arquivos sem mês identificado
df = df[df["mes_ref"].notna()].copy()

if df.empty:
    st.error("Os arquivos foram lidos, mas nenhum possui mês válido no nome no padrão MM-AA.")
    st.stop()

meses_disponiveis = sorted(df["mes_ref"].dropna().unique().tolist())

# ============================================================
# FILTROS
# ============================================================
st.subheader("Filtros")

col1, col2 = st.columns([2, 2])

with col1:
    meses_selecionados = st.multiselect(
        "Meses",
        options=meses_disponiveis,
        default=meses_disponiveis
    )

with col2:
    tipo_plano_opcoes = sorted([x for x in df["tipo_plano"].dropna().unique().tolist()])
    tipos_plano = st.multiselect(
        "Tipo de plano",
        options=tipo_plano_opcoes,
        default=tipo_plano_opcoes
    )

df_filtrado = df.copy()

if meses_selecionados:
    df_filtrado = df_filtrado[df_filtrado["mes_ref"].isin(meses_selecionados)]

if tipos_plano:
    df_filtrado = df_filtrado[df_filtrado["tipo_plano"].isin(tipos_plano)]

# ============================================================
# KPIs
# ============================================================
st.subheader("Visão Geral")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros", f"{len(df_filtrado):,}".replace(",", "."))
k2.metric("Meses analisados", len(df_filtrado["mes_ref"].dropna().unique()))
k3.metric("Arquivos carregados", df_filtrado["arquivo_origem"].nunique())
k4.metric("Tipos de plano", df_filtrado["tipo_plano"].nunique())

# ============================================================
# EVOLUÇÃO MENSAL
# ============================================================
st.subheader("Evolução por Mês")
grafico_evolucao_mensal(df_filtrado)

# ============================================================
# ANÁLISE POR MÊS
# ============================================================
st.subheader("Análises por Mês")

abas = st.tabs([f"📅 {mes}" for mes in meses_selecionados] if meses_selecionados else ["Sem mês"])

for aba, mes in zip(abas, meses_selecionados):
    with aba:
        base_mes = df_filtrado[df_filtrado["mes_ref"] == mes].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros no mês", len(base_mes))
        c2.metric("Ocorridos distintos", base_mes["ocorrido"].nunique(dropna=True))
        c3.metric("Motivos distintos", base_mes["motivo"].nunique(dropna=True))

        st.markdown("---")

        top_ocorridos = tabela_top(base_mes, "ocorrido").head(top_n)
        top_motivos = tabela_top(base_mes, "motivo").head(top_n)
        top_just = tabela_top(base_mes, "justificativa").head(top_n)
        pareto = montar_pareto(base_mes, "motivo")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Ocorridos")
            grafico_barras_horizontal(top_ocorridos, "ocorrido", f"Top {top_n} Ocorridos - {mes}")
            with st.expander("Ver tabela completa de Ocorridos"):
                st.dataframe(tabela_top(base_mes, "ocorrido"), use_container_width=True)

        with col_b:
            st.markdown("### Motivos")
            grafico_barras_horizontal(top_motivos, "motivo", f"Top {top_n} Motivos - {mes}")
            with st.expander("Ver tabela completa de Motivos"):
                st.dataframe(tabela_top(base_mes, "motivo"), use_container_width=True)

        st.markdown("### Justificativa")
        grafico_barras_horizontal(top_just, "justificativa", f"Top {top_n} Justificativas - {mes}")
        with st.expander("Ver tabela completa de Justificativas"):
            st.dataframe(tabela_top(base_mes, "justificativa"), use_container_width=True)

        st.markdown("### Pareto dos Motivos")
        grafico_pareto(pareto.head(max(top_n, 10)))
        with st.expander("Ver tabela completa do Pareto"):
            st.dataframe(pareto, use_container_width=True)

# ============================================================
# ANÁLISE CONSOLIDADA
# ============================================================
st.subheader("Consolidado do Período Selecionado")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ocorridos", "Motivos", "Justificativas", "Pareto dos Motivos"]
)

with tab1:
    base = tabela_top(df_filtrado, "ocorrido").head(top_n)
    grafico_barras_horizontal(base, "ocorrido", f"Top {top_n} Ocorridos - Consolidado")
    with st.expander("Ver tabela completa"):
        st.dataframe(tabela_top(df_filtrado, "ocorrido"), use_container_width=True)

with tab2:
    base = tabela_top(df_filtrado, "motivo").head(top_n)
    grafico_barras_horizontal(base, "motivo", f"Top {top_n} Motivos - Consolidado")
    with st.expander("Ver tabela completa"):
        st.dataframe(tabela_top(df_filtrado, "motivo"), use_container_width=True)

with tab3:
    base = tabela_top(df_filtrado, "justificativa").head(top_n)
    grafico_barras_horizontal(base, "justificativa", f"Top {top_n} Justificativas - Consolidado")
    with st.expander("Ver tabela completa"):
        st.dataframe(tabela_top(df_filtrado, "justificativa"), use_container_width=True)

with tab4:
    base = montar_pareto(df_filtrado, "motivo")
    grafico_pareto(base.head(max(top_n, 10)))
    with st.expander("Ver tabela completa"):
        st.dataframe(base, use_container_width=True)

# ============================================================
# DOWNLOAD
# ============================================================
st.subheader("Exportação")

arquivo_excel = exportar_excel(df_filtrado)
st.download_button(
    label="📥 Baixar análise consolidada em Excel",
    data=arquivo_excel,
    file_name="analise_dashboard_consolidada.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ============================================================
# RODAPÉ / DIAGNÓSTICO
# ============================================================
with st.expander("Diagnóstico da base"):
    preenchimento = pd.DataFrame({
        "coluna": df_filtrado.columns,
        "nao_nulos": [df_filtrado[c].notna().sum() for c in df_filtrado.columns],
        "nulos": [df_filtrado[c].isna().sum() for c in df_filtrado.columns],
    })
    preenchimento["perc_preenchimento"] = (
        preenchimento["nao_nulos"] / len(df_filtrado) * 100
    ).round(2)
    st.dataframe(preenchimento.sort_values("perc_preenchimento", ascending=False), use_container_width=True)
