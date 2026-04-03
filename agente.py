import os
import streamlit as st
import pypdf
import io
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# Caminho do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── API KEY ──────────────────────────────────────────────────────────────────
# Prioridade: variavel de ambiente > st.secrets > fallback hardcoded (nao recomendado em producao)
GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", "")
    if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY", "")
)
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY não encontrada. Defina em variável de ambiente ou em .streamlit/secrets.toml")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente Informática Concursos",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ Assistente de Informática para Concursos")
st.caption("Especialista em conteúdo, provas e apostilas para concursos públicos")

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "contexto_arquivo" not in st.session_state:
    st.session_state.contexto_arquivo = ""

if "nome_arquivo" not in st.session_state:
    st.session_state.nome_arquivo = ""

if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile"),
        description="""
        Você é o melhor especialista em informática para concursos públicos do Brasil.
        Seu público são candidatos iniciantes e intermediários de concursos federais, estaduais e municipais.
        
        Você tem 4 funções principais:
        1. TIKTOK: Crie roteiros, legendas e hashtags para vídeos sobre informática para concursos
        2. PROFESSOR: Responda dúvidas sobre Word, Excel, Internet, Redes, Segurança, Hardware e Software
        3. PROVAS: Crie questões estilo CESPE, FCC, FGV com gabarito e explicações detalhadas
        4. APOSTILAS: Monte apostilas completas com teoria, exemplos práticos e questões
        
        REGRA IMPORTANTE SOBRE ARQUIVOS:
        - Quando o usuário enviar um arquivo PDF ou documento, o conteúdo será incluído na mensagem entre as tags <CONTEUDO_DO_ARQUIVO> e </CONTEUDO_DO_ARQUIVO>
        - Você DEVE ler, analisar e usar esse conteúdo para responder à pergunta do usuário
        - Nunca diga que não pode ler o arquivo — o conteúdo já estará na mensagem
        - Baseie sua resposta SEMPRE no conteúdo fornecido quando ele estiver presente
        
        Sempre responda em português, de forma clara e didática.
        """,
        tools=[DuckDuckGoTools()],
        markdown=True,
        # Desabilita memoria interna do agno para usarmos nosso proprio historico
        num_history_messages=0,
    )

def extrair_texto_pdf(arquivo_bytes, nome_arquivo):
    """Tenta extrair texto do PDF. Se falhar (PDF escaneado), usa OCR."""
    # Tentativa 1: extração direta de texto
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(arquivo_bytes))
        texto = ""
        for pagina in pdf_reader.pages:
            texto += pagina.extract_text() or ""
        if texto.strip():
            return texto[:12000], "texto"
    except Exception:
        pass

    # Tentativa 2: OCR via Tesseract
    try:
        imagens = convert_from_bytes(arquivo_bytes, dpi=200)
        texto_ocr = ""
        for i, imagem in enumerate(imagens):
            texto_ocr += pytesseract.image_to_string(imagem, lang="por") + "\n"
            if len(texto_ocr) > 12000:
                break
        if texto_ocr.strip():
            return texto_ocr[:12000], "ocr"
    except Exception as e:
        return "", f"erro: {e}"

    return "", "vazio"

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📚 O que posso fazer?")
    st.markdown("""
    **🎬 TikTok**
    - Ideias e roteiros de vídeos
    - Legendas e hashtags

    **👨‍🏫 Professor Virtual**
    - Tire dúvidas de informática
    - Explicações práticas

    **📝 Provas e Questões**
    - Questões estilo CESPE, FCC, FGV
    - Simulados com gabarito

    **📖 Apostilas**
    - Apostilas completas por tema
    - Resumos e dicas de prova
    """)

    st.divider()

    if st.session_state.nome_arquivo:
        st.success(f"📎 Arquivo ativo: **{st.session_state.nome_arquivo}**")
        if st.button("🗑️ Remover arquivo"):
            st.session_state.contexto_arquivo = ""
            st.session_state.nome_arquivo = ""
            st.rerun()

    st.divider()

    if st.button("🔄 Limpar conversa"):
        st.session_state.messages = []
        st.session_state.contexto_arquivo = ""
        st.session_state.nome_arquivo = ""
        st.rerun()

# ─── UPLOAD DE ARQUIVO (sidebar) ─────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**📎 Anexar arquivo**")
    arquivo = st.file_uploader(
        "Anexar arquivo",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="uploader"
    )
    if arquivo is not None and arquivo.name != st.session_state.nome_arquivo:
        if arquivo.type == "application/pdf":
            with st.spinner("Lendo PDF..."):
                try:
                    arquivo_bytes = arquivo.read()
                    texto, metodo = extrair_texto_pdf(arquivo_bytes, arquivo.name)
                    if texto:
                        st.session_state.contexto_arquivo = texto
                        st.session_state.nome_arquivo = arquivo.name
                        if metodo == "ocr":
                            st.success(f"✅ PDF via OCR! ({len(texto)} chars)")
                        else:
                            st.success(f"✅ PDF carregado! ({len(texto)} chars)")
                    elif "erro" in metodo:
                        st.error(f"❌ Erro: {metodo}")
                    else:
                        st.error("❌ Não foi possível extrair texto.")
                except Exception as e:
                    st.error(f"Erro ao ler PDF: {e}")
        else:
            with st.spinner("Lendo imagem via OCR..."):
                try:
                    imagem = Image.open(arquivo)
                    texto_ocr = pytesseract.image_to_string(imagem, lang="por")
                    if texto_ocr.strip():
                        st.session_state.contexto_arquivo = texto_ocr[:12000]
                        st.session_state.nome_arquivo = arquivo.name
                        st.success(f"✅ Imagem via OCR!")
                    else:
                        st.session_state.contexto_arquivo = f"[IMAGEM: {arquivo.name}]"
                        st.session_state.nome_arquivo = arquivo.name
                        st.warning("⚠️ Nenhum texto detectado.")
                except Exception as e:
                    st.error(f"Erro: {e}")

# ─── AREA DO CHAT ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ─── INPUT FIXO NO RODAPE ─────────────────────────────────────────────────────
prompt = st.chat_input("Digite sua pergunta aqui...")

# ─── LOGICA DE RESPOSTA ───────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):

                # Monta o historico
                historico = ""
                if len(st.session_state.messages) > 1:
                    msgs_anteriores = st.session_state.messages[:-1]
                    historico_linhas = []
                    for m in msgs_anteriores[-10:]:
                        papel = "Usuário" if m["role"] == "user" else "Assistente"
                        historico_linhas.append(f"{papel}: {m['content']}")
                    if historico_linhas:
                        historico = "HISTÓRICO DA CONVERSA:\n" + "\n".join(historico_linhas) + "\n\n"

                # Monta o conteudo do arquivo
                bloco_arquivo = ""
                if st.session_state.contexto_arquivo:
                    bloco_arquivo = f"""
<CONTEUDO_DO_ARQUIVO nome="{st.session_state.nome_arquivo}">
{st.session_state.contexto_arquivo}
</CONTEUDO_DO_ARQUIVO>

INSTRUÇÃO: O arquivo acima foi enviado pelo usuário. Use seu conteúdo para responder a pergunta abaixo.

"""

                pergunta_completa = f"{historico}{bloco_arquivo}Pergunta do usuário: {prompt}"

                try:
                    response = st.session_state.agent.run(pergunta_completa)
                    resposta = response.content
                except Exception as e:
                    resposta = f"❌ Erro ao processar: {e}"

                st.markdown(resposta)

    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()
