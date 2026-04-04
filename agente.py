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
GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY", "")
    if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY", "")
)
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY não encontrada.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

# Modelos em ordem de preferência (fallback automático)
MODELOS_GROQ = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mistral-saba-24b",
]

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente Informática Concursos",
    page_icon="🖥️",
    layout="wide"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Esconde label e dropzone do uploader, mantém só o botão */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    min-height: unset !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 8px !important;
    padding: 6px 12px !important;
}
[data-testid="stFileUploader"] { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexto_arquivo" not in st.session_state:
    st.session_state.contexto_arquivo = ""
if "nome_arquivo" not in st.session_state:
    st.session_state.nome_arquivo = ""

if "modelo_atual" not in st.session_state:
    st.session_state.modelo_atual = 0

def criar_agente(modelo_idx=0):
    # Modelos menores (fallback) nao suportam tools bem, desabilita
    usar_tools = modelo_idx == 0
    return Agent(
        model=Groq(id=MODELOS_GROQ[modelo_idx]),
        description="""
        Você é o melhor especialista em informática para concursos públicos do Brasil.
        Seu público são candidatos iniciantes e intermediários de concursos federais, estaduais e municipais.

        Você tem 4 funções principais:
        1. TIKTOK: Crie roteiros, legendas e hashtags para vídeos sobre informática para concursos
        2. PROFESSOR: Responda dúvidas sobre Word, Excel, Internet, Redes, Segurança, Hardware e Software
        3. PROVAS: Crie questões estilo CESPE, FCC, FGV com gabarito e explicações detalhadas.
        Quando criar questões de múltipla escolha, SEMPRE use este formato exato:

        [Enunciado da questão]

        Alternativas

        A
        [Texto da alternativa A]

        B
        [Texto da alternativa B]

        C
        [Texto da alternativa C]

        D
        [Texto da alternativa D]

        E (se houver)
        [Texto da alternativa E]

        ---
        **Gabarito:** [Letra]
        **Explicação:** [Explicação detalhada]
        4. APOSTILAS: Monte apostilas completas com teoria, exemplos práticos e questões

        REGRA CRÍTICA SOBRE ARQUIVOS — NUNCA IGNORE:
        - O conteúdo do arquivo JÁ ESTÁ incluído na mensagem entre as tags <CONTEUDO_DO_ARQUIVO> e </CONTEUDO_DO_ARQUIVO>
        - Você NÃO precisa acessar nenhum sistema externo para ler o arquivo
        - O texto já foi extraído e está disponível para você AGORA na mensagem
        - NUNCA diga que não pode ler arquivos — o conteúdo já está aqui
        - SEMPRE use o conteúdo entre as tags para responder
        - Se as tags estiverem presentes, LEIA o conteúdo e responda baseado nele

        Sempre responda em português, de forma clara e didática.
        """,
        tools=[DuckDuckGoTools()] if usar_tools else [],
        markdown=True,
        num_history_messages=0,
    )

if "agent" not in st.session_state:
    st.session_state.agent = criar_agente(st.session_state.modelo_atual)

# ─── FUNCAO OCR ───────────────────────────────────────────────────────────────
def extrair_texto_pdf(arquivo_bytes, nome_arquivo):
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(arquivo_bytes))
        texto = ""
        for pagina in pdf_reader.pages:
            texto += pagina.extract_text() or ""
        if texto.strip():
            return texto[:12000], "texto"
    except Exception:
        pass
    try:
        imagens = convert_from_bytes(arquivo_bytes, dpi=200)
        texto_ocr = ""
        for imagem in imagens:
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

    st.divider()
    st.caption(f"🤖 Modelo: `{MODELOS_GROQ[st.session_state.modelo_atual]}`")

    if st.session_state.nome_arquivo:
        st.success(f"📎 **{st.session_state.nome_arquivo}**")
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

# ─── TITULO ───────────────────────────────────────────────────────────────────
st.title("🖥️ Assistente de Informática para Concursos")
st.caption("Especialista em conteúdo, provas e apostilas para concursos públicos")

# ─── HISTORICO DO CHAT ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── RODAPE: UPLOAD + CHAT INPUT LADO A LADO ──────────────────────────────────
col_upload, col_input = st.columns([1, 11])

with col_upload:
    arquivo = st.file_uploader(
        "📎",
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
                        st.toast(f"✅ PDF carregado!" if metodo == "texto" else f"✅ PDF via OCR!")
                    elif "erro" in metodo:
                        st.toast(f"❌ Erro: {metodo}")
                    else:
                        st.toast("❌ Não foi possível extrair texto.")
                except Exception as e:
                    st.toast(f"Erro ao ler PDF: {e}")
        else:
            with st.spinner("Lendo imagem via OCR..."):
                try:
                    imagem = Image.open(arquivo)
                    texto_ocr = pytesseract.image_to_string(imagem, lang="por")
                    if texto_ocr.strip():
                        st.session_state.contexto_arquivo = texto_ocr[:12000]
                        st.session_state.nome_arquivo = arquivo.name
                        st.toast("✅ Imagem via OCR carregada!")
                    else:
                        st.session_state.contexto_arquivo = f"[IMAGEM: {arquivo.name}]"
                        st.session_state.nome_arquivo = arquivo.name
                        st.toast("⚠️ Nenhum texto detectado.")
                except Exception as e:
                    st.toast(f"Erro: {e}")

with col_input:
    prompt = st.chat_input("Digite sua pergunta aqui...")

# ─── LOGICA DE RESPOSTA ───────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):

            historico = ""
            if len(st.session_state.messages) > 1:
                historico_linhas = []
                for m in st.session_state.messages[:-1][-10:]:
                    papel = "Usuário" if m["role"] == "user" else "Assistente"
                    historico_linhas.append(f"{papel}: {m['content']}")
                if historico_linhas:
                    historico = "HISTÓRICO DA CONVERSA:\n" + "\n".join(historico_linhas) + "\n\n"

            bloco_arquivo = ""
            if st.session_state.contexto_arquivo:
                bloco_arquivo = f"""
ATENÇÃO: O usuário enviou um arquivo. O conteúdo completo já foi extraído e está abaixo. USE ESSE CONTEÚDO para responder.

<CONTEUDO_DO_ARQUIVO nome="{st.session_state.nome_arquivo}">
{st.session_state.contexto_arquivo}
</CONTEUDO_DO_ARQUIVO>

INSTRUÇÃO OBRIGATÓRIA: Você tem acesso ao conteúdo acima. Responda a pergunta do usuário usando esse conteúdo. Não diga que não pode ler arquivos.

"""

            pergunta_completa = f"{historico}{bloco_arquivo}Pergunta do usuário: {prompt}"

            def is_rate_limit(text):
                keywords = ["rate_limit", "rate limit", "429", "tokens per day", "tpd", "try again", "rate limit reached", "resource_exhausted"]
                return any(k in text.lower() for k in keywords)

            try:
                resposta = None
                for i in range(len(MODELOS_GROQ)):
                    idx = (st.session_state.modelo_atual + i) % len(MODELOS_GROQ)
                    try:
                        if i > 0:
                            st.session_state.modelo_atual = idx
                            st.session_state.agent = criar_agente(idx)
                            st.toast(f"⚠️ Trocando para {MODELOS_GROQ[idx]}...")
                        response = st.session_state.agent.run(pergunta_completa)
                        conteudo = response.content

                        # Verifica se o conteudo em si é um erro de rate limit
                        if conteudo and is_rate_limit(conteudo):
                            continue

                        resposta = conteudo
                        break
                    except Exception as e:
                        if is_rate_limit(str(e)):
                            continue
                        raise e

                if not resposta:
                    resposta = "❌ Todos os modelos atingiram o limite. Tente novamente em algumas horas."
            except Exception as e:
                resposta = f"❌ Erro ao processar: {e}"

            st.markdown(resposta)

    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()
