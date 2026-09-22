"""
get_model(): arma el chat model de LangChain según el proveedor, leyendo
todo de variables de entorno. 

Nota sobre timeouts: cada integración de LangChain usa un nombre de
parámetro distinto para esto (no hay un "timeout" único y genérico), así
que hay que setearlo a mano en cada una. ChatOllama especialmente lo requiere
ya que no tiene timeouts por default al correr localmente.
"""
import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel

# La librería google-genai (la usa ChatGoogleGenerativeAI por debajo) loguea
# un aviso informativo sobre "automatic function calling" en CADA llamada
# con tools -> es justo lo que hace with_structured_output(). No es un error
# ni afecta el resultado, así que lo silenciamos subiendo el nivel de su
# logger puntual (no tocamos logging en general, solo este logger).
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

TIMEOUT_SEGUNDOS = 30
TIMEOUT_OLLAMA_SEGUNDOS = 60  # local, a veces tarda más en arrancar el modelo


def get_model(provider: str) -> BaseChatModel:
    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("falta OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0,
            request_timeout=TIMEOUT_SEGUNDOS,
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("falta ANTHROPIC_API_KEY")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=api_key,
            temperature=0,
            default_request_timeout=TIMEOUT_SEGUNDOS,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("falta GEMINI_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=0,
            timeout=TIMEOUT_SEGUNDOS,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
            async_client_kwargs={"timeout": TIMEOUT_OLLAMA_SEGUNDOS},
            client_kwargs={"timeout": TIMEOUT_OLLAMA_SEGUNDOS},
        )

    raise ValueError(f"proveedor '{provider}' no soportado")
