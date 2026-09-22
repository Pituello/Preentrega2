"""
Acá se arma la cadena LCEL de verdad:

    prompt | modelo.with_structured_output(EntityExtraction)

con dos capas de resiliencia:
- .with_retry(): si el LLM devuelve un JSON mal formado o incompleto,
  reintenta con backoff exponencial.
- .with_fallbacks(): si el proveedor principal se queda sin reintentos,
  pasa automáticamente al siguiente proveedor de PROVIDER_CHAIN.
- build_parallel_chain() usa RunnableParallel para correr la
extracción de entidades y un resumen ejecutivo en texto plano, sobre el
mismo texto de entrada, al mismo tiempo.
"""
import os

from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from loguru import logger
from pydantic import ValidationError

from .models import get_model
from .prompt import extraction_prompt, resumen_prompt
from .schemas import EntityExtraction

ERRORES_REINTENTABLES = (
    ModelRateLimitError,
    ModelTimeoutError,
    ModelConnectionError,
    ModelAPIError,
    OutputParserException,
    ValidationError,
)

"""
Cada proveedor le pone un nombre distinto al campo que dice "por qué
# terminó la respuesta" dentro de response_metadata: OpenAI y Gemini usan
# 'finish_reason', Anthropic usa 'stop_reason', Ollama usa 'done_reason'.
# No hay un nombre único estandarizado, así que probamos los tres.
"""
_CLAVES_FINISH_REASON = ("finish_reason", "stop_reason", "done_reason")

"""Valores que indican "se cortó por límite de tokens", según cada proveedor
(OpenAI/Ollama: 'length', Anthropic: 'max_tokens', Gemini: 'MAX_TOKENS')."""
_VALORES_CORTE_POR_TOKENS = {"length", "max_tokens", "MAX_TOKENS", "max_output_tokens"}


def _finish_reason(raw_message) -> str | None:
    """Busca el finish_reason en response_metadata, sea cual sea el nombre
    que le puso el proveedor."""
    metadata = getattr(raw_message, "response_metadata", {}) or {}
    for clave in _CLAVES_FINISH_REASON:
        if clave in metadata:
            return metadata[clave]
    return None


def _revisar_resultado_estructurado(resultado: dict) -> EntityExtraction:
    """
    Post-procesa lo que devuelve with_structured_output(..., include_raw=True):
    {'raw': BaseMessage, 'parsed': EntityExtraction | None, 'parsing_error': Exception | None}

    - Chequea el finish_reason: si el modelo cortó la respuesta por falta de
      tokens, lo loguea -> suele ser la causa real de un parsing_error, y así
      no queda solo con un ValidationError genérico sin justificacion.
    - Con include_raw=True, LangChain NO lanza el error de parseo solo, te
      lo devuelve en 'parsing_error'. Acá lo relanzamos adrede, para
      que .with_retry() lo siga capturando exactamente igual que antes.

    Esta función corre UNA VEZ POR CADA INTENTO de .with_retry() (está adentro
    de la cadena que se reintenta), así que es el lugar correcto para loguear
    "se va a reintentar" con información real de qué pasó en ese intento
    puntual -> LangChain no expone un hook propio para esto en with_retry().
    """
    raw = resultado.get("raw")
    razon = _finish_reason(raw) if raw is not None else None
    if razon in _VALORES_CORTE_POR_TOKENS:
        logger.warning(f"el modelo cortó la respuesta por falta de tokens (finish_reason={razon!r})")

    if resultado.get("parsing_error") is not None:
        logger.warning(
            f"el LLM devolvió algo que no cumple el esquema, se reintenta: {resultado['parsing_error']}"
        )
        raise resultado["parsing_error"]

    logger.debug("respuesta validada correctamente contra el esquema")
    return resultado["parsed"]


def _proveedores_disponibles() -> list[str]:
    """
    Lee PROVIDER_CHAIN del .env y devuelve solo los nombres que realmente
    tienen credenciales configuradas (en el mismo orden del .env). Esto se
    calcula UNA sola vez y se reutiliza en toda la cadena, para no repetir
    en cada lugar la lógica de "salteo si falta la key".
    """
    nombres = os.getenv("PROVIDER_CHAIN", "openai,anthropic").split(",")
    nombres = [n.strip() for n in nombres if n.strip()]

    disponibles = []
    for nombre in nombres:
        try:
            get_model(nombre)  # solo valida que se pueda armar, no lo guardamos
            disponibles.append(nombre)
        except ValueError as e:
            logger.warning(f"no pude armar el modelo de '{nombre}': {e}")

    if not disponibles:
        raise RuntimeError("no se pudo armar ningún modelo, revisá tus API keys en el .env")
    return disponibles


def _con_log_de_intento(provider: str):
    """
    Un paso mudo al principio de la cadena que solo loguea "probando con
    este proveedor" y deja pasar el input sin tocarlo. Existe porque
    .with_fallbacks() SOLO te muestra el error del PRIMER proveedor que
    falló si al final todos fallan (RunnableWithFallbacks.ainvoke
    hace `raise first_error`) -> sin este log no hay forma de saber si
    un fallback llegó a intentarse o no.
    """

    def _log(inputs):
        logger.info(f"probando extracción con proveedor '{provider}'")
        return inputs

    return RunnableLambda(_log)


def _con_log_de_falla_final(provider: str, cadena):
    """
    Envuelve la cadena (que ya tiene su propio .with_retry() adentro) para
    loguear el error REAL si agota todos los reintentos y termina fallando,
    antes de que .with_fallbacks() pase al siguiente proveedor -> o, si es
    el último de la cadena, antes de que ese error se pierda adentro del
    `raise first_error` de with_fallbacks() (ver build_resilient_chain).

    Sin esto, si un proveedor falla por algo que pasa ANTES de llegar a
    _revisar_resultado_estructurado (ej: no se pudo conectar, el modelo no
    existe, timeout de red), no queda ningún registro de qué pasó
    específicamente con ESE proveedor.
    """

    async def _invocar(inputs):
        try:
            return await cadena.ainvoke(inputs)
        except Exception as e:
            logger.error(f"proveedor '{provider}' agotó sus intentos y falló: {type(e).__name__}: {e}")
            raise

    return RunnableLambda(_invocar)


def _cadena_estructurada(provider: str):
    """
    prompt | modelo.with_structured_output(EntityExtraction, include_raw=True) | revisión

    include_raw=True nos da acceso al mensaje crudo (para leer finish_reason)
    además del objeto parseado. _revisar_resultado_estructurado() se encarga
    de sacarle el objeto EntityExtraction final, avisar si se cortó por
    tokens, y relanzar el error de parseo si lo hubo (para que el retry de
    abajo lo agarre).
    """
    modelo = get_model(provider)
    modelo_estructurado = modelo.with_structured_output(EntityExtraction, include_raw=True)
    cadena = (
        _con_log_de_intento(provider)
        | extraction_prompt
        | modelo_estructurado
        | RunnableLambda(_revisar_resultado_estructurado)
    )
    cadena_con_retry = cadena.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    return _con_log_de_falla_final(provider, cadena_con_retry)


def build_resilient_chain(disponibles: list[str] | None = None):
    """
    Arma la cadena principal + fallback. El primer proveedor disponible es
    el principal; el resto son fallbacks, en orden (.with_fallbacks()).
    `disponibles` es opcional: si no se pasa, se calcula acá mismo. Se puede
    pasar ya calculado para no recorrer PROVIDER_CHAIN dos veces (ver
    build_parallel_chain más abajo).
    """
    if disponibles is None:
        disponibles = _proveedores_disponibles()

    cadena_principal, *fallbacks = [_cadena_estructurada(n) for n in disponibles]

    if fallbacks:
        return cadena_principal.with_fallbacks(fallbacks)
    return cadena_principal


async def process_text(texto: str) -> EntityExtraction:
    """
    Punto de entrada "oficial": una función asíncrona
    que recibe el texto libre, corre la cadena LCEL completa (con retry +
    fallback) vía .ainvoke(), y devuelve el objeto EntityExtraction validado.

    Los logs de warning con cada reintento salen de
    _revisar_resultado_estructurado() (arriba), que corre dentro de la
    cadena en cada intento; acá solo logueamos el inicio y el resultado
    final (éxito o fallo definitivo tras agotar reintentos y fallbacks).
    """
    logger.info(f"process_text: iniciando extracción sobre un texto de {len(texto)} caracteres")
    cadena = build_resilient_chain()

    try:
        resultado = await cadena.ainvoke({"texto": texto})
    except Exception as e:
        logger.error(f"process_text: falló después de agotar reintentos y fallbacks -> {e}")
        raise

    logger.info("process_text: extracción completada y validada OK")
    return resultado


def build_parallel_chain():
    """
    RunnableParallel corriendo dos tareas independientes a la vez
    sobre el mismo texto de entrada -> la extracción de entidades (con toda
    la resiliencia de arriba) y un resumen ejecutivo en texto plano.

    Importante: RunnableParallel usa asyncio.gather() por debajo, que por
    default CANCELA las demás tareas en cuanto UNA falla (aunque las otras
    estén a mitad de un reintento que iba bien). Por eso `cadena_resumen`
    también necesita su propio .with_retry(): si no lo tuviera y fallara
    rápido ante un error transitorio, romperia `cadena_entidades`
    aunque esta se hubiera recuperado sola un instante después.
    """
    disponibles = _proveedores_disponibles()

    cadena_entidades = build_resilient_chain(disponibles)

    modelo_resumen = get_model(disponibles[0])
    cadena_resumen = (resumen_prompt | modelo_resumen | StrOutputParser()).with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    return RunnableParallel(
        entidades=cadena_entidades,
        resumen_ejecutivo=cadena_resumen,
    )
