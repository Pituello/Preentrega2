# Pre-Entrega 2 — Pipeline de Extracción de Entidades Técnicas

**AI Engineering — Coderhouse**

Pipeline construido con LangChain y LCEL (LangChain Expression Language)
que recibe un texto libre (un ticket, un mensaje de soporte, un log de
error) y devuelve un objeto validado contra un esquema de Pydantic,
independientemente de la ambigüedad del texto de entrada.

## 1. Resumen del proyecto

| Aspecto | Detalle |
|---|---|
| Objetivo | Extraer entidades técnicas estructuradas a partir de texto libre |
| Framework | LangChain (LCEL) |
| Validación | Pydantic v2 |
| Proveedores soportados | OpenAI, Anthropic, Google Gemini, Ollama (local) |
| Resiliencia | Reintentos con backoff exponencial + fallback automático entre proveedores |
| Punto de entrada | `process_text(texto: str) -> EntityExtraction` |

## 2. Ejemplo de entrada y salida

**Entrada:**
> "API con caché en Redis y persistencia en PostgreSQL, usando FastAPI.
> Cuello de botella en conexiones concurrentes."

**Salida:**

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## 3. Arquitectura

```
texto libre
     │
     ▼
ChatPromptTemplate (prompt.py)
     │
     ▼
modelo.with_structured_output(EntityExtraction, include_raw=True)
     │
     ▼
revisión de finish_reason + validación Pydantic
     │
     ├── OK ──────────────────────────► EntityExtraction
     │
     └── error retryable ──► .with_retry() (backoff exponencial, hasta 3 intentos)
                │
                └── reintentos agotados ──► .with_fallbacks() (siguiente proveedor)
```

### Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `pipeline/schemas.py` | Modelo Pydantic `EntityExtraction` (contrato de datos) |
| `pipeline/prompt.py` | `ChatPromptTemplate` de extracción y de resumen ejecutivo |
| `pipeline/models.py` | Instanciación del modelo de LangChain según proveedor (timeouts, configuración) |
| `pipeline/chain.py` | Composición LCEL, lógica de resiliencia, `process_text()` |
| `main.py` | Script de uso interactivo (entrada por consola) |
| `test_manual.py` | Prueba de estrés con texto ambiguo |

## 4. Instalación

```bash
pip install -r requirements.txt
cp .env.example .env
```

Completar en `.env` al menos una de las siguientes credenciales:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`. Ollama no requiere
credenciales, pero necesita el servicio corriendo localmente.

## 5. Uso

### Función principal

```python
from pipeline.chain import process_text

resultado = await process_text("texto libre a analizar")
# resultado: EntityExtraction ya validado
```

`process_text()` es el punto de entrada asíncrono que ejecuta la cadena
LCEL completa —con reintentos y fallback incluidos— mediante `.ainvoke()`.

### Scripts incluidos

```bash
python main.py          # uso interactivo: solicita texto por consola
python test_manual.py   # prueba de estrés con un texto ambiguo predefinido
```

## 6. Diseño de la resiliencia

### 6.1 Reintentos acotados por tipo de error

Cada cadena está envuelta con `.with_retry(stop_after_attempt=3,
wait_exponential_jitter=True)`, restringido a un conjunto específico de
excepciones (`ERRORES_REINTENTABLES`): límite de tasa, timeout, error de
conexión, error de API transitorio, y salida que no cumple el esquema
esperado. Deliberadamente **no** se reintenta ante errores de autenticación
o de configuración (API key inválida, modelo inexistente), ya que estos no
se resuelven reintentando.

LangChain 1.x expone excepciones estandarizadas por tipo
(`ModelRateLimitError`, `ModelTimeoutError`, `ModelConnectionError`,
`ModelAPIError`), consistentes entre proveedores.

### 6.2 Fallback automático entre proveedores

`build_resilient_chain()` compone una cadena por cada proveedor disponible
en `PROVIDER_CHAIN` (variable de entorno) y las encadena con
`.with_fallbacks()`: si el proveedor principal agota sus reintentos, la
solicitud pasa automáticamente al siguiente.

### 6.3 Detección de `finish_reason`

`with_structured_output(..., include_raw=True)` expone el mensaje crudo
del modelo además del objeto parseado. Antes de dar una respuesta por
válida, se verifica si el modelo cortó la generación por límite de tokens
(el nombre del campo varía por proveedor: `finish_reason` en OpenAI/Gemini,
`stop_reason` en Anthropic, `done_reason` en Ollama). Si la respuesta no
cumple el esquema, el error se relanza intencionalmente para que
`.with_retry()` lo capture.

### 6.4 Observabilidad

El mecanismo de reintento de LangChain no expone un *hook* nativo para
loguear cada intento. Para resolverlo, la verificación post-respuesta
(`_revisar_resultado_estructurado`) se ejecuta dentro de la cadena
reintentable, lo que permite loguear (vía `loguru`) cada intento fallido,
cada proveedor probado, y el motivo específico cuando un proveedor agota
sus reintentos —incluyendo el caso en que `.with_fallbacks()` solo expone
el error del primer proveedor al usuario final, ocultando fallos
posteriores en la cadena de fallback.

## 7. Concurrencia: `RunnableParallel`

Como extensión sobre lo requerido por la consigna, `build_parallel_chain()`
ejecuta dos tareas en paralelo sobre el mismo texto de entrada, usando
`RunnableParallel`:

- Extracción de entidades (con retry y fallback completos)
- Resumen ejecutivo en lenguaje simple

**Consideración de diseño:** `RunnableParallel` utiliza `asyncio.gather()`
internamente, que por comportamiento por defecto cancela las tareas
restantes si una de ellas falla. Por este motivo, la cadena de resumen
ejecutivo también está envuelta en `.with_retry()`: sin esa protección, una
falla transitoria en esa rama podría cancelar la extracción de entidades
aun cuando esta se hubiera recuperado exitosamente un instante después.

## 8. Soporte multi-proveedor

La consigna no exige un proveedor específico. Se optó por un diseño
configurable —reutilizando el mismo criterio de la Pre-Entrega 1— que
soporta OpenAI, Anthropic, Google Gemini y Ollama (modelo local) de forma
intercambiable, definidos por la variable `PROVIDER_CHAIN` en el `.env`
(orden de proveedor principal y fallbacks).

## 9. Cumplimiento de la consigna

| Requisito | Estado |
|---|---|
| Esquema Pydantic (`schemas.py`) con `tecnologias`, `nivel_de_criticidad`, `resumen_tecnico` | ✅ |
| Cadena LCEL (`prompt \| model.with_structured_output(...)`) | ✅ |
| Resiliencia con `.with_retry()` ante JSON mal formado o incompleto | ✅ |
| Función asíncrona `process_text(text: str)` con `.ainvoke()` y logging | ✅ |
| Mini-script de prueba asíncrono (`test_manual.py`) | ✅ |
| Prompt modular vía `ChatPromptTemplate` (sin f-strings hardcodeadas) | ✅ |
| Detección de `finish_reason` antes de validar la respuesta | ✅ |
| Configuración por variables de entorno | ✅ |
| Ejemplo de salida documentado (sin informe externo) | ✅ |

### Extensiones no requeridas por la consigna

- Fallback automático entre múltiples proveedores (`.with_fallbacks()`)
- Soporte para Google Gemini y Ollama como proveedores adicionales
- Ejecución en paralelo de tareas independientes (`RunnableParallel`)
- Observabilidad extendida sobre el proceso de reintento y fallback