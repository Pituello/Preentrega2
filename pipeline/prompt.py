"""
El prompt como ChatPromptTemplate, no como f-string suelta.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_EXTRACCION = """Sos un analista técnico. Tu trabajo es leer un texto libre
(puede ser un ticket, un mensaje de Slack, un README, un log) y extraer:

- las tecnologías, lenguajes o herramientas mencionadas
- qué tan crítico es el tema para el negocio (baja, media o alta)
- un resumen técnico corto (1 o 2 oraciones)

Si el texto es ambiguo o no menciona ninguna tecnología explícita, hacé tu
mejor inferencia razonable en vez de dejar el campo vacío o inventar datos
que no están en el texto.
"""

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_EXTRACCION),
        ("human", "{texto}"),
    ]
)

SYSTEM_RESUMEN = "Resumí el siguiente texto en una sola oración, en lenguaje simple, para alguien no técnico."

resumen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_RESUMEN),
        ("human", "{texto}"),
    ]
)
