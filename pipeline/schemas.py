"""
El contrato de datos: lo que la cadena SIEMPRE tiene que devolver, no
importa qué tan raro venga el texto de entrada. Esto es lo que
with_structured_output() usa para obligar al modelo a responder en este
formato (vía tool calling / JSON mode del proveedor).
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EntityExtraction(BaseModel):
    """Extracción de entidades técnicas a partir de un texto libre."""

    tecnologias: list[str] = Field(
        ..., min_length=1, description="Tecnologías, lenguajes o herramientas mencionadas en el texto"
    )
    nivel_de_criticidad: Literal["baja", "media", "alta"] = Field(
        ..., description="Qué tan crítico es el tema para el negocio"
    )
    resumen_tecnico: str = Field(..., min_length=1, description="Resumen técnico corto, 1 o 2 oraciones")

    @field_validator("tecnologias")
    @classmethod
    def sin_tecnologias_vacias(cls, v):
        limpio = [t.strip() for t in v if t.strip()]
        if not limpio:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpio
