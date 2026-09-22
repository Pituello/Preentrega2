"""
Prueba de estrés: Necesita al menos un proveedor configurado en el .env 
(no se puede simular con un fake acá porque with_structured_output depende de que el
proveedor real soporte tool calling / JSON mode).
"""
import asyncio

from dotenv import load_dotenv
from pydantic import ValidationError

from pipeline.chain import process_text

load_dotenv()

# A propósito no menciona ninguna tecnología puntual, para ver cómo se
# las arregla el modelo.
TEXTO_AMBIGUO = """
Andaba todo mas o menos bien pero de golpe se rompio todo y nadie
entiende bien por que, capaz fue algo del deploy de ayer o un tema de la
base, hay que verlo antes de que se entere el cliente
"""


async def main():
    print("Probando con un texto ambiguo (sin tecnologías explícitas)...\n")
    print(f"Texto: {TEXTO_AMBIGUO.strip()}\n")

    try:
        resultado = await process_text(TEXTO_AMBIGUO)
    except ValidationError as e:
        print(f"❌ el modelo no logró devolver un objeto válido incluso con retries: {e}")
        return
    except Exception as e:
        print(f"❌ falló por otra razón (revisar API keys / conexión): {e}")
        return

    print("✅ el modelo se recuperó y devolvió un objeto válido:")
    print(f"  tecnologias: {resultado.tecnologias}")
    print(f"  nivel_de_criticidad: {resultado.nivel_de_criticidad}")
    print(f"  resumen_tecnico: {resultado.resumen_tecnico}")


if __name__ == "__main__":
    asyncio.run(main())
