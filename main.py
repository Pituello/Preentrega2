"""
Mini-script de prueba asíncrono: Pide un texto por
consola y corre la cadena completa: extracción de entidades + resumen
ejecutivo en paralelo con RunnableParallel.
"""
import asyncio

from dotenv import load_dotenv
from pydantic import ValidationError

from pipeline.chain import build_parallel_chain

load_dotenv()


async def main():
    texto = input("Pegá el texto a analizar: ")

    cadena = build_parallel_chain()

    try:
        resultado = await cadena.ainvoke({"texto": texto})
    except ValidationError as e:
        print(f"\n❌ el modelo no logró devolver algo que cumpla el esquema, ni con reintentos: {e}")
        return
    except Exception as e:
        print(f"\n❌ falló por otra razón (API key, conexión, etc.): {e}")
        return

    entidades = resultado["entidades"]
    print("\n--- entidades técnicas ---")
    print("Tecnologías:", entidades.tecnologias)
    print("Criticidad:", entidades.nivel_de_criticidad)
    print("Resumen técnico:", entidades.resumen_tecnico)

    print("\n--- resumen ejecutivo (calculado en paralelo con lo de arriba) ---")
    print(resultado["resumen_ejecutivo"])


if __name__ == "__main__":
    asyncio.run(main())
