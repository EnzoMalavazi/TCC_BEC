import os
import queue
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Configurações de áudio
# -----------------------------
SAMPLERATE = 16000   # 16 kHz é o padrão recomendado
CHANNELS = 1
WAV_PATH = "captura.wav"

# -----------------------------
# API OpenAI
# -----------------------------
load_dotenv()
client = OpenAI(api_key = "key")

# -----------------------------
# Captura de áudio (bloqueante)
# -----------------------------
def gravar_audio(duracao_segundos: float, samplerate=SAMPLERATE, channels=CHANNELS, out_path=WAV_PATH):
    print(f"Iniciando gravação por {duracao_segundos:.1f}s...")
    audio = sd.rec(int(duracao_segundos * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
    sd.wait()
    # Converte float32 [-1,1] em int16 ao salvar (compatibilidade ampla)
    audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    sf.write(out_path, audio_i16, samplerate, subtype="PCM_16")
    print(f"Áudio salvo em: {out_path}")
    return out_path

# -----------------------------
# Transcrição via nuvem
# -----------------------------
def transcrever_via_api(audio_path: str, modelo: str = "gpt-4o-transcribe", prompt: str | None = None):
    """
    Envia o arquivo de áudio para a API de STT.
    model: gpt-4o-transcribe ou gpt-4o-mini-transcribe (mais barato/rápido)
    prompt: opcional para guiar nomes próprios/estilo (curto)
    """
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=modelo,
            file=f,
            # language="pt",  # normalmente desnecessário; a detecção é automática
            prompt=prompt    # útil para nomes/regras de escrita
            # OBS: alguns modelos suportam "response_format='verbose_json'" se quiser timestamps
        )
    # O SDK retorna um objeto com .text na maioria dos formatos
    return resp.text if hasattr(resp, "text") else str(resp)

# -----------------------------
# Tradução do texto (opcional)
# -----------------------------
def traduzir_texto(texto: str, idioma_destino: str = "pt-BR", modelo_texto: str = "gpt-4.1-mini"):
    """
    Traduz o texto transcrito para o idioma desejado usando um modelo de chat.
    """
    msg = (
        f"Traduza fielmente para {idioma_destino}. "
        f"Mantenha termos técnicos e formatação natural. Texto:\n\n{texto}"
    )
    chat = client.chat.completions.create(
        model=modelo_texto,
        messages=[{"role": "user", "content": msg}],
        temperature=0
    )
    return chat.choices[0].message.content.strip()

# -----------------------------
# Exemplo de uso (CLI simples)
# -----------------------------
if __name__ == "__main__":
    # 1) Grave um trecho curto (ajuste a duração)
    caminho = gravar_audio(duracao_segundos=8.0)

    # 2) Transcreva na nuvem
    print("Transcrevendo na nuvem...")
    transcricao = transcrever_via_api(caminho, modelo="gpt-4o-transcribe", prompt="Contexto: consulta médica, nomes próprios em português.")
    print("\n=== TRANSCRIÇÃO (idioma original) ===")
    print(transcricao)
    
# 3) Salvar a transcrição em um arquivo TXT
with open("transcricao.txt", "w", encoding="utf-8") as f:
    f.write(transcricao)

print("\nTranscrição salva em 'transcricao.txt'")
