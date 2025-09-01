import os
import time
import glob
import queue
import threading
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf

import tkinter as tk
from tkinter import ttk, messagebox

from dotenv import load_dotenv
from openai import OpenAI

import google.generativeai as genai

# -----------------------------
# Configurações de áudio
# -----------------------------
SAMPLERATE = 16000   # 16 kHz
CHANNELS = 1

# -----------------------------
# Ambiente / chaves
# -----------------------------
client = OpenAI(api_key='sk-ChaveAPI-OPENAI')

GEMINI_API_KEY = 'GeminiKey'



def transcrever_via_api(audio_path: str,
                        modelo: str = "gpt-4o-transcribe",
                        prompt: str | None = "Contexto: consulta médica, português do Brasil."):
    """
    Envia o arquivo de áudio (WAV/MP3/...) para a API de STT.
    """
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=modelo,
            file=f,
            prompt=prompt
        )
    return resp.text if hasattr(resp, "text") else str(resp)


def connect_to_gemini_from_txt(api_key: str | None, txt_path: str,
                               model: str = "gemini-1.5-flash") -> str:
    """
    Lê um arquivo .txt (transcrição) e envia ao Gemini com um prompt pedindo RESUMO.
    Retorna apenas o texto do resumo.
    """
    try:
        if not txt_path.endswith(".txt"):
            return "Erro: forneça um arquivo .txt válido."

        if not os.path.exists(txt_path):
            return f"Erro: arquivo não encontrado: {txt_path}"

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return "Erro: o arquivo de transcrição está vazio."

        # Configuração da API (caso não tenha sido configurada acima)
        if api_key and not os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=api_key)

        # Prompt de resumo (em PT-BR)
        resumo_prompt = (
            "Resuma de forma objetiva e estruturada o conteúdo abaixo em português do Brasil.\n"
            "- Foque em: queixas principais, sinais/sintomas, medicamentos citados, condutas e próximos passos.\n"
            "- Use no máximo 8 tópicos (bullets) curtos.\n"
            "- Não invente informações.\n\n"
            "Transcrição:\n"
            f"\"\"\"\n{text}\n\"\"\"\n"
        )

        model_g = genai.GenerativeModel(model)
        response = model_g.generate_content(resumo_prompt)
        return response.text

    except Exception as e:
        return f"Erro ao conectar à API do Gemini: {e}"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transcrição Cloud — Start/Stop (ttk)")

        # Estado de captura
        self.is_recording = False
        self.stream = None
        self.buffer = []            # lista de np.ndarray (chunks)
        self.buffer_queue = queue.Queue()  # opcional se quiser processar em tempo-real

        # Arquivo temporário para WAV final e último TXT
        self.tmp_wav = None
        self.last_txt_path = None   # guarda o último TXT salvo para usar no Gemini

        self._build_ui()

    # -----------------------------
    # UI
    # -----------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Linha de botões
        line = ttk.Frame(main)
        line.pack(fill=tk.X)

        self.btn_toggle = ttk.Button(line, text="Iniciar", command=self.toggle_recording)
        self.btn_toggle.pack(side=tk.LEFT)

        self.btn_clear = ttk.Button(line, text="Limpar", command=self.clear_text, state=tk.NORMAL)
        self.btn_clear.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_resumo = ttk.Button(line, text="Resumo (Gemini)", command=self._resumir_gemini)
        self.btn_resumo.pack(side=tk.LEFT, padx=(8, 0))

        # Status
        self.lbl_status = ttk.Label(main, text="Status: parado")
        self.lbl_status.pack(anchor="w", pady=(10, 6))

        # Transcrição
        ttk.Label(main, text="Transcrição:").pack(anchor="w")
        self.txt = tk.Text(main, height=12, width=90, wrap="word")
        self.txt.pack(fill=tk.BOTH, expand=True)

        # Resumo Gemini
        ttk.Label(main, text="Resumo (Gemini):").pack(anchor="w", pady=(8, 2))
        self.txt_resumo = tk.Text(main, height=10, width=90, wrap="word")
        self.txt_resumo.pack(fill=tk.BOTH, expand=True)

        # Tempo da última transcrição
        self.lbl_tempo = ttk.Label(main, text="Tempo da última transcrição: -")
        self.lbl_tempo.pack(anchor="w", pady=(6, 0))

    # -----------------------------
    # Captura — Start/Stop
    # -----------------------------
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # limpa buffer e prepara stream
        self.buffer.clear()
        self.tmp_wav = None

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Áudio", f"Erro ao iniciar captura: {e}")
            return

        self.is_recording = True
        self.btn_toggle.config(text="Parar")
        self.lbl_status.config(text="Status: gravando…")

    def stop_recording(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.is_recording = False
        self.btn_toggle.config(text="Iniciar")
        self.lbl_status.config(text="Status: processando…")

        # Processa em thread para não travar a UI
        threading.Thread(target=self._finalizar_e_transcrever, daemon=True).start()

    # -----------------------------
    # Callbacks e pipeline
    # -----------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # imprime warnings no console, mas não interrompe
            print("Audio status:", status)
        # copia chunk para o buffer
        self.buffer.append(indata.copy())

    def _finalizar_e_transcrever(self):
        """
        Concatena os chunks, salva WAV, chama a API e atualiza a UI.
        """
        t0 = time.time()
        try:
            if not self.buffer:
                self._ui_set_status("Status: nenhum áudio capturado.")
                return

            # concatena tudo
            audio = np.concatenate(self.buffer, axis=0)  # shape: (n, 1)
            audio = np.squeeze(audio)  # (n,)

            # salva WAV temporário (16-bit PCM)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                self.tmp_wav = tmp.name

            # normaliza para int16 e salva
            audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            sf.write(self.tmp_wav, audio_i16, SAMPLERATE, subtype="PCM_16")

            # chama API
            texto = transcrever_via_api(self.tmp_wav)

            # salva .txt com timestamp
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            txt_name = f"transcricao_{ts}.txt"
            with open(txt_name, "w", encoding="utf-8") as f:
                f.write(texto or "")

            # guarda último txt salvo
            self.last_txt_path = os.path.abspath(txt_name)

            dt = time.time() - t0
            self._ui_update_transcricao(texto or "")
            self._ui_set_tempo(dt)
            self._ui_set_status(f"Status: pronto — salvo em {txt_name}")

        except Exception as e:
            self._ui_update_transcricao(f"[ERRO] {e}")
            self._ui_set_status("Status: erro ao transcrever.")
        finally:
            # remove WAV temporário
            try:
                if self.tmp_wav and os.path.exists(self.tmp_wav):
                    os.remove(self.tmp_wav)
            except Exception:
                pass
            self.tmp_wav = None

    # -----------------------------
    # Resumo com Gemini
    # -----------------------------
    def _resumir_gemini(self):
        """
        Lê o .txt da transcrição (último salvo ou o mais recente do padrão transcricao_*.txt),
        envia ao Gemini com um prompt de resumo e mostra o resultado na UI.
        """
        try:
            # tenta usar o último caminho salvo; senão, procura o mais recente
            txt_path = self.last_txt_path
            if not txt_path or not os.path.exists(txt_path):
                candidatos = sorted(glob.glob("transcricao_*.txt"), key=os.path.getmtime, reverse=True)
                if candidatos:
                    txt_path = candidatos[0]

            if not txt_path:
                messagebox.showwarning("Resumo", "Nenhuma transcrição encontrada para resumir.")
                return

            # chama Gemini
            resumo = connect_to_gemini_from_txt(GEMINI_API_KEY, txt_path)
            if not resumo:
                resumo = "(Sem retorno do Gemini.)"

            # atualiza área de resumo
            self.root.after(0, lambda: (self.txt_resumo.delete("1.0", tk.END), self.txt_resumo.insert(tk.END, resumo)))

            self._ui_set_status(f"Resumo gerado a partir de: {os.path.basename(txt_path)}")

        except Exception as e:
            messagebox.showerror("Resumo (Gemini)", f"Erro ao gerar resumo: {e}")

    # -----------------------------
    # Helpers de UI (thread-safe)
    # -----------------------------
    def _ui_update_transcricao(self, texto: str):
        self.root.after(0, lambda: (self.txt.delete("1.0", tk.END), self.txt.insert(tk.END, texto + "\n")))

    def _ui_set_status(self, texto: str):
        self.root.after(0, lambda: self.lbl_status.config(text=texto))

    def _ui_set_tempo(self, seconds: float):
        self.root.after(0, lambda: self.lbl_tempo.config(text=f"Tempo da última transcrição: {seconds:.2f}s"))

    def clear_text(self):
        self.txt.delete("1.0", tk.END)
        self.txt_resumo.delete("1.0", tk.END)
        self.lbl_tempo.config(text="Tempo da última transcrição: -")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = App(root)
    root.mainloop()
