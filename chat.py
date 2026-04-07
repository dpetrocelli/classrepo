#!/usr/bin/env python3
"""Chat con LLM via OpenAI SDK. MISMO archivo en todas las demos."""

import os
import time

from openai import OpenAI

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL_NAME", "tinyllama:latest")

try:
    client = OpenAI(base_url=f"{INFERENCE_URL}/v1", api_key=os.getenv("API_KEY", "ollama"))
    print ("ANDA CHEEE")
except: 
    print ("NO ANDA")

PREGUNTA = "Explica en 2 oraciones que es un transformer en el mundo de inteligencia artificial."
print(f"Endpoint:  {INFERENCE_URL}")
print(f"Modelo:    {MODEL}")
print(f"Pregunta:  {PREGUNTA}")
print("-" * 50)

start = time.time()
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Responde de forma concisa en espanol."},
        {"role": "user", "content": PREGUNTA},
    ],
    temperature=0.7,
    max_tokens=256,
)
elapsed = time.time() - start
text = response.choices[0].message.content
tokens = response.usage.completion_tokens if response.usage else len(text.split())
tok_s = tokens / elapsed if elapsed > 0 else 0

print(f"\nRespuesta: {text}")
print(f"\n{'=' * 50}")
print(f"Tiempo:    {elapsed:.1f} seg")
print(f"Tokens:    ~{tokens}")
print(f"Velocidad: ~{tok_s:.0f} tok/s")
print(f"Endpoint:  {INFERENCE_URL}")
print(f"{'=' * 50}")