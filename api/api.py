#!/usr/bin/env python3
"""API de chat con LLM usando OpenAI SDK compatible (ej: Ollama)."""

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# Configuración
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL_NAME", "tinyllama:latest")
API_KEY = os.getenv("API_KEY", "ollama")

# Cliente OpenAI-compatible
client = OpenAI(base_url=f"{INFERENCE_URL}/v1", api_key=API_KEY)

# App FastAPI
app = FastAPI(title="LLM Chat API")

# Request schema
class ChatRequest(BaseModel):
    pregunta: str
    temperature: float = 0.7
    max_tokens: int = 256


# Healthcheck
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL,
        "endpoint": INFERENCE_URL
    }


# Endpoint principal
@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Responde de forma concisa en español."},
            {"role": "user", "content": req.pregunta},
        ],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    elapsed = time.time() - start
    text = response.choices[0].message.content
    tokens = response.usage.completion_tokens if response.usage else len(text.split())
    tok_s = tokens / elapsed if elapsed > 0 else 0

    return {
        "respuesta": text,
        "metrics": {
            "tiempo_seg": round(elapsed, 2),
            "tokens": tokens,
            "tokens_por_seg": round(tok_s, 2),
        },
        "model": MODEL,
        "endpoint": INFERENCE_URL,
    }