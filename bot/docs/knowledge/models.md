# Modelos del Server

Todos los modelos se acceden por la misma API OpenAI-compatible con el mismo `base URL`: `https://api.nan.builders/v1`

## qwen3.6 - Generación de texto y chat

El modelo principal de NaN.

- **Tipo:** MoE (35B total, 3B activos por token)
- **Cuantización:** FP8
- **Contexto:** 128K tokens
- **Speculative decoding:** MTP → ~2x throughput
- **Sampling:** temp=0.6, top_p=0.95
- **Reasoning:** reasoning_config={}

### Capacidades

- Tool calling (formato XML)
- Reasoning mode
- Multimodal (vision / imágenes)
- Generación streaming (SSE)

### Ejemplos de uso

#### curl

```bash
curl https://api.nan.builders/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key-personal" \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
    "max_tokens": 500
  }'
```

#### Python (openai)

```python
from openai import OpenAI

client = OpenAI(
  api_key="sk-tu-key-aqui",
  base_url="https://api.nan.builders/v1"
)

response = client.chat.completions.create(
  model="qwen3.6",
  messages=[{"role": "user", "content": "Escribe un hola mundo en Rust"}],
  max_tokens=500,
  stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

#### Node.js (openai)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-tu-key-aqui",
  baseURL: "https://api.nan.builders/v1",
});

const stream = await client.chat.completions.create({
  model: "qwen3.6",
  messages: [{ role: "user", content: "Escribe un hola mundo en Zig" }],
  max_tokens: 500,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
```

---

## qwen3-embedding - Embeddings vectoriales

- **Dimensión:** 4096
- **Precisión:** Float32 (CPU)
- **RPM:** 60
- **Batch size:** 32
- **MMTEB score:** 70.58 (top open-source)

Soporta 100+ idiomas incluyendo español y código.

### Casos de uso

- Similitud cross-lingual (ES↔EN: 0.915)
- Búsqueda semántica
- Clasificación de texto
- RAG / retrieval aumentado

#### curl

```bash
curl https://api.nan.builders/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key-personal" \
  -d '{
    "model": "qwen3-embedding",
    "input": ["Hola mundo", "Hello world"]
  }'
# → 4096-dimensional vectors per input
```

#### Python

```python
from openai import OpenAI

client = OpenAI(
  api_key="sk-tu-key-aqui",
  base_url="https://api.nan.builders/v1"
)

response = client.embeddings.create(
  model="qwen3-embedding",
  input=["Kubernetes pod scheduling", "Programación de pods Kubernetes"]
)

embeddings = [d.embedding for d in response.data]
print(len(embeddings[0]))  # 4096
```

#### OpenClaw config

```json
{
  "models": {
    "providers": {
      "nan": {
        "baseUrl": "https://api.nan.builders/v1",
        "apiKey": "sk-...",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-embedding",
            "name": "Qwen3 Embedding",
            "input": ["text"],
            "dimensions": 4096
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "nan/qwen3-embedding" }
    }
  }
}
```

Config para `~/.openclaw/openclaw.json`

---

## kokoro - Text-to-Speech

- **Latencia:** < 1s
- **Partes:** 82M
- **RPM:** 15
- **Voice packs:** 67 disponibles

### Voces disponibles

- `af_heart` — English (female)
- `ef_dora` — Spanish (female)
- `em_alex` — Spanish (male)
- 67 voice packs en total

#### curl

```bash
curl https://api.nan.builders/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key-personal" \
  -d '{
    "model": "kokoro",
    "input": "Bienvenido a NaN builders.",
    "voice": "ef_dora"
  }' \
  -o speech.mp3
```

#### Python

```python
from openai import OpenAI

client = OpenAI(
  api_key="sk-tu-key-aqui",
  base_url="https://api.nan.builders/v1"
)

response = client.audio.speech.create(
  model="kokoro",
  voice="ef_dora",
  input="Hola, bienvenido a NaN builders.",
  speed=1.0,
  response_format="mp3"
)

response.stream_to_file("output.mp3")
```

---

## whisper - Speech-to-Text

- **Tamaño:** ~3 GB (INT8)
- **WER ES:** ~3.2%
- **RPM:** 10
- **Idiomas:** 99+
- **Realtime:** ~1x en CPU

#### curl

```bash
# Transcribe audio file
curl https://api.nan.builders/v1/audio/transcriptions \
  -H "Authorization: Bearer tu-api-key-personal" \
  -F "model=whisper" \
  -F "file=@recording.mp3" \
  -F "language=es"

# → {"text":"Texto transcrito","language":"es","duration":5.2}
```

#### Python

```python
from openai import OpenAI

client = OpenAI(
  api_key="sk-tu-key-aqui",
  base_url="https://api.nan.builders/v1"
)

with open("grabacion.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper",
        file=f,
        language="es",
        response_format="verbose_json"
    )

print(result.text)       # Texto transcrito
print(result.language)   # "es"
print(result.duration)   # 5.2
```

---

## Rate Limits por API Key

- **Requests / min:** 100 rpm
- **Paralelo máximo:** 5 concurrentes
