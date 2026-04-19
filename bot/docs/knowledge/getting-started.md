# Conectarse - Getting Started

El acceso es vía LiteLLM con una API compatible con OpenAI. Funciona con cualquier herramienta que acepte un `base URL` + `API key`: Cursor, Cline, Continue, Aider, Open Code, Open WebUI o cualquier SDK compatible con OpenAI.

## Paso 1: Obtener tu API Key

Debes estar dentro de la comunidad NaN. Si ya estás suscrito, solicita tu API Key al Staff. La key es personal e intransferible.

El soporte en Discord es solo para temas técnicos y gestión de keys.

## Paso 2: Configurar tu herramienta

Usa estos valores en tu IDE o herramienta:

- **base URL:** `https://api.nan.builders/v1`
- **API Key:** `tu-api-key-personal`
- **Model:** `qwen3.6`

### Ejemplo: config en OpenAI-compatible

```json
{
  "provider": {
    "openai": {
      "npm": "@ai-sdk/openai",
      "name": "NaN",
      "apiKey": "tu-api-key-personal",
      "baseURL": "https://api.nan.builders/v1",
      "model": "qwen3.6"
    }
  }
}
```

## Configuración por herramienta

### OpenCode

Crea o edita `opencode.json` en tu proyecto:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NaN",
      "options": {
        "baseURL": "https://api.nan.builders/v1",
        "apiKey": "sk-tu-key-aqui"
      },
      "models": {
        "qwen3.6": {
          "name": "Qwen 3.6",
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          }
        }
      }
    }
  }
}
```

### Cursor

1. Ve a **Settings → OpenAI API**
2. Base URL: `https://api.nan.builders/v1`
3. API Key: tu key personal
4. Model: `qwen3.6`

### Cline / Continue / Aider

Configura las variables de entorno:

```bash
export OPENAI_BASE_URL="https://api.nan.builders/v1"
export OPENAI_API_KEY="sk-tu-key-aqui"
```

### Open WebUI

En la configuración del provider:
- API Base URL: `https://api.nan.builders/v1`
- API Key: tu key
- Model: `qwen3.6`
