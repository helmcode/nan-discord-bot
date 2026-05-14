# Bienvenido a NaN

Esta documentación explica cómo conectar tus herramientas a las GPUs de NaN. El servidor corre modelos open-source con una API compatible con OpenAI. Si algo acepta un `base URL` + `API key`, funciona con NaN.

## Para obtener tu API Key

Debes estar dentro de la comunidad NaN y solicitar tu API Key al Staff a través del canal `#support` en Discord. La key es personal e intransferible.

## Rate Limits

No hay límites de uso de tokens. Las únicas restricciones son para evitar DDoS o que un usuario acapare la GPU.

- **Requests / min:** 100 rpm
- **Paralelo máximo:** 5 concurrentes

## Hardware

- **GPU:** NVIDIA RTX PRO 6000 Blackwell
- **VRAM:** 96 GB GDDR7 ECC
- **RAM:** 256 GB DDR5 ECC
- **CPU:** 48 threads · Xeon Gold 5412U
- **Inference:** vLLM → LiteLLM
- **Embedding:** HuggingFace TEI → LiteLLM
- **TTS:** kokoro-fastapi → LiteLLM
- **STT:** speaches → LiteLLM
