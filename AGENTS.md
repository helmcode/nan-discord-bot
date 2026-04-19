# NaN Discord Bot

## Idioma

Todo el código debe escribirse en inglés aunque la conversación sea en español

## Flujo de Trabajo

**CRÍTICO**: La imagen Docker se construye en GH Actions y se sube a GHCR. El servidor SOLO hace `docker pull`.

1. **Editar código** localmente
2. **Testear en local**: `docker compose up --build` — verificar que el bot levanta
3. **Subir código** al repositorio (`git push origin main`)
4. **Monitorear GH Actions** — el workflow construye la imagen, la sube a GHCR, y despliega al servidor
5. **Validar en servidor**: `docker ps` y `docker compose logs`

## Stack Técnico

- **Python 3.11** con hatchling
- **discord.py** 2.7.1 — framework de Discord
- **OpenAI SDK** — cliente para LiteLLM API
- **Pydantic Settings** — configuración con `.env`
- **SQLite** — vector store con embeddings (cosine similarity)
- **Docker** — contenedorización
- **Docker Compose** — orquestación

## Configuración del Bot

### `.env` variables

```env
DISCORD_TOKEN=                    # Token del bot
DISCORD_GUILD_ID=                 # Server ID de Discord
LITELLM_BASE_URL=https://api.nan.builders/v1
LITELLM_API_KEY=                  # API key de LiteLLM
EMBEDDING_MODEL=qwen3-embedding
EMBEDDING_DIM=4096
TOP_K=5
ALLOWED_CHANNELS=                 # Channel IDs donde el bot responde (múltiples separados por coma)
```

**Comportamiento**: El bot SOLO responde cuando lo mencionan (`@NaN Builders`). No responde automáticamente en canales de soporte.

**Múltiples canales**: `ALLOWED_CHANNELS` acepta múltiples IDs separados por coma, ej: `123456789,987654321,111222333`

### Intents de Discord

**CRÍTICO**: En [Discord Developer Portal](https://discord.com/developers/applications) para la aplicación del bot, habilitar:
- [x] MESSAGE CONTENT INTENT
- [x] MEMBERS INTENT

Sin estos intents, el bot no puede conectar (`PrivilegedIntentsRequired`).

## Seguridad

- **NUNCA** committear `.env` o tokens
- `.env` está en `.gitignore`
- Secrets van en GitHub repository settings y en el servidor
- API keys de LiteLLM solo en variables de entorno del servidor
