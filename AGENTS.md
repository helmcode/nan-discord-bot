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
- **discord.py** >=2.3.2 (pin en `pyproject.toml`) — framework de Discord
- **OpenAI SDK** (`AsyncOpenAI`) — cliente para LiteLLM API (chat + embeddings)
- **aiohttp** — usado por `bot/metrics.py` para consultar el LiteLLM proxy
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
LITELLM_API_KEY=                  # API key de LiteLLM (chat + embeddings)
LITELLM_PROXY_URL=http://localhost:4000   # Base URL del proxy LiteLLM (métricas)
LITELLM_ADMIN_KEY=                # Admin key del proxy; sin esto se deshabilitan /metrics y el reporte diario
EMBEDDING_MODEL=qwen3-embedding
EMBEDDING_DIM=4096
TOP_K=5
ALLOWED_CHANNELS=                 # Channel IDs donde el bot responde (múltiples separados por coma)
STATUS_CHANNEL_ID=                # Canal donde se publica el reporte diario de métricas
METRICS_SEND_HOUR=9               # Hora UTC (0-23) del reporte diario
SLACK_WEBHOOK_URL=                # Incoming Webhook de Slack; vacío = notificaciones deshabilitadas
SLACK_BOT_TOKEN=                  # Bot token (xoxb-...) con scope chat:write; habilita el espejo de hilos
SLACK_CHANNEL_ID=                 # Canal destino cuando se usa SLACK_BOT_TOKEN
SUPPORT_CHANNEL_IDS=              # Channel IDs cuyos hilos nuevos se avisan en Slack (coma-separados)
SLACK_HTTP_TIMEOUT=10             # Timeout (s) de las peticiones a Slack
```

**Comportamiento**: El bot SOLO responde cuando lo mencionan (`@NaN Builders`). No responde automáticamente en canales de soporte.

**Múltiples canales**: `ALLOWED_CHANNELS` acepta múltiples IDs separados por coma, ej: `123456789,987654321,111222333`

### Notificaciones a Slack

Cuando se crea un hilo en un canal listado en `SUPPORT_CHANNEL_IDS` (ej. `#support`), el bot
publica un mensaje en Slack vía Incoming Webhook con el título del hilo, el autor, el canal
y un preview del mensaje inicial. Funciona con canales de texto y con canales de foro.

Requiere `SLACK_WEBHOOK_URL`: en Slack, crear una app → **Incoming Webhooks** → activar →
**Add New Webhook to Workspace** y elegir el canal de destino. El canal se fija ahí, no en el `.env`.

Sin Slack configurado o sin `SUPPORT_CHANNEL_IDS` la feature queda inactiva y el bot funciona igual.

**Dos modos, y el bot prefiere el segundo:**

| | Webhook (`SLACK_WEBHOOK_URL`) | Bot token (`SLACK_BOT_TOKEN` + `SLACK_CHANNEL_ID`) |
|---|---|---|
| Avisa de hilos nuevos | Sí | Sí |
| Espeja los mensajes del hilo | **No** | Sí |
| Configuración | Una URL | App con scope `chat:write` e invitada al canal |

El webhook **no puede** espejar hilos: su respuesta no incluye el `ts` del mensaje,
y sin ese `ts` no hay forma de colgar respuestas de él. La Web API sí lo devuelve.

El mapeo `hilo de Discord -> ts de Slack` se guarda en `vector_db/slack_threads.db`
(dentro del volumen, así sobrevive a los despliegues) y se purga a los 30 días.

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
