# Embedding Service

High-performance SentenceTransformer embedding service with batching and caching, built with FastAPI.

## Features

- **Dynamic Batching**: Automatically batches incoming requests for efficient GPU/CPU utilization
- **Smart Caching**: LRU cache with TTL to avoid redundant computations for repeated queries
- **Configurable**: All settings can be configured via environment variables
- **Docker Ready**: Includes Dockerfile for CPU and GPU deployments
- **Production Ready**: Health checks, metrics, and proper error handling
- **Standalone Encoder**: `CachedSentenceTransformer` class for direct use without HTTP

## Project Structure

```
├── main.py              # FastAPI application entry point
├── service.py           # Async embedding service with batching
├── cache.py             # LRU cache implementation
├── cached_encoder.py    # Standalone cached encoder (no HTTP)
├── config.py            # Pydantic settings configuration
├── models.py            # Request/response models
├── requirements.txt     # Python dependencies
├── Dockerfile           # CPU Docker image
├── Dockerfile.gpu       # GPU Docker image
├── docker-compose.yml   # Docker Compose configuration
└── .env.example         # Example environment variables
```

## Quick Start

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Open API docs at http://localhost:8000/docs

### Docker

```bash
# Build and run with Docker Compose
docker compose up --build

# Or build and run manually
docker build -t embedding-service .
docker run -p 8000:8000 embedding-service
```

For GPU support:
```bash
docker compose --profile gpu up embedding-service-gpu
```

## API Endpoints

### POST /embed
Generate embeddings for one or more texts.

**Request:**
```json
{
  "text": "Hello, world!"
}
// Or for multiple texts:
{
  "texts": ["Hello", "World", "How are you?"]
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "dimension": 384,
  "num_texts": 1
}
```

### GET /health
Health check endpoint.

### GET /stats
Get service statistics including cache hit rates.

### GET /config
Get current service configuration.

### DELETE /cache
Clear the embedding cache.

## Configuration

All settings can be configured via environment variables with the `EMBEDDING_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `EMBEDDING_DEVICE` | `cpu` | Device to run model on (cpu/cuda/mps) |
| `EMBEDDING_BATCH_SIZE` | `32` | Maximum batch size |
| `EMBEDDING_BATCH_TIMEOUT` | `0.1` | Max time to wait for batch to fill (seconds) |
| `EMBEDDING_CACHE_ENABLED` | `true` | Enable/disable caching |
| `EMBEDDING_CACHE_MAX_SIZE` | `10000` | Maximum cached embeddings |
| `EMBEDDING_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `EMBEDDING_HOST` | `0.0.0.0` | Server host |
| `EMBEDDING_PORT` | `8000` | Server port |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    │
│  │ Request  │───▶│ Cache Check │───▶│ Batch Queue      │    │
│  └──────────┘    └─────────────┘    └────────┬─────────┘    │
│                        │                     │              │
│                   Cache Hit                  │              │
│                        │                     ▼              │
│                        │            ┌──────────────────┐    │
│                        │            │ Batch Processor  │    │
│                        │            │ (waits for batch │    │
│                        │            │  size or timeout)│    │
│                        │            └────────┬─────────┘    │
│                        │                     │              │
│                        │                     ▼              │
│                        │            ┌──────────────────┐    │
│                        │            │ SentenceTransfor-│    │
│                        │            │ mer.encode()     │    │
│                        │            └────────┬─────────┘    │
│                        │                     │              │
│                        ▼                     ▼              │
│                  ┌─────────────────────────────────────┐    │
│                  │           Response                  │    │
│                  └─────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Popular Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| `paraphrase-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ |

## Example Usage

### Standalone Encoder (No HTTP)

`CachedSentenceTransformer` is a drop-in replacement for `SentenceTransformer` with caching:

```python
from cached_encoder import CachedSentenceTransformer

# Same interface as SentenceTransformer
encoder = CachedSentenceTransformer(
    "thenlper/gte-large",
    cache_max_size=10000,
    cache_ttl=3600,  # optional TTL
)

# Single text
embedding = encoder.encode("Hello, world!")

# Multiple texts (efficient - only encodes uncached)
embeddings = encoder.encode(["Hello", "World", "Hello"])  # "Hello" cached

# All SentenceTransformer parameters work
embeddings = encoder.encode(texts, normalize_embeddings=True, batch_size=64)

# Check cache stats
print(encoder.get_cache_stats())
# {'cache_hits': 1, 'cache_misses': 2, 'hit_rate': 0.33, ...}
```

### HTTP Client

```python
import requests

# Single text
response = requests.post(
    "http://localhost:8000/embed",
    json={"text": "Hello, world!"}
)
embedding = response.json()["embeddings"][0]

# Multiple texts
response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": ["Hello", "World", "How are you?"]}
)
embeddings = response.json()["embeddings"]
```

### cURL

```bash
# Single text
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Multiple texts
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'

# Check stats
curl "http://localhost:8000/stats"
```

## Performance Tips

1. **Use batch requests**: Send multiple texts in a single request for better throughput
2. **Tune batch settings**: Adjust `BATCH_SIZE` and `BATCH_TIMEOUT` based on your load pattern
3. **Enable caching**: Keep caching enabled if you have repeated queries
4. **Use GPU**: Set `EMBEDDING_DEVICE=cuda` for GPU acceleration
5. **Choose the right model**: Smaller models (MiniLM) are faster, larger models (mpnet) are more accurate

## License

MIT
