# insight-audio

Python FastAPI microservice for audio analysis — powers the [Insight](https://github.com/Yedidya10/insight) music analytics platform with deep audio feature extraction.

## Architecture

```
insight-audio/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py             # Environment-based configuration
│   ├── api/
│   │   ├── router.py         # API route configuration
│   │   └── endpoints/
│   │       ├── health.py     # Health & readiness checks
│   │       └── analysis.py   # Analysis request endpoints
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response schemas
│   ├── services/
│   │   ├── audio_acquirer.py    # YouTube audio download (yt-dlp)
│   │   ├── librosa_extractor.py # Tier 1: librosa feature extraction
│   │   ├── panns_classifier.py  # Tier 2: PANNs Cnn14 classification
│   │   ├── clap_embedder.py     # Tier 3: CLAP semantic embeddings
│   │   └── supabase_client.py   # Database client
│   └── workers/
│       └── queue_worker.py   # Background queue processor
├── tests/                    # pytest test suite
├── Dockerfile               # Production container
├── docker-compose.yml       # Local development
└── pyproject.toml           # Project config & dependencies
```

## Model Stack

All models use **permissive open-source licenses** (MIT, Apache-2.0, ISC):

| Tier | Model | Purpose | License |
|------|-------|---------|---------|
| 1 | **librosa** | Tempo, key, spectral features, MFCCs, chroma | ISC |
| 1 | **PANNs Cnn14** | 527 AudioSet classes (genre, mood, instruments) | MIT |
| 2 | **CLAP** | 512-dim audio embeddings, text-audio similarity | Apache-2.0 |
| 3 | **CLAP zero-shot** | Natural language audio search | Apache-2.0 |

## Setup

### Prerequisites

- Python 3.11+
- ffmpeg (`apt install ffmpeg` or `brew install ffmpeg`)
- yt-dlp (`pip install yt-dlp`)

### Local Development

```bash
# Clone
git clone https://github.com/Yedidya10/insight-audio.git
cd insight-audio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# Run
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
# Build and run
docker compose up --build

# Or build manually
docker build -t insight-audio .
docker run -p 8000:8000 --env-file .env insight-audio
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness check (verifies Supabase connection) |
| `POST` | `/api/v1/analyze` | Queue a track for analysis |
| `GET` | `/api/v1/status/{track_id}` | Get analysis status |

### Example: Queue Analysis

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"track_id": "abc123", "track_name": "Bohemian Rhapsody", "artist_name": "Queen"}'
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=html
```

## Communication with Insight App

This service communicates with the main Insight app via a shared Supabase database:

1. **Insight app** inserts jobs into `audio_analysis_queue`
2. **insight-audio** polls the queue, downloads audio, runs analysis
3. **Results** are written to `track_audio_analysis` and `track_audio_embeddings`
4. **Webhook** notification sent to Insight app on completion

## License

MIT
