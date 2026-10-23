# Running the project

Everything runs locally: no cloud account and no paid API. This guide rebuilds the project
from a fresh clone. For what the project does and how it performs, see [README.md](README.md).

## What you need

| | Requirement | Notes |
|---|---|---|
| OS | Linux (tested on Ubuntu) | macOS should work; on Windows use WSL2 |
| Python | 3.11 or newer | |
| GPU | NVIDIA, 12 GB+ memory recommended | tested on an RTX A4000 (16 GB); CPU-only works but is very slow |
| Ollama | any recent version | runs the local models |
| Docker | with Docker Compose | runs Neo4j |
| Disk | about 15 GB | ~9 GB LLM, ~1 GB Docker image, ~200 MB data |
| Time | about 1–1.5 hours for a first build | most of it is the one-off LLM extraction |

## 1. Install the tools (once)

**Ollama** (from ollama.com), then the two models:
```bash
ollama pull qwen3:14b          # answering, routing, query writing, extraction (~9 GB)
ollama pull nomic-embed-text   # embeddings for search (~300 MB)
ollama ps                      # while a model is loaded: PROCESSOR should say 100% GPU
```

**Docker** (official Ubuntu instructions at docs.docker.com), then allow your user to run it:
```bash
sudo usermod -aG docker $USER   # then log out and back in
docker run hello-world
```

## 2. Get the code and set up Python

```bash
git clone https://github.com/<your-username>/graphrag-enterprise-copilot.git
cd graphrag-enterprise-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions used for the reported results
```
`requirements.txt` holds the same packages with minimum versions, if the lock file does not
install on your system.

## 3. Configure

```bash
cp .env.example .env
```
Edit `.env`:
- `SEC_USER_AGENT`: your name and a real email, e.g. `"Jane Doe jane@example.com"`.
  The SEC requires this for downloads and blocks requests without it.
- `NEO4J_PASSWORD`: choose one (at least 8 characters).

Load the settings. **Repeat this in every new terminal:**
```bash
set -a && source .env && set +a
```

## 4. Start Neo4j

```bash
docker compose up -d        # browser at http://localhost:7474, user neo4j
```
The container restarts automatically with Docker, and the graph is kept in a Docker volume.

## 5. Check the setup

```bash
python scripts/check_setup.py
```
Every line should say `[ OK ]` except the data files, which are built next.

## 6. Build the data

All at once (about 1 hour, mostly step 7):
```bash
bash scripts/build_all.sh
```
Or step by step:

| Step | Command | Time | Output |
|---|---|---|---|
| Download 24 annual reports, subsidiary lists, XBRL facts | `python ingest/sec_download.py` | 3 min | `data/raw/` |
| Financial database | `python ingest/build_sql.py` | seconds | `data/finance.db` |
| Parse reports into sections | `python ingest/parse_10k.py` | 1 min | `data/sections/` |
| Chunk and embed into FAISS | `python ingest/build_vector_index.py` | 1 min | `data/index/` |
| Subsidiaries from Exhibit 21 | `python ingest/parse_subsidiaries.py` | seconds | `data/graph/subsidiaries.csv` |
| LLM extraction: competitors, products, segments, partners, risk topics | `python ingest/extract_entities.py` | ~45 min | `data/graph/*.csv` |
| Load the knowledge graph | `python graph/load_graph.py --reset` | seconds | Neo4j |

Notes:
- **The SEC data changes over time.** A later download can include newer annual reports,
  so counts and some answers will differ from the README.
- **The extraction resumes** from `data/graph/llm_cache.jsonl` if interrupted. Try it on a
  sample first with `python ingest/extract_entities.py --limit 40`.
- **Do not rebuild the vector index after the extraction** without re-running the
  extraction: the graph cites passages by chunk id, and re-chunking changes the ids.

## 7. Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-server-header
```
- Web page: http://localhost:8000
- API documentation: http://localhost:8000/docs
- Status of every dependency: http://localhost:8000/health

From the command line, without the server:
```bash
python agent/graph.py "Which companies name AMD as a competitor?"
python agent/graph.py -v "Why did Intel's revenue decline?"     # -v shows queries and context
```

Optional security settings (access key, rate limit, allowed origins) are listed in
`.env.example` and explained in the README.

## 8. Reproduce the evaluation (optional)

```bash
python eval/build_testset.py --apply-review   # uses the reviewed questions in eval/vector_review.csv
python eval/run_eval.py                        # agent and baseline answer 48 questions (~25 min)
python eval/score.py                           # exact checks (seconds)
python eval/score.py --ragas --smoke           # check the RAGAS judge on 2 answers
python eval/score.py --ragas                   # RAGAS for all answers (~1-2 hours, resumable)
python eval/show_failures.py agent             # questions the agent got wrong
```
The committed results are in `eval/results/` (version 2) and `eval/v1/` (version 1).

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `NEO4J_PASSWORD is not set` | The terminal has not loaded `.env`: `set -a && source .env && set +a` |
| `/health` shows `neo4j: unreachable` | The container is not running: `docker compose up -d`, wait 20 seconds |
| `permission denied ... docker.sock` | Your user is not in the docker group yet: log out and back in after `usermod` |
| SEC download returns 403 | `SEC_USER_AGENT` is missing or has no email address |
| `ModuleNotFoundError: langchain_community.chat_models.vertexai` | `langchain-community` 0.4 is installed; RAGAS 0.4.3 needs `pip install "langchain-community<0.4"` |
| Slow answers, `ollama ps` shows CPU | The model does not fit in GPU memory: close other GPU programs, or use a smaller model with `export LLM_MODEL=qwen3:8b` |
| `Expecting value` / `Unterminated string` in extraction | The model sometimes emits whitespace until the length limit; such calls are retried automatically. Re-run the script; finished passages are cached |
| The web page shows "No answer: ... busy" | One question is answered at a time; wait for the current one |
| Search answers seem off after rebuilding data | The vector index and the graph must come from the same build; see the note in step 6 |

## Project layout

```
config.py              settings (paths, models, chunk sizes), read from environment variables
ingest/                SEC download, financial database, 10-K parsing, vector index, extraction
graph/load_graph.py    loads the knowledge graph into Neo4j
retrieval/, tools/     vector search and the SQL, Cypher and vector tools with their safeguards
agent/                 the LangGraph agent: route, run tools, answer, check citations
api/                   FastAPI service, security controls, web page
eval/                  test set, evaluation runs, scoring, RAGAS judge adapter, results
scripts/               setup checker and one-command data build
```
