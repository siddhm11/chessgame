# Deploy to Hugging Face Spaces

Result: a public web app at `https://huggingface.co/spaces/<you>/chess-vision-coach`
(HTTPS included, nothing to host). One command; ~5 minutes including the build.

> **Needs a paid Hugging Face plan.** Hugging Face's docs (Spaces Overview):
> *"Gradio and Docker Spaces run on compute and require a paid plan to create:
> PRO for personal accounts."* The Space then runs on free "CPU Basic"
> hardware. Without PRO, see **Other hosts** below: the Dockerfile is portable.
> `deploy_hf_space.py` checks your plan before doing anything.

## 1. Get a write token (once)

<https://huggingface.co/settings/tokens> → **Create new token** → type **Write** → copy it.

## 2. Deploy

In a terminal, inside the `chess-vision-coach` folder of an up-to-date clone
of this branch (`git pull` first):

```bash
pip install huggingface_hub

# Optional: see exactly what will be uploaded (25 files, ~12 MB). Uploads nothing.
python deploy_hf_space.py --space siddhm11/chess-vision-coach --dry-run

# Deploy. Creates the Space if needed; asks for the token (or set HF_TOKEN).
python deploy_hf_space.py --space siddhm11/chess-vision-coach
```

On Windows use `py` instead of `python` if `python` isn't found. Add
`--private` to make the Space visible only to you.

## 3. Wait for the build

Open the URL the script prints. The page shows the build log; after ~3–5
minutes it switches to the app. Upload a chessboard photo to try it.

**To update later:** pull the latest code and run the same command again. The
Space is kept an exact mirror of the bundle and rebuilds automatically.

---

## What gets uploaded, and why not `git push`

`deploy_hf_space.py` uploads an explicit allowlist: the Dockerfile,
`requirements.txt`, `app/`, `chess_vision/` (code + the one model the app
uses), `templates/`, `static/`, and `SPACE_README.md` as the Space's
`README.md` (its config). It never uploads `uploads/` (people's photos),
`.venv/`, training data, tests, or the unused large models.

A plain `git push` to the Space fails: Hugging Face rejects files over 10 MB
sent that way, and the model is 12 MB. The upload API handles that.

The exact bundle is smoke-tested by `bench/smoke_bundle.py`: it is installed
into an empty folder with a fresh virtualenv, started the way the Dockerfile
starts it, and run through upload → analyze → best move → play move.

## CPU Basic hardware: what to expect

- 2 vCPU, 16 GB RAM. One photo takes ~0.6 s to read; several people uploading
  at once queue up (a few seconds each), they don't fail.
- Sleeps after ~48 h without visitors; the next visit wakes it (~30 s).
- Uploaded photos live only in the running container: private to the
  uploader's session, deleted after 24 h, and gone on restart.

## Troubleshooting

- **401 / 403 from the script:** the token is read-only or for another
  account. Create a **Write** token for the account that owns the Space.
- **Build fails:** open the build log (the script prints the link). The
  Dockerfile installs Stockfish with apt and the Python packages from
  `requirements.txt`; a network hiccup there is fixed by re-running the
  deploy (or "Factory reboot" in the Space settings).
- **Stuck on "Starting":** the app must listen on port 8000 —
  `SPACE_README.md` says `app_port: 8000` and the Dockerfile serves on 8000.
  The script checks both before uploading.

## License

The model weights are Ultralytics YOLO derivatives (AGPL-3.0). Fine for
personal / learning use; a commercial deployment needs an Ultralytics
license or a different model.

## Other hosts

The same image runs anywhere that runs a Docker container listening on port
8000 with ~1 GB of memory, e.g. Google Cloud Run (`gcloud run deploy --source .`
from this folder, `--port 8000 --memory 2Gi`), Fly.io, Render, or a small VM
(`docker build -t chess-coach . && docker run -p 8000:8000 chess-coach`).
Keep a single instance/worker: sessions live in process memory.

Note: the Dockerfile has not been built in the environment where it was
written (no Docker daemon there). Its base image is pinned by digest to
Debian 12, and its system packages (libcairo2, stockfish) were derived from
what the Python wheels actually link against.
