# Deploy to Hugging Face Spaces

This deploys the app as a free public web service at
`https://huggingface.co/spaces/<your-username>/chess-vision-coach`.
Takes about 5 minutes; the actual click-through is 3 steps.

## What you get on the free tier

- Public URL with HTTPS, no DNS or hosting to manage
- 16 GB RAM, 2 vCPU, 50 GB disk — comfortably above what we need
- Auto-sleeps after ~48 h of inactivity, auto-wakes on the next request
  (cold-start: ~30 s while the model loads)
- Stockfish-at-strong (depth 22) takes ~2 s/move on CPU; Fast and
  Normal feel snappy

## Step 1 — create the Space (1 minute, in the browser)

1. Open <https://huggingface.co/new-space> (logged in as `siddhm11`).
2. **Owner:** your user. **Space name:** `chess-vision-coach`
   (or anything you like — the URL will follow).
3. **License:** AGPL-3.0
4. **Select the SDK:** **Docker** → **Blank** template.
5. Click **Create Space**.

You now have an empty Space repo at
`https://huggingface.co/spaces/siddhm11/chess-vision-coach`.

## Step 2 — push the code (one shell snippet)

```bash
# Clone the empty Space repo (HF gives you a username/password prompt;
# the password is an HF access token from huggingface.co/settings/tokens).
git clone https://huggingface.co/spaces/siddhm11/chess-vision-coach hf-space
cd hf-space

# Copy the app into the Space repo.
# (Run this from the root of the chessgame repo, ADJUST path if you are
# elsewhere.)
cp -r ../chessgame/chess-vision-coach/. .

# HF Spaces reads the front-matter YAML from README.md, not SPACE_README.md.
mv SPACE_README.md README.md

# Spaces does not auto-handle Git LFS for free; the 12 MB ONNX model is
# under GitHub's 100 MB limit AND under HF's 10 GB Space limit, so a
# plain commit is fine.
git add .
git commit -m "initial deploy"
git push
```

## Step 3 — wait for the build (~3 minutes)

Open the Space URL in the browser; you'll see a build log. When the
status flips to **Running** the app is live. Upload a chessboard photo
and try it.

---

## Troubleshooting

**"Build failed: pip out of memory"** — extremely unlikely with 16 GB,
but if it ever happens, edit `requirements.txt` to install onnxruntime
separately with `--no-cache-dir`.

**"Permission denied writing to /app/uploads"** — fixed in the
Dockerfile in this commit (runs as UID 1000 with explicit `chown`). If
you see this in an older Space, redeploy.

**"Stockfish: command not found"** — the Dockerfile installs it via
apt. If the Space falls back to classical-only, check the build log for
the apt step.

**"App went to sleep"** — that's expected on the free tier after 48 h
of no requests. The first request after sleep wakes the container; it
costs you ~30 s of cold start while the ONNX session warms.

---

## Updating the Space after a code change

Once the GitHub repo is updated:

```bash
cd hf-space
# Pull the latest code over.
cp -r ../chessgame/chess-vision-coach/. .
mv SPACE_README.md README.md  # only the YAML changes if you touched it
git add .
git commit -m "update from main repo"
git push
```

The Space rebuilds automatically on push.

---

## When to leave the free tier

- **You hit the rate limit** — the `/best-move` endpoint is already
  throttled per-session, but a Space with many concurrent users on
  Strong depth would spend most of its CPU in Stockfish. Upgrade to
  CPU upgrade tier (~$0.05/hr) or move Stockfish to a separate worker.
- **You want persistent sessions across cold starts** — swap
  `app/session.py` for Redis (the dataclass is clean; the store is
  ~50 lines). HF Spaces does not provide managed Redis, so you'd point
  at an Upstash / Redis Cloud free tier.
- **You hit the 48 h sleep at an awkward time** — pay for a Space that
  doesn't sleep, or hit it with a cron pinger.
