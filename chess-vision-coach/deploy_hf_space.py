"""Deploy Chess Vision Coach to a Hugging Face Docker Space -- one command.

    pip install huggingface_hub
    python deploy_hf_space.py --space <your-hf-username>/chess-vision-coach

Creates the Space if it doesn't exist, then uploads ONLY the files the app
needs (an explicit allowlist below). Never uploads uploads/, .venv/, training
data, tests, or the unused large models.

Why not `git push`: Hugging Face rejects plain git pushes of files over
10 MB, and the model is 12 MB. The upload API used here handles large files
automatically.

Auth: --token, or the HF_TOKEN environment variable, or a saved login
(`hf auth login` / `huggingface-cli login`), or you are prompted. The token
needs WRITE access: https://huggingface.co/settings/tokens

    --dry-run   list exactly what would be uploaded, upload nothing
    --private   create the Space as private
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Everything the container needs, nothing else. Paths are relative to ROOT.
INCLUDE = [
    "Dockerfile",
    "requirements.txt",
    "app/*.py",
    "chess_vision/*.py",
    "chess_vision/models/yolov8n640-chess-finetuned.onnx",
    "templates/*.html",
    "static/*.css",
    "static/*.js",
]
SPACE_README = "SPACE_README.md"      # uploaded as README.md (Space config)
APP_PORT = 8000

NEEDS_PAID_PLAN = """\
Hugging Face only lets paid accounts create Docker Spaces ("Gradio and Docker
Spaces run on compute and require a paid plan to create: PRO for personal
accounts" -- https://huggingface.co/docs/hub/spaces-overview). The account
'{user}' is not PRO, so the Space can't be created.

Options:
  1. Subscribe to HF PRO (https://huggingface.co/pro), then re-run this
     command. The Space itself runs on free "CPU Basic" hardware.
  2. Host the same Docker image elsewhere -- see DEPLOY_HF_SPACE.md,
     "Other hosts".
"""


def collect_files(root: Path = ROOT) -> list[Path]:
    """Relative paths of every file to upload (README excluded)."""
    files: set[Path] = set()
    for pattern in INCLUDE:
        matches = sorted(p for p in root.glob(pattern) if p.is_file())
        if not matches:
            raise SystemExit(f"deploy bundle is missing required files: {pattern}")
        files.update(p.relative_to(root) for p in matches)
    return sorted(files)


def check_config(root: Path = ROOT) -> None:
    """Fail early on the mistakes that leave a Space stuck at 'Starting'."""
    readme = (root / SPACE_README).read_text(encoding="utf-8")
    front = re.match(r"^---\n(.*?)\n---\n", readme, re.S)
    if not front:
        raise SystemExit(f"{SPACE_README} has no YAML front matter")
    meta = front.group(1)
    if not re.search(r"^sdk:\s*docker\s*$", meta, re.M):
        raise SystemExit(f"{SPACE_README}: front matter must say `sdk: docker`")
    if not re.search(rf"^app_port:\s*{APP_PORT}\s*$", meta, re.M):
        raise SystemExit(f"{SPACE_README}: front matter must say `app_port: {APP_PORT}`")
    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    if f'"--port", "{APP_PORT}"' not in docker:
        raise SystemExit(f"Dockerfile CMD must serve on port {APP_PORT}")


def stage(root: Path, dest: Path) -> list[Path]:
    files = collect_files(root)
    for rel in files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest / rel)
    shutil.copy2(root / SPACE_README, dest / "README.md")
    return files + [Path("README.md")]


def get_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token as hf_saved_token
        saved = hf_saved_token()
        if saved:
            return saved
    except Exception:
        pass
    print("Paste a Hugging Face token with WRITE access "
          "(https://huggingface.co/settings/tokens):")
    return getpass.getpass("token: ").strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", required=True, help="owner/name, e.g. siddhm11/chess-vision-coach")
    ap.add_argument("--token", help="HF write token (default: HF_TOKEN env or saved login)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="list files, upload nothing")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.space):
        ap.error("--space must look like owner/name")
    check_config()

    with tempfile.TemporaryDirectory(prefix="cvc-space-") as tmp:
        staged = stage(ROOT, Path(tmp))
        total = sum((Path(tmp) / f).stat().st_size for f in staged)
        print(f"{len(staged)} files, {total / 1e6:.1f} MB:")
        for f in staged:
            print(f"  {f.as_posix():<55} {(Path(tmp) / f).stat().st_size / 1e3:>9.1f} kB")
        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is not installed: pip install huggingface_hub", file=sys.stderr)
            return 1
        api = HfApi(token=get_token(args.token))
        if not api.repo_exists(args.space, repo_type="space"):
            # Hugging Face: "Gradio and Docker Spaces run on compute and
            # require a paid plan to create: PRO for personal accounts."
            # (hub docs, Spaces Overview). Say so up front instead of failing
            # later with a permission error that looks like a bad token.
            me = api.whoami()
            orgs = {o.get("name") for o in me.get("orgs", [])}
            owner = args.space.split("/")[0]
            if owner == me.get("name") and not me.get("isPro"):
                print(NEEDS_PAID_PLAN.format(user=owner), file=sys.stderr)
                return 2
            if owner != me.get("name") and owner not in orgs:
                print(f"You ({me.get('name')}) can't create Spaces under '{owner}'.", file=sys.stderr)
                return 2
            try:
                api.create_repo(args.space, repo_type="space", space_sdk="docker",
                                private=args.private)
            except Exception as exc:   # e.g. 402/403 for an org without a paid plan
                print(f"Hugging Face refused to create the Space: {exc}\n\n"
                      + NEEDS_PAID_PLAN.format(user=owner), file=sys.stderr)
                return 2
        api.upload_folder(
            folder_path=tmp, repo_id=args.space, repo_type="space",
            commit_message="Deploy Chess Vision Coach",
            # Keep the Space an exact mirror of the bundle: remove files a
            # previous deploy uploaded that are no longer part of it.
            delete_patterns="*",
        )

    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nUploaded. The Space now builds (~3-5 min): {url}")
    print(f"Build logs: {url}?logs=build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
