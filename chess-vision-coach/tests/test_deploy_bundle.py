"""The Hugging Face deploy bundle ships what the app needs -- and nothing
private or bloated. (bench/smoke_bundle.py goes further: it installs and runs
the staged bundle from an empty folder.)"""

from __future__ import annotations

from pathlib import Path

import pytest

import deploy_hf_space as d


def test_bundle_contains_everything_the_app_loads():
    files = {p.as_posix() for p in d.collect_files()}
    for required in [
        "Dockerfile", "requirements.txt", "app/main.py", "app/engine.py",
        "app/session.py", "chess_vision/vision.py", "chess_vision/model_backend.py",
        "chess_vision/sanity.py", "templates/base.html", "templates/_analyze.html",
        "static/app.css", "static/htmx.min.js",
        "chess_vision/models/yolov8n640-chess-finetuned.onnx",
    ]:
        assert required in files, required
    # Every template the app references must ship.
    for tpl in (d.ROOT / "templates").glob("*.html"):
        assert f"templates/{tpl.name}" in files


def test_bundle_excludes_private_and_bulky_files():
    files = [p.as_posix() for p in d.collect_files()]
    forbidden = ("uploads/", ".venv/", "finetune_data/", "tests/", "bench/",
                 "model-upload/", "__pycache__", "yolov8m-chess.onnx", "test_photos/")
    assert not [f for f in files if any(x in f for x in forbidden)]
    total = sum((d.ROOT / f).stat().st_size for f in files)
    assert total < 20e6, f"bundle unexpectedly large: {total / 1e6:.1f} MB"


def test_default_model_is_the_one_shipped():
    """ModelBackend picks the first existing model from its candidate list;
    the bundle must contain exactly that first choice."""
    from chess_vision import model_backend
    src = Path(model_backend.__file__).read_text()
    first = src.split("for candidate in (", 1)[1].split('"')[1]
    assert first == "yolov8n640-chess-finetuned.onnx"
    assert f"chess_vision/models/{first}" in {p.as_posix() for p in d.collect_files()}


def test_space_config_consistent():
    d.check_config()   # raises SystemExit on mismatch


def test_stage_writes_space_readme_as_readme(tmp_path):
    staged = d.stage(d.ROOT, tmp_path)
    assert Path("README.md") in staged
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n") and "sdk: docker" in text


def test_dry_run_uploads_nothing(capsys):
    assert d.main(["--space", "someone/chess-vision-coach", "--dry-run"]) == 0
    assert "dry run: nothing uploaded" in capsys.readouterr().out


def test_rejects_malformed_space_id():
    with pytest.raises(SystemExit):
        d.main(["--space", "no-slash-here", "--dry-run"])


class _FakeApi:
    """Stands in for huggingface_hub.HfApi (not an app dependency)."""
    calls: list = []

    def __init__(self, token=None, exists=False, pro=False):
        self._exists, self._pro = exists, pro

    def repo_exists(self, repo_id, repo_type=None):
        return self._exists

    def whoami(self):
        return {"name": "siddhm11", "isPro": self._pro, "orgs": []}

    def create_repo(self, *a, **k):
        _FakeApi.calls.append(("create_repo", a, k))

    def upload_folder(self, **k):
        _FakeApi.calls.append(("upload_folder", (), k))


def _fake_hub(monkeypatch, **state):
    import sys, types
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = lambda token=None: _FakeApi(token, **state)
    mod.get_token = lambda: "hf_test"
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    _FakeApi.calls = []


def test_free_account_gets_a_clear_paid_plan_message(monkeypatch, capsys):
    _fake_hub(monkeypatch, exists=False, pro=False)
    assert d.main(["--space", "siddhm11/chess-vision-coach", "--token", "x"]) == 2
    err = capsys.readouterr().err
    assert "require a paid plan" in err and "PRO" in err
    assert _FakeApi.calls == [], "nothing created or uploaded"


def test_pro_account_creates_then_uploads_mirror(monkeypatch):
    _fake_hub(monkeypatch, exists=False, pro=True)
    assert d.main(["--space", "siddhm11/chess-vision-coach", "--token", "x"]) == 0
    kinds = [c[0] for c in _FakeApi.calls]
    assert kinds == ["create_repo", "upload_folder"]
    assert _FakeApi.calls[0][2]["space_sdk"] == "docker"
    assert _FakeApi.calls[1][2]["delete_patterns"] == "*"


def test_existing_space_is_updated_without_creating(monkeypatch):
    _fake_hub(monkeypatch, exists=True, pro=False)
    assert d.main(["--space", "siddhm11/chess-vision-coach", "--token", "x"]) == 0
    assert [c[0] for c in _FakeApi.calls] == ["upload_folder"]
