# Repository Guidelines

## Project Structure & Module Organization
`manga_pipeline/` contains the FastAPI app, task orchestration, provider adapters, SQLite access, and the image-core bridge. `static/` holds the browser UI (`index.html`, `styles.css`, `app.js`). `tests/` contains the `pytest` suite. Runtime data is written to `uploads/`, `output/`, `data/pipeline.db`, `models/`, and `.local/settings.json`. The vendored upstream core lives in `vendor/manga-image-translator/`; avoid editing it unless a change truly belongs upstream.

## Build, Test, and Development Commands
Use the local virtualenv after first install:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/vendor/manga-image-translator:$PWD"
python -m uvicorn manga_pipeline.main:app --host 127.0.0.1 --port 8765
```

This starts the local web app on `http://127.0.0.1:8765`. Run all tests with:

```bash
source .venv/bin/activate
pytest
```

Use `python scripts/check_environment.py` when diagnosing missing models or runtime setup.

## Coding Style & Naming Conventions
Target Python `3.11` and keep 4-space indentation. Match the existing style: type hints on public functions, small helper functions, and explicit error messages. Keep comments/docstrings concise and in Chinese when explaining repo-specific behavior. In frontend code, prefer plain, readable DOM logic over abstraction-heavy helpers. File and test names follow `snake_case`; task/image identifiers are generated, short hex strings.

## Testing Guidelines
The project uses `pytest` with tests under `tests/` and filenames like `test_tasks.py` or `test_engine.py`. Add or update tests whenever a change affects task state, provider behavior, editor routes, or rerender/reprocess flows. Prefer focused unit-style coverage around the touched module, then run full `pytest` before finishing.

## Commit & Pull Request Guidelines
Recent history mixes concise English and Chinese subjects such as `fix: refine rotation handle interaction` and `新增注释和任务失败重启功能`. Follow that pattern: one-line, imperative, specific to the user-visible change. PRs should explain the behavior change, list affected areas, mention test coverage, and include screenshots for UI updates.

## Security & Repository Notes
Never commit `.local/settings.json`, API keys, output artifacts, or model files. Treat `vendor/manga-image-translator/` as potentially dirty user state and do not revert it casually. The repository intentionally avoids bulk destructive cleanup; remove files only by explicit path.
