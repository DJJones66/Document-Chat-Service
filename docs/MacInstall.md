# Mac Install Plan

## Scope
- Use existing scripts in `service_scripts` only; no changes to Windows/Ubuntu flows.

## Prereqs
- macOS with Python 3.11-3.13 available (`python3.11` recommended).
- Xcode Command Line Tools (needed for native builds): `xcode-select --install`
- Optional: Ollama installed if you want model auto-pulls.

## Plan
1. Create venv
   - From repo root: `python3.11 service_scripts/create_venv.py`
   - Rebuild if needed: `VENV_FORCE_RECREATE=1 python3.11 service_scripts/create_venv.py`
   - Custom location: `VENV_PATH=/path/to/.venv python3.11 service_scripts/create_venv.py`
2. Install deps
   - Full/pinned install: `python3.11 service_scripts/install_with_venv.py --full`
   - Editable install: `python3.11 service_scripts/install_with_venv.py`
   - Skip pip upgrade if needed: `python3.11 service_scripts/install_with_venv.py --no-upgrade-pip`
3. Start service
   - `python3.11 service_scripts/start_with_venv.py`
   - Defaults: `API_HOST=0.0.0.0`, `API_PORT=18000`
   - Hot reload: `UVICORN_RELOAD=1 python3.11 service_scripts/start_with_venv.py`
   - Note: first run will create `.env` from `.env.local.example` if missing.
4. Shutdown service
   - `python3.11 service_scripts/shutdown_with_venv.py`
   - If port changed: `API_PORT=19000 python3.11 service_scripts/shutdown_with_venv.py`
5. Restart service
   - `python3.11 service_scripts/restart_with_venv.py`
   - Adjust delay: `RESTART_DELAY=2 python3.11 service_scripts/restart_with_venv.py`
6. Run Mac system test script (optional)
   - `bash service_scripts/mac_system_test.sh`
   - Overrides: `API_HOST=127.0.0.1 PORT_START=18081 PORT_COUNT=10 PYTHON_BIN=python3.11 VENV_PATH=.venv`
   - Ollama controls: `REQUIRE_OLLAMA=1` to hard-fail if Ollama missing, or `OLLAMA_HEALTH_SKIP=1` to skip health checks.
   - Remote Ollama: defaults to `OLLAMA_REMOTE_HOST=10.1.2.149` and `OLLAMA_REMOTE_PORT=11434`. Override with `OLLAMA_REMOTE_HOST=localhost` or set `OLLAMA_LLM_BASE_URL/OLLAMA_EMBEDDING_BASE_URL/OLLAMA_CONTEXTUAL_LLM_BASE_URL`.
   - Script runs: create -> install -> start -> health -> restart -> health -> shutdown.
   - Logs: `data/test_start_stdout.log`, `data/test_start_stderr.log`, `data/test_restart_stdout.log`, `data/test_restart_stderr.log`

## Verify
- `curl http://localhost:18000/health`

## Unresolved questions
- None.

## Run notes (Mac system test)
- Ran with conda env `BrainDriveDev` using `conda run -n BrainDriveDev bash service_scripts/mac_system_test.sh` and `PYTHON_BIN` set to conda python.
- Initial run timed out during dependency install (took >120s in this environment).
- Next run failed at startup: app exited because Ollama health check could not connect (`httpx.ConnectError`), and health endpoint never came up.
- Added opt-in skip for Ollama health checks via `OLLAMA_HEALTH_SKIP=1` and had the Mac test script auto-set it when Ollama is missing/not reachable.
- Restart step failed due to address in use; root cause was shutdown script skipping process scan when `API_PORT` set on non-Windows. Fixed to scan when port is set but no PID targets found.
- Final rerun succeeded: start -> health -> restart -> health -> shutdown completed; restart log shows clean startup.
- Note: because Ollama is not installed in this environment, the run validates service lifecycle but not Ollama-backed features; rerun with Ollama running or `REQUIRE_OLLAMA=1` to enforce.
- Cleaned up leftover uvicorn on port 18081 from earlier run via `API_PORT=18081 .venv/bin/python service_scripts/shutdown_with_venv.py`.
- Full test run with remote Ollama: `REQUIRE_OLLAMA=1 OLLAMA_REMOTE_HOST=10.1.2.149` set; start/health/restart/health/shutdown completed successfully.
- Verified script reports `Using Ollama base URL: http://10.1.2.149:11434` during full test run.
