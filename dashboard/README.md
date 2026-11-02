# Results dashboard

From the project root, run:

```bash
python3 dashboard/serve.py
```

Open **http://127.0.0.1:8080**. Requires Python 3.9+; no packages, frontend build, internet connection, or model weights are needed. Use `--port 8081` if the default port is occupied. The server binds to localhost by default.

**Paper results** loads `analysis/results`. Refresh after generating or updating analysis outputs. Analysis files are read only; the dashboard neither runs experiments nor rewrites results. Source paths embedded in archived JSON are not followed.

