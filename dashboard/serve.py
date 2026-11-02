#!/usr/bin/env python3
"""Run the local, read-only research dashboard using Python's standard library."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from data import ROOTS, build

STATIC = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route.startswith('/api/'):
            source = route.removeprefix('/api/')
            if source not in ROOTS:
                self.send_error(404)
                return
            try:
                payload = json.dumps(build(source), allow_nan=False).encode()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.send_error(500, f'Could not read analysis: {exc}')
                return
            self.respond(payload, 'application/json; charset=utf-8')
            return
        if route.startswith('/results/'):
            parts = route.split('/', 3)
            if len(parts) != 4 or parts[2] not in ROOTS:
                self.send_error(404)
                return
            root, relative = ROOTS[parts[2]], parts[3]
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or path.suffix not in ('.pdf', '.png', '.jpg', '.json', '.md'):
                self.send_error(403)
                return
        else:
            name = route.lstrip('/') or 'index.html'
            if name not in ('index.html', 'app.js', 'style.css'):
                self.send_error(404)
                return
            path = STATIC / name
        try:
            payload = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.respond(payload, mimetypes.guess_type(path.name)[0] or 'application/octet-stream')

    def respond(self, payload, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Dashboard: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
