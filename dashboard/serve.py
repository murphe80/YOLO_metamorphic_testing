#!/usr/bin/env python3
"""
Simple HTTP server for the metamorphic testing dashboard.
Serves the dashboard on http://localhost:8000
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
import os

PORT = 8000

def main():
    # Change to the project root (parent of dashboard directory)
    # This allows the server to access both dashboard/ and analysis/results/
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Dashboard server starting on http://localhost:{PORT}")
        print(f"Serving files from: {Path.cwd()}")
        print("\nPress Ctrl+C to stop the server\n")

        # Open browser automatically
        webbrowser.open(f"http://localhost:{PORT}/dashboard/dashboard.html")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
