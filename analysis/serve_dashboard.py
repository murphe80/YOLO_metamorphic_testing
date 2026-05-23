#!/usr/bin/env python3
"""
Simple HTTP server to view the YOLO metamorphic testing dashboard.
Run this script and open the URL in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8000
DASHBOARD_FILE = "dashboard.html"

def main():
    # Change to the analysis directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Create a simple HTTP request handler with CORS enabled
    class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

        def log_message(self, format, *args):
            # Custom logging to make it cleaner
            if not args[0].startswith('GET /favicon.ico'):
                super().log_message(format, *args)

    # Start server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        url = f"http://localhost:{PORT}/{DASHBOARD_FILE}"

        print("=" * 70)
        print("YOLO Metamorphic Testing Dashboard")
        print("=" * 70)
        print(f"\n✓ Server started successfully")
        print(f"✓ Dashboard URL: {url}")
        print(f"✓ Serving from: {script_dir}")
        print(f"\nOpening dashboard in your browser...")
        print(f"\n⌨Press Ctrl+C to stop the server\n")
        print("=" * 70)

        # Open browser
        webbrowser.open(url)

        # Start serving
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped")
            print("=" * 70)

if __name__ == "__main__":
    main()
