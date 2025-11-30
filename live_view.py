# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:44:18 2025

@author: Judson
"""

# live_view.py
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import webbrowser, pathlib

PORT = 8000
root = pathlib.Path(__file__).parent.resolve()

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kw):
        super().__init__(*args, directory=str(root), **kw)

if __name__ == "__main__":
    webbrowser.open(f"http://localhost:{PORT}/viewer.html")
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Serving on http://localhost:{PORT}")
        httpd.serve_forever()
