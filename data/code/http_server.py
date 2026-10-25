"""A minimal HTTP server that responds with Hello on every GET request (standard library).

tags: http web server simple serve requests
"""

from http.server import BaseHTTPRequestHandler, HTTPServer


class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"Hello from MNX"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def http_server(port: int = 8000) -> HTTPServer:
    """Create the server; call .serve_forever() to run it."""
    return HTTPServer(("127.0.0.1", port), HelloHandler)


# tests
import threading
import urllib.request
srv = http_server(0)
threading.Thread(target=srv.handle_request, daemon=True).start()
assert urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/").read() == b"Hello from MNX"
srv.server_close()
