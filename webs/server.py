import http.server
import json
import os
import socketserver
import urllib.parse
from pathlib import Path
import sys

# Thiết lập đường dẫn
DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
sys.path.append(str(DIR))
import simulation_core

PORT = int(os.environ.get('PORT', 3000))

# Đọc file HTML từ thư mục templates
def get_html_content():
    template_path = DIR / 'templates' / 'index.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

class WCRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(get_html_content().encode('utf-8'))
        elif path == '/api/simulate':
            result = simulation_core.simulate_full_tournament(seed=None)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        elif path == '/api/monte_carlo':
            runs = int(query.get('runs', ['100'])[0])
            summary = simulation_core.run_monte_carlo(runs=min(max(runs, 10), 1000))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(summary).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('0.0.0.0', PORT), WCRequestHandler) as httpd:
        print(f"Server World Cup Simulation dang chay tai: http://localhost:{PORT}")
        httpd.serve_forever()
