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
            swap_in = query.get('swap_in', [''])[0].strip() or None
            swap_out = query.get('swap_out', [''])[0].strip() or None
            try:
                result = simulation_core.simulate_full_tournament(
                    seed=None, swap_in=swap_in, swap_out=swap_out
                )
                status = 200
            except ValueError as exc:
                status = 400
                result = {'error': str(exc)}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        elif path == '/api/monte_carlo':
            runs = int(query.get('runs', ['100'])[0])
            swap_in = query.get('swap_in', [''])[0].strip() or None
            swap_out = query.get('swap_out', [''])[0].strip() or None
            try:
                summary = simulation_core.run_monte_carlo(
                    runs=min(max(runs, 10), 1000),
                    swap_in=swap_in,
                    swap_out=swap_out,
                )
                status = 200
            except ValueError as exc:
                status = 400
                summary = {'error': str(exc)}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(summary).encode('utf-8'))
        elif path == '/api/draw':
            seed = query.get('seed', [''])[0].strip()
            try:
                payload = simulation_core.draw_groups(int(seed) if seed else None)
                simulation_core.set_current_draw(payload)
                status = 200
            except (ValueError, KeyError) as exc:
                status = 400
                payload = {'error': str(exc)}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        elif path == '/api/draw/reset':
            simulation_core.clear_current_draw()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'ok': True,
                'draw': simulation_core.real_groups_payload(),
            }).encode('utf-8'))
        elif path == '/api/teams':
            teams = simulation_core.selectable_teams()
            all_teams = sorted(simulation_core.load_initial_states())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({
                'teams': teams,
                'all_teams': all_teams,
                'draw': simulation_core.current_draw_payload(),
                'real_groups': simulation_core.real_groups_payload(),
                'swap_pool': simulation_core.swap_pool(20),
            }).encode('utf-8'))
        elif path == '/api/match_prediction':
            home = query.get('home', [''])[0].strip()
            away = query.get('away', [''])[0].strip()
            try:
                result = simulation_core.predict_match(home, away)
                status = 200
            except (KeyError, ValueError) as exc:
                status = 400
                result = {'error': str(exc)}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

if __name__ == '__main__':
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(('0.0.0.0', PORT), WCRequestHandler) as httpd:
        print(f"Server World Cup Simulation dang chay tai: http://localhost:{PORT}")
        httpd.serve_forever()
