"""GEFS PWAT/CAPE downloader against a local stand-in for NOAA's S3 bucket."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline import download_gefs as dg

# A fake pgrb2a file: 4 "messages"; PWAT and CAPE are not adjacent and not first.
MSGS = [(b'AAAA' * 10, ':HGT:500 mb:'), (b'PWAT' * 7, ':PWAT:entire atmosphere (considered as a single layer):'),
        (b'RHRH' * 5, ':RH:850 mb:'), (b'CAPE' * 9, ':CAPE:180-0 mb above ground:')]
BODY = b''.join(m for m, _ in MSGS)
IDX, off = [], 0
for i, (m, tag) in enumerate(MSGS, 1):
    IDX.append(f'{i}:{off}:d=2026101500{tag}anl:'); off += len(m)
IDX = '\n'.join(IDX) + '\n'
FAIL_FIRST = {'n': 0}


class S3(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if 'f006' in self.path and FAIL_FIRST['n'] < 2:      # throttle the first two requests for f006
            FAIL_FIRST['n'] += 1; self.send_response(503); self.end_headers(); return
        if 'f123' in self.path and 'gespr' in self.path:     # this file "does not exist"
            self.send_response(404); self.end_headers(); return
        if self.path.endswith('.idx'):
            body = IDX.encode()
        else:
            a, b = self.headers['Range'].split('=')[1].split('-')
            body = BODY[int(a): int(b) + 1 if b else None]
        self.send_response(206 if not self.path.endswith('.idx') else 200); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(('127.0.0.1', 0), S3)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(dg, 'BASE_URL', f'http://127.0.0.1:{srv.server_address[1]}')
    monkeypatch.setattr(dg.time, 'sleep', lambda s: None)
    yield
    srv.shutdown()


def test_leads_cover_d1_to_d5():
    assert dg.LEADS == list(range(3, 124, 3)) and len(dg.LEADS) == 41


def test_download_writes_pwat_then_cape_and_reports_missing(server, tmp_path):
    with pytest.raises(RuntimeError, match='1 GEFS files failed'):
        dg.download_cycle('20261015', str(tmp_path))
    for product in dg.PRODUCTS:
        for lead in dg.LEADS:
            path = dg.file_path(str(tmp_path), product, lead)
            if product == 'gespr' and lead == 123:
                assert not (tmp_path / path).exists()
                continue
            assert open(path, 'rb').read() == MSGS[1][0] + MSGS[3][0], (product, lead)
    assert not list(tmp_path.rglob('*.part'))
