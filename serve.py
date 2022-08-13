from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            with open('index.html','rb') as f:
                self.send_response(200)
                self.send_header('content-type','text/html')
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path == '/api/model':
            models = []
            for entry in os.scandir('models'):
                if entry.is_dir():
                    models.append(entry.name)
            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'items': [{'name':name} for name in sorted(models)]
            }).encode('utf-8'))
        else:
            self.send_error(404)

def serve(args):
    port = 8000
    HTTPServer(('127.0.0.1',port), RequestHandler).serve_forever()