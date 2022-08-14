from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import re

from model import get_config, get_results

re_model_name = re.compile(r'^[-_a-zA-Z0-9]+$')

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            with open('index.html','rb') as f:
                self.send_response(200)
                self.send_header('content-type','text/html')
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path == '/api/model':
            model_names = []
            for entry in os.scandir('models'):
                if entry.is_dir():
                    model_names.append(entry.name)
            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()

            models = []
            for model_name in sorted(model_names):
                results = get_results(f'models/{model_name}/results.yaml')
                models.append({'name':model_name, 'accuracy':results['accuracy']})

            self.wfile.write(json.dumps({
                'items': models
            }).encode('utf-8'))
        elif self.path.startswith('/api/model/'):
            model_name = self.path[len('/api/model/'):]
            if not re_model_name.match(model_name):
                self.send_error(400)
                return
            _,config = get_config(f'models/{model_name}/config.yaml')
            results = get_results(f'models/{model_name}/results.yaml')
            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'data': {
                    'layers': results['layers']
                }
            }).encode('utf-8'))
        else:
            self.send_error(404)

def serve(args):
    port = 8000
    server = HTTPServer(('127.0.0.1',port), RequestHandler)
    print(f"Serving on port {port}")
    server.serve_forever()
