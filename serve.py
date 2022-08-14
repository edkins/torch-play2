from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import re

from model import get_config, get_results
from transparency import create_highly_activating_input

re_model_name = re.compile(r'^[-_a-zA-Z0-9]+$')
re_model = re.compile(r'^\/api\/model\/([-_a-zA-Z0-9]+)$')
re_test = re.compile(r'^\/api\/model\/([-_a-zA-Z0-9]+)\/test$')
re_test_filename = re.compile(r'^test-[-_a-zA-Z0-9]+\.yaml$')

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if re_test.match(self.path):
            model_name = re_test.match(self.path)[1]

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

            models = []
            for model_name in sorted(model_names):
                results = get_results(f'models/{model_name}/results.yaml')
                models.append({'name':model_name, 'accuracy':results['accuracy']})

            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'items': models
            }).encode('utf-8'))
        elif re_model.match(self.path):
            model_name = re_model.match(self.path)[1]
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
        elif re_test.match(self.path):
            model_name = re_test.match(self.path)[1]
            test_names = []
            for entry in os.scandir(f'models/{model_name}/'):
                if not entry.is_dir() and re_test_filename.match(entry.name):
                    test_names.append(entry.name)
            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'items': [{'name':name} for name in sorted(test_names)]
            }).encode('utf-8'))
        else:
            self.send_error(404)

def serve(args):
    port = 8000
    server = HTTPServer(('127.0.0.1',port), RequestHandler)
    print(f"Serving on port {port}")
    server.serve_forever()
