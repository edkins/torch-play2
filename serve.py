from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import jsonschema
import os
import re
from traceback import print_exception
from urllib.parse import urlparse, parse_qs

from model import get_config, get_results
from transparency import create_and_run_test, get_test

re_model_name = re.compile(r'^[-_a-zA-Z0-9]+$')
re_model = re.compile(r'^\/api\/model\/([-_a-zA-Z0-9]+)$')
re_test = re.compile(r'^\/api\/model\/([-_a-zA-Z0-9]+)\/test$')
re_test_filename = re.compile(r'^(test-[-_a-zA-Z0-9,]+)\.yaml$')

test_payload_schema = {
    'type': 'object',
    'properties': {
        'layer': {'type':'string'},
        'neuron': {'type':'string'}
    },
    'required': ['layer','neuron']
}

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if re_test.match(self.path):
            model_name = re_test.match(self.path)[1]
            length = int(self.headers['content-length'])
            payload_data = self.rfile.read(length)
            try:
                payload = json.loads(payload_data)
                jsonschema.validate(instance=payload, schema=test_payload_schema)
            except:
                self.send_error(400)
                return

            try:
                test_name = create_and_run_test(model_name, payload['layer'], payload['neuron'])
            except Exception as e:
                print_exception(e)
                self.send_error(500)
                return
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)


    def do_GET(self):
        o = urlparse(self.path)
        path = o.path
        if path in ['/', '/index.html']:
            with open('index.html','rb') as f:
                self.send_response(200)
                self.send_header('content-type','text/html')
                self.end_headers()
                self.wfile.write(f.read())
        elif path == '/api/model':
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
        elif re_model.match(path):
            model_name = re_model.match(path)[1]
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
        elif re_test.match(path):
            query = parse_qs(o.query)
            model_name = re_test.match(path)[1]
            test_names = []
            for entry in os.scandir(f'models/{model_name}/'):
                m = re_test_filename.match(entry.name)
                if m:
                    test_names.append(m[1])
            self.send_response(200)
            self.send_header('content-type','application/json')
            self.end_headers()
            tests = []
            for name in sorted(test_names):
                if len(query) > 0:
                    details = get_test(f'models/{model_name}/{name}.yaml')
                    if 'layer' in query and details['layer'] != query['layer'][0]:
                        continue
                    if 'neuron' in query and details['neuron'] != query['neuron'][0]:
                        continue
                tests.append({'name': name, 'layer':details['layer'], 'neuron':details['neuron']})
            self.wfile.write(json.dumps({
                'items': tests
            }).encode('utf-8'))
        else:
            self.send_error(404)

def serve(args):
    port = 8000
    server = HTTPServer(('127.0.0.1',port), RequestHandler)
    print(f"Serving on port {port}")
    server.serve_forever()
