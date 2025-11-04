import json
from flask import jsonify, abort, request, make_response
import gzip

def read_json(path: str):
    """
    Load json from file path.

    :param path string - path to file relative to the /api dir.
    """
    with open("./{}".format(path), 'r') as file:
        return json.loads(file.read())

def json_abort(status_code, data=None):
    response = jsonify(data)
    response.status_code = status_code
    abort(response)

def check_body_json_content(key_list:list):
    """
        Route decorator. Given a list of JSON keys to check for,
        will provide the route with a list of destructured values
        and handle malformed request error if missing.

        :params list key_list - list of keys expected on json data.
    """
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            if request.headers.get('Content-Type') != 'application/json':
                raise Exception({'message':"Could not read body content. Ensure to use json format and set the 'Content-Type' header to 'application/json'.", 'status_code': 400})
            params = []
            for key in key_list:
                value = request.json.get(key)
                if (not value): 
                    raise Exception({'message':'Malformed request, body content key {} is required'.format(key), 'status_code': 400})
                params.append(value)
            return func(params, *args, **kwargs)
        wrapper.__name__ = func.__name__         
        return wrapper
    return real_decorator

def check_query_params(key_list:list, required = True):
    """
        Check for query parameter existence and handle with error message if missing. Used for required params.
    """
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            params = []
            for key in key_list:
                query_param = request.args.get(key)
                if (not query_param and required): 
                    raise Exception({'message':'Malformed request, query parameter {} is required'.format(key), 'status_code': 400})
                params.append(query_param)
            return func(params, *args, **kwargs)
        wrapper.__name__ = func.__name__         
        return wrapper

    return real_decorator

# TODO: Look into handling compression at the nginx level instead of the flask level
def get_compressed_response(raw_content, content_type='application/json'):
    content = gzip.compress(json.dumps(raw_content).encode('utf8'), 5)
    response = make_response(content)
    response.headers['Content-length'] = len(content)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Type'] = content_type
    return response