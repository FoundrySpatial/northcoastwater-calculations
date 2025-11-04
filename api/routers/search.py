import json
from flask import Blueprint, jsonify, current_app as app
from db_utils import get_search_distance
from utils.api_utils import check_query_params, read_json

search = Blueprint('search', __name__)

POSTGRES_MAX_BIGINT = 9223372036854775807

@search.route('/', methods=['GET'])
@check_query_params(['placeName', 'waterRight', 'lat', 'lng', 'stream'], False)
def search_all(params):
    
    (place_name,
    water_right,
    lat,
    lng,
    stream
    ) = params

    if all(param == None for param in params):
        raise Exception({"message": "Please provide at least one search term in the query parameters. Valid search criteria are 'placeName', 'waterRight', 'lat', 'lng', and 'stream'.", "status_code": 422})
    
    if ((lat and not lng) or (lng and not lat)):
        raise Exception({"message": "To search by coordinates, provide both a 'lat' and 'lng' parameter.", "status_code": 422})        

    if (place_name):
        result = app.db.get_stream_by_name(place_name)
        if (result is None):
            raise Exception({'status_code':404, 'message': 'Searched name does not match any database entries'})
        else:
            return result, 200

    return jsonify({'pod_ids': [1,2,3]}), 200

@search.route('/nhd-id', methods=['GET'])
@check_query_params(['lat', 'lng', 'zoom'])
def search_nhd(params): 
    (
    lat,
    lng,
    zoom
    ) = params

    try:
        zoom = round(float(zoom))
        distance = get_search_distance(zoom)
        if (distance == None):
            raise Exception({'status_code': 400, "message": "zoom level must be a non-negative integer"})
        
        result = app.db.get_nhd_id_by_lat_lng(lat, lng, distance)
        if (result.get('nhdplusid') == None):
            raise Exception({'status_code': 404, "message": "Selected location is outside the study area or too far from a stream. Please select a different stream reach."})
        if (result.get('nhdplusid') == 'Flow-regulated mainstem river'):
            raise Exception({'status_code': 400, "message": "Selected stream reach ({}) is likely considered a flow-regulated mainstem river under the North Coast Policy and is not supported by this tool. Please consult Policy section 3.2 and Water Board staff for necessary considerations or choose another stream reach.".format(result.get('name'))})
        return result, 200
    except ValueError:
        raise Exception({'status_code': 400, "message": "zoom level must be a valid integer"})
    
@search.route('/nhd-id/<int:nhdplusid>', methods=['GET'])
def search_ws_by_nhdplusid(nhdplusid):

    if(nhdplusid > POSTGRES_MAX_BIGINT or nhdplusid < 0):
        return json.dumps({}), 200

    result = app.db.get_ws_bbox_by_nhdplusid(nhdplusid)
    if (result == None):
        return json.dumps({}), 200
    else:
        return result, 200
    
@search.route('/poi-data-latlng', methods = ['GET'])
@check_query_params(['lat', 'lng'])
def get_poi_data_by_lat_lng(params):
    (lat,lng) = params
    #Set 100 = minimum search distance
    distance = 100
    
    result = app.db.get_nhd_id_by_lat_lng(lat, lng, distance)
    if (result.get('nhdplusid') == None):
        raise Exception({'status_code': 404, "message": "Selected location is outside the study area or too far from a stream. Please select a different stream reach."})
    nhdplusid = result.get('nhdplusid')
    if(nhdplusid == "Flow-regulated mainstem river"):
        raise Exception({"status_code": 400, "message": "Cumulative Diversion Analysis is not supported on the selected stream reach as it is considered flow regulated in practice. Please consult with the Water Board staff before proceeding to develop a Cumulative Diversion Analysis for water right application purposes."})
    response = app.db.get_poi_data_by_nhd_id(nhdplusid)
    if(response == None):
        raise Exception({"status_code": 404, "message": "Unable to find poi data for given lat and long"})
    response['nhd_id'] = nhdplusid
    return response, 200