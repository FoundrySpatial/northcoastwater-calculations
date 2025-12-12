from flask import Blueprint, current_app as app, jsonify, request, Response, g
from authentication.guards import authorization_guard
from fixtures.water_rights import water_right_fixture, water_rights_fixture
from utils.api_utils import get_compressed_response, check_query_params
from utils.wsr_csv_utils import get_wsr_water_rights_csv_formatted, get_wsr_water_rights_json_formatted, get_intermediate_data_json_formatted, get_adjusted_csv_data, sort_and_format_unsorted_csv_data
from datetime import datetime
import decimal
import pytz

points_of_diversion = Blueprint('points_of_diversion', __name__)

@points_of_diversion.route('/', methods=['GET'])
@authorization_guard
def get_all_points_of_diversion():
    """
    Get all PODs rights.
    """
    pods = app.db.get_all_pods()
    content = get_compressed_response(pods)
    return content, 200

@points_of_diversion.route('/water-rights', methods=['GET'])
@authorization_guard
@check_query_params(['lat', 'lng', 'nhd', 'session_id'])
def get_points_of_diversion_water_rights(params):
    """
    Get all PODs water rights.
    """
    ( lat, lng, nhd, session_id ) = params
    watershed = app.db.get_watershed_by_nhd_id(nhd)
    date = datetime.now(pytz.utc)
    unsorted_water_rights_csv_data = app.db.get_unsorted_senior_diverter_csv(nhdplusid = nhd, lat = lat, lng = lng, date = date)
    wsr_session = app.db.get_wsr_session_by_id(g.user_id, session_id)['session']
    water_rights_dicts = [dict(row) for row in unsorted_water_rights_csv_data]
    application_numbers = [row['application_number'] for row in water_rights_dicts]
    pod_diverter_rain_and_area = app.db.get_diverter_size_and_mean_precip(application_numbers = application_numbers)
    pod_diverter_rain_and_area_dicts = []
    for row in pod_diverter_rain_and_area:
        pod_diverter_rain_and_area_dicts.append(dict(row))
    pod_diverter_rain_and_area_dicts = [dict(row) for row in pod_diverter_rain_and_area]
    output = []
    for row in water_rights_dicts:
        output_dict = row
        matching_rain_area_row = next(
                r for r in pod_diverter_rain_and_area_dicts if r['application_number'] == row['application_number']
            )
        output_dict['annual_precip_in'] = matching_rain_area_row['map_1991_2020_in']
        output_dict['drainage_area_sqmi'] = matching_rain_area_row['drainage_area_sqmi']
        for key in output_dict.keys():
            if(type(output_dict[key]) is decimal.Decimal):
                # little bit of type handling
                output_dict[key] = float(output_dict[key])
        output.append(output_dict)
    raw_water_rights_csv_data = sort_and_format_unsorted_csv_data(output, nhd, lat, lng, wsr_session)
    water_rights_csv_data = get_adjusted_csv_data(raw_water_rights_csv_data, watershed)
    water_rights_json = get_intermediate_data_json_formatted(water_rights_csv_data)
    app.db.save_raw_senior_diverters(g.user_id, session_id, water_rights_json)

    if request.mimetype == 'text/csv':
        water_rights_csv = get_wsr_water_rights_csv_formatted(water_rights_csv_data, False)
        app.db.update_wsr_session_by_id(g.user_id, session_id, {'freezeDate': str(date)})
        return Response(water_rights_csv,  mimetype='application/zip'), 200
    else:
        sd_geojson = get_wsr_water_rights_json_formatted(raw_water_rights_csv_data)
        return jsonify(sd_geojson), 200

@points_of_diversion.route('/labels', methods=['GET'])
@authorization_guard
def get_points_of_diversion_labels():
    """
    Get label mapping for points of diversion
    """
    label_mapping = [
        {"color": '#4363d8', "label": 'Upstream of Downstream Flow Path', 'id': 1},
        {"color": '#f58231', "label": 'Upstream of POD', 'id': 2},
        {"color": '#e6194b', "label": 'Proposed POD', 'id': 3},
        {"color": '#3cb44b', "label": 'Downstream Flow Path', 'id': 4},
        {"color": '#ffe119', "label": 'Other Diverters / PODs', 'id': 5},
        {"color": '#0000bd', "label": 'Downstream of POD', 'id': 6},
    ]
    return label_mapping, 200

@points_of_diversion.route('/coordinates', methods=['GET'])
def get_all_point_of_diversion_coordinates():
    """
    Get all point-of-diversion coordinates.
    """
    water_right_coords = [{k: v for k, v in water_right.items() if k in (
        "latitude", "longitude", "pod_id")} for water_right in water_rights_fixture]
    return water_right_coords, 200


@points_of_diversion.route('/<int:id>', methods=['GET'])
@authorization_guard
def get_point_of_diversion(id):
    """
    Get a full length point-of-diversion by id.
    """
    pod = app.db.get_pod_by_pod_id(id)
    if (pod == None):
        raise Exception({'message': 'No pod found for the given id', 'status_code': 404})
    return pod, 200


@points_of_diversion.route('/<int:id>/coordinates', methods=['GET'])
def get_water_right_coords(id):
    """
    Get a single point-of-diverion's coordinates by id.
    """
    water_right_coords = {k: v for k, v in water_right_fixture.items() if k in (
        "latitude", "longitude", "pod_id")}
    return water_right_coords, 200

@points_of_diversion.route('/pod-in-study-area', methods=['GET'])
@authorization_guard
@check_query_params(['lat', 'lng'])
def check_pod_in_study_area(params):
    """
    Uses DB stored study area to check whether a given POD is contained within the area
    """
    (lat, lng) = params
    in_policy_geom = app.db.lat_long_in_policy(lat, lng)
    return {"inArea": in_policy_geom}, 200