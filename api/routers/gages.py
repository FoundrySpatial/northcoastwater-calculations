from flask import Blueprint, current_app as app, request, Response
from authentication.guards import authorization_guard
from utils.api_utils import check_query_params
import pandas as pd
import csv

gages = Blueprint('gages', __name__)
@gages.route('/', methods=['GET'])
@authorization_guard
def get_all_gages():
    """
    Return array of all gages with their upstream watersheds.
    """
    gages = app.db.get_all_gage_names()
    return gages, 200

@gages.route('/candidate-gages', methods=['GET'])
@authorization_guard
@check_query_params(['nhd', 'lat', 'lng'])
def get_project_session_candidate_gages(params):
    """
    Retrieve candidate gages of the proposed point of diversion.
    """
    (nhd, lat, lng) = params

    candidates = app.db.get_candidate_gages_by_nhd_id(nhd, lat, lng)

    if (candidates == None):
        raise Exception({"message": "No candidates found for requested session.", "status_code": 404})

    return candidates, 200

@gages.route('/<int:id>', methods=['GET'])
@authorization_guard
def get_gage_by_id(id):
    """
    Return gage by its id.
    """
    result = app.db.get_gage_by_id(id)

    if (result == None):
        raise Exception({"message": "No gage found. Ensure you are passing a valid id.", "status_code": 404})
    return result, 200

@gages.route('/streamflow-record', methods=['GET'])
@authorization_guard
@check_query_params(['session_id', 'user_id'])
def get_gage_streamflow_by_id(params):
    """
    Return streamflow for the given gage.
    """
    (session_id, user_id) = params
    streamflow_data = app.db.get_gage_streamflow(session_id, user_id)

    if (len(streamflow_data) == 0):
        raise Exception({"message": "No data found for session_id, user_id pair.", "status_code": 404})

    if request.mimetype == 'text/csv':
        summary_csv = pd.DataFrame.from_dict(streamflow_data).to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
        return Response(summary_csv,  mimetype='application/zip'), 200
    else:
        return streamflow_data, 200
