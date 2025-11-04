from flask import Blueprint, jsonify, current_app as app
from utils.api_utils import check_query_params, read_json, get_compressed_response
from authentication.guards import authorization_guard

watersheds = Blueprint('watersheds', __name__)

@watersheds.route('/<int:nhd_id>', methods=['GET'])
@authorization_guard
@check_query_params(['downstream'], False)
def get_watershed(params, nhd_id):
    """
    Get a watershed for the given nhd_id.

    :param str downstream: If included, return the most downstream watershed instead of nearest.
    """
    (
    downstream,
    ) = params

    found_watershed = None

    if downstream is not None:
        found_watershed = app.db.get_downstream_watershed_by_nhd_id(nhd_id)
    else:
        found_watershed = app.db.get_watershed_by_nhd_id(nhd_id)

    found_watershed = found_watershed['watershed']

    response = get_compressed_response(found_watershed)
    return response, 200
