from flask import Blueprint, current_app as app
from authentication.guards import authorization_guard

stream_reach = Blueprint('streamreach', __name__)

@stream_reach.route('/<int:id>', methods=['GET'])
@authorization_guard
def get_streamreach(id):
    """
    Get the stream reach for a given nhdid. Returns the validity of the reach.
    """
    stream = app.db.verify_streamreach(id)
    if stream == None:
        raise Exception({"message": "No streampath found. Please ensure you are using a valid nhd id.", "status_code": 404})
    if(stream.get('message') == "Flow-regulated mainstem river"):
        stream['message'] = "A Water Availability Analysis is not supported on the selected stream reach as it is considered flow regulated in practice. Please consult with the Water Board staff before proceeding to develop a Water Availability Analysis for water right application purposes."
    return stream, 200