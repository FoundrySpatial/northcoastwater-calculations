from flask import Blueprint, current_app as app
from authentication.guards import authorization_guard

streampath = Blueprint('streampath', __name__)

@streampath.route('/<int:id>', methods=['GET'])
@authorization_guard
def get_streampath(id):
    """
    Get the streampath geometry for a given nhd id. Note that the streampath is the full downstream path from the nhd id,
    not to be confused with a stream reach which is just the segment for an nhdid.
    """
    stream = app.db.get_streampath_by_nhd_id(id)
    if stream == None:
        raise Exception({"message": "No streampath found. Please ensure you are using a valid nhd id.", "status_code": 404})
    
    return stream, 200
