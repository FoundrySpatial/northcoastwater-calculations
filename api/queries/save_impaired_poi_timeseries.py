save_impaired_poi_timeseries_query = """
UPDATE
    cwat_app.daily_flow_data
SET
    diverters_impaired_poi_data = %(diverters)s,
    pod_impaired_poi_data = %(diverters_with_pod)s
WHERE
    cda_session_id = %(id)s AND poi_id = %(poi_id)s;
"""