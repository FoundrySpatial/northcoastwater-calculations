get_gage_size_and_mean_precip_query = """
WITH  station_info AS (
    SELECT
        (wsr_session ->> 'selectedGage')::integer AS station_id
    FROM
        cwat_app.project_sessions
    WHERE
        id = %(wsr_session_id)s
)
SELECT
    drainage_area_sqmi,
    map_1991_2020_in
FROM
    cwat_data.stations
JOIN
    station_info
USING
    (station_id)
JOIN
    cwat_data.ws_physical_characteristics
USING
    (nhdplusid)
"""