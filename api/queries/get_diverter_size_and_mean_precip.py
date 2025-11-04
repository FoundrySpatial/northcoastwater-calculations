get_diverter_size_and_mean_precip_query = """
    SELECT
        wr_pods.application_number,
        dwg.drainage_area_sqmi,
        dwg.map_1991_2020_in
    FROM
        cwat_data.wr_pods
    JOIN
        cwat_data.diverter_watershed_geoms dwg
    ON
        dwg.pod_id = wr_pods.id
    WHERE
        wr_pods.application_number = ANY(%(application_numbers)s)
"""