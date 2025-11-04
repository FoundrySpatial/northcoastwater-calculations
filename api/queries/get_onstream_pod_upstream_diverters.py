get_onstream_pod_upstream_diverters_query = f"""
WITH unspooled_json AS (
	SELECT
		json_arr->>'order_upstream_to_downstream' as order_upstream_to_downstream,
		json_arr->>'lat' as lat,
		json_arr->>'lng' as lng
	FROM
		json_array_elements(%(current_upstream_diverters)s::json) json_arr
), pod_watershed AS (
	SELECT
		watershed_geom4326
	FROM
		cwat_data.diverter_watershed_geoms dwg
	JOIN
		cwat_data.wr_pods wp
	ON
		dwg.pod_id = wp.id
	WHERE
		wp.wr_water_right_id = %(water_right_id)s
)
SELECT
	order_upstream_to_downstream
FROM
	unspooled_json
JOIN
	pod_watershed
ON 
	ST_Contains(pod_watershed.watershed_geom4326, ST_POINT(unspooled_json.lng::numeric, unspooled_json.lat::numeric, 4326));
"""

