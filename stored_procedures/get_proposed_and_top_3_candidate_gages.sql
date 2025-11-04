-- stored procedure definition for cwat_app.get_flow_frequency_points_of_analysis

CREATE OR REPLACE FUNCTION cwat_app.get_proposed_and_top_3_candidate_gages(
	input_nhd_id bigint,
	input_lat double precision,
	input_lng double precision)
    RETURNS SETOF json
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
BEGIN
    RETURN query
	WITH user_info(
		nhdplusid,
		lat,
		lon
	) AS (
		VALUES(input_nhd_id, input_lat, input_lng)
	), proposed_and_cands as (
		SELECT
			'Candidate ' || cands.rank as watershed_label,
			cands.rank as rank,
		 	s.nhdplusid,
		 	s.site_no as site_number,
		 	s.station_name,
			s.station_id,
			cands.distance,
		 	number_of_full_years as complete_water_years,
		 	full_water_year_start || ' - ' || full_water_year_end as historic_water_record,
		    json_build_object(
				'units', 'mi²',
				'value', phy.area_sqmi
				) as upstream_area,
		    json_build_object(
				'units', 'in',
				'value', phy.map_1991_2020_in
				) as annual_mean_precip,
			json_build_object(
						'type', 'Feature',
						'geometry', ST_AsGeoJSON(s.geom4326, 6)::jsonb
			 ) as site_location,
		 	phy.gage_selection_params,
			phy.monthly_precip
		FROM
			(select * from cwat_data.ungaged_candidates JOIN user_info using (nhdplusid)) cands
		JOIN
			cwat_data.stations s
		ON
			cands.station_id = s.station_id
		JOIN
		 	cwat_data.ws_physical_characteristics phy
		ON
			s.nhdplusid = phy.nhdplusid
		UNION ALL
		SELECT
			'Proposed Watershed' as watershed_label,
		 	NULL::int as rank,
			user_info.nhdplusid,
		 	NULL::bigint as site_number,
		 	NULL::text as station_name,
			NULL::integer as station_id,
			NULL::real as distance,
		 	NULL::int as complete_water_years,
		 	NULL::text as historic_water_record,
		    json_build_object(
				'units', 'mi²',
				'value', phy.area_sqmi
				) as upstream_area,
		    json_build_object(
				'units', 'in',
				'value', phy.map_1991_2020_in
				) as annual_mean_precip,
			json_build_object(
						'type', 'Feature',
						'geometry', ST_AsGeoJSON(ST_SetSRID(ST_Point(user_info.lon, user_info.lat), 4326), 6)::jsonb
			) as site_location,
		 	phy.gage_selection_params,
			phy.monthly_precip
		FROM
			user_info
		JOIN
		 	cwat_data.ws_physical_characteristics phy
		USING
			(nhdplusid)
		)
		SELECT
			json_agg(v.*)
		FROM
			proposed_and_cands v;
END;
$BODY$;