-- stored procedure definition for cwat_app.get_raw_senior_diverters_by_session_id

CREATE OR REPLACE FUNCTION cwat_app.get_raw_senior_diverters_by_session_id(
	input_nhd_id bigint,
	input_lat double precision,
	input_lng double precision,
	date_limit timestamp with time zone,
	OUT analysis_label character varying,
	OUT order_upstream_to_downstream integer,
	OUT analysis_label_map smallint,
	OUT application_number character varying,
	OUT application_primary_owner character varying,
	OUT appl_pod character varying,
	OUT nhdplusid bigint,
	OUT source_name character varying,
	OUT wr_water_right_id bigint,
	OUT year_diversion_commenced smallint,
	OUT water_right_type character varying,
	OUT water_right_status character varying,
	OUT use_codes text[],
	OUT pod_type character varying,
	OUT pod_count integer,
	OUT latitude double precision,
	OUT longitude double precision,
	OUT drainage_area_sqmi double precision,
	OUT annual_precip_in double precision,
	OUT direct_div_season_start_month smallint,
	OUT direct_div_season_start_day smallint,
	OUT direct_div_season_end_month smallint,
	OUT direct_div_season_end_day smallint,
	OUT storage_season_start_month smallint,
	OUT storage_season_start_day smallint,
	OUT storage_season_end_month smallint,
	OUT storage_season_end_day smallint,
	OUT max_storage_af double precision,
	OUT face_amount_af double precision,
	OUT max_rate_of_diversion_cfs double precision,
	OUT minimum_bypass_flow_cfs double precision,
	OUT riparian boolean,
	OUT comments text,
	OUT seasonal_demand_af double precision,
	OUT overwrite_seasonal_demand_af_justification text)
    RETURNS SETOF record
    LANGUAGE 'sql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 8000

AS $BODY$
WITH proposed (
	nhdplusid
	, lat
	, lng
) AS (
	VALUES(input_nhd_id, input_lat, input_lng)
),  wr_related_to_streams as (

-- join wr with ordered streams in downstream basin
SELECT
	-- WATER RIGHT fields:
	distinct on (pod.application_number)
	pod.application_number
	, pod.pod_id
	, pod.wr_water_right_id
	, pod.year_diversion_commenced
	, pod.use_codes
	, pod.water_right_type
	, pod.water_right_status
	, pod.application_primary_owner
	, pod.appl_pod
	, array_to_string(pod.pod_type, ', ') as pod_type
	, pod.pod_count
	, pod.latitude
	, pod.longitude
	, extract(month from pod.direct_div_season_start) as direct_div_season_start_month
	, extract(day from pod.direct_div_season_start) as direct_div_season_start_day
	, extract(month from pod.direct_div_season_end) as direct_div_season_end_month
	, extract(day from pod.direct_div_season_end) as direct_div_season_end_day
	, extract(month from pod.storage_season_start) as storage_season_start_month
	, extract(day from pod.storage_season_start) as storage_season_start_day
	, extract(month from pod.storage_season_end) as storage_season_end_month
	, extract(day from pod.storage_season_end) as storage_season_end_day
	, pod.max_storage_af
	, CASE
		WHEN coalesce(pod.max_dd_ann_af, 0) > 0 AND coalesce(pod.max_dd_ann_af, 0) < coalesce(pod.face_value_af, 0) THEN coalesce(pod.max_dd_ann_af, 0)
		ELSE coalesce(pod.face_value_af, 0)
	END AS face_amount_af
	, CASE
		WHEN pod.max_diversion_rate_cfs IS NOT NULL THEN pod.max_diversion_rate_cfs
		ELSE pod.direct_diversion_rate_cfs
	END AS max_rate_of_diversion_cfs
	-- POSITIONAL fields
	, CASE
		WHEN nhdids_upstream_of_proposed_pod.up_nhdplusid IS NOT null THEN True
		ELSE False
	 END AS in_pod_basin
	, CASE
		WHEN mainstem.nhdplusid IS NOT NULL THEN true
		ELSE False
	END AS mainstem
	, pod.geom4326 as pod_geom
	,	CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.closest_pt
			ELSE fuzzy_match_name_within_1600m.closest_pt
		END AS closest_pt
	, pod.source_name
	,	CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.nhdplusid
			ELSE fuzzy_match_name_within_1600m.nhdplusid
		END AS nhdplusid
	,	CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.gnis_name
			ELSE fuzzy_match_name_within_1600m.gnis_name
		END AS gnis_name
	,	CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.stream_geom
			ELSE fuzzy_match_name_within_1600m.stream_geom
		END AS stream_geom
	,	CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.dist
			ELSE fuzzy_match_name_within_1600m.dist
		END AS dist
    , pod.riparian as riparian
	, phy.area_sqmi
FROM
	cwat_data.wr_pods pod
JOIN
-- return wr within the upstream geometry of downstream watershed
(
	SELECT
		up.upstream_geom
	FROM
		cwat_data.downstream_ids
	JOIN
		proposed
	USING
		(nhdplusid)
	JOIN
		cwat_data.ws_geoms_all up
	ON
		ds_nhdplusid = up.nhdplusid
) ds_geom
ON
	ST_Intersects(ds_geom.upstream_geom, pod.geom4326)
-- JOIN wr with streams that fuzzy match stream name and are within 1 mile / 1.6 km
LEFT JOIN LATERAL
	(
		SELECT
			streams.nhdplusid
			, streams.gnis_name
			, ST_Distance(streams.geog4326, pod.geom4326::geography) as dist
			, ST_ClosestPoint(streams.geom4326, pod.geom4326) as closest_pt
			, streams.geom4326 as stream_geom
		FROM
			cwat_data.nhdflowline streams
		WHERE
			(
				ST_DWithin(streams.geog4326, pod.geom4326::geography, 1600)
				AND
				soundex(pod.source_name) = soundex(streams.gnis_name)
			)
		AND
			in_study_area
		ORDER BY dist
		LIMIT 1
	) fuzzy_match_name_within_1600m on true
-- JOIN wr with nearest streams
LEFT JOIN LATERAL
	(
		SELECT
			streams.nhdplusid
			, streams.gnis_name
			, ST_Distance(streams.geog4326, pod.geom4326::geography) as dist
			, ST_ClosestPoint(streams.geom4326, pod.geom4326) as closest_pt
			, streams.geom4326 as stream_geom
		FROM
			cwat_data.nhdflowline streams
		WHERE
			in_study_area
		ORDER BY
			streams.geom4326 <-> pod.geom4326
		LIMIT 1
	) nearest_stream on true
LEFT JOIN
-- get nhdids upstream of proposed POD
-- JOIN with snapped stream
-- NOT joining with pod geom as after its been snapped, it may no longer be in basin
(
	SELECT
		CASE WHEN up.up_nhdplusid is NULL AND proposed.nhdplusid = input_nhd_id THEN input_nhd_id
		ELSE up.up_nhdplusid
		END
	FROM
		cwat_data.upids up
	JOIN
		proposed
	USING
		(nhdplusid)
) nhdids_upstream_of_proposed_pod
ON
	nhdids_upstream_of_proposed_pod.up_nhdplusid = CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.nhdplusid
			ELSE fuzzy_match_name_within_1600m.nhdplusid
		END
LEFT JOIN
cwat_data.ws_physical_characteristics phy
ON
phy.nhdplusid = CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.nhdplusid
			ELSE fuzzy_match_name_within_1600m.nhdplusid
		END
LEFT JOIN
		(
		SELECT
			unnest(ds_nhdplusids) AS nhdplusid
		FROM
			cwat_data.downstream_flow_paths
		JOIN
			proposed
		USING
			(nhdplusid)
		) mainstem
ON
mainstem.nhdplusid = CASE
			WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.nhdplusid
			ELSE fuzzy_match_name_within_1600m.nhdplusid
		END
WHERE
	date_limit > pod.foundry_date
ORDER BY
	pod.application_number
)
-- Get the most downstream WR on mainstem
-- if there is no water rights on the mainstem, return the proposed pod nhd
, most_downstream_wr as (
SELECT
	CASE
		WHEN wr.nhdplusid IS NULL THEN proposed.nhdplusid
		ELSE wr.nhdplusid
	END AS nhdplusid
FROM
	proposed
LEFT JOIN
	(SELECT
	 	nhdplusid
	 FROM
	 	wr_related_to_streams
	 WHERE
		mainstem
	 ORDER BY
		area_sqmi DESC
	 LIMIT 1) wr
	 on true
)
, ordered_nhds_above_most_downstream_wr_on_mainstem as (
		SELECT
			d.nhdplusid
			-- Using the mainstem depth to get the order, with tree depth tie-breaking
-- 			, analysis_label
			, analysis_label_map
			, is_mainstem
			, tree_depth
			, mainstem_depth
			, geom4326
			, geog4326
			, gnis_name
			, nhd_order
	FROM
		most_downstream_wr
	CROSS JOIN
		proposed
	CROSS JOIN
		cwat_app.get_ordered_nhds_upstream_of_nhd_id(most_downstream_wr.nhdplusid, proposed.nhdplusid) d
)
, wr_records_upstream_of_proposed_basin as (
SELECT
    application_number
    , pod_id
	, wr_water_right_id
	, year_diversion_commenced
	, use_codes
    , water_right_type
    , water_right_status
    , application_primary_owner
    , appl_pod
    , pod_type
    , pod_count
    , latitude
    , longitude
    , direct_div_season_start_month
    , direct_div_season_start_day
    , direct_div_season_end_month
    , direct_div_season_end_day
    , storage_season_start_month
    , storage_season_start_day
    , storage_season_end_month
    , storage_season_end_day
    , max_storage_af
    , face_amount_af
    , max_rate_of_diversion_cfs
    , in_pod_basin
    , pod_geom
    , closest_pt
    , source_name
    , nhdplusid
    , p.gnis_name
    , stream_geom
    , dist
    , 2::int as analysis_label_map -- Upstream of POD
    , nhd_order
    , riparian
FROM
	wr_related_to_streams p
LEFT JOIN
	ordered_nhds_above_most_downstream_wr_on_mainstem nhd
USING
	(nhdplusid)
WHERE
	in_pod_basin
), pod_record as (
	SELECT
    NULL as application_number
    , NULL::int as pod_id
	, NULL::bigint as wr_water_right_id
	, NULL::smallint as year_diversion_commenced
	, NULL::text[] as use_codes
    , NULL as water_right_type
    , NULL as water_right_status
    , NULL as application_primary_owner
    , NULL as appl_pod
    , NULL as pod_type
    , NULL::int as pod_count
    , p.lat as latitude
    , p.lng as longitude
    , NULL::int as direct_div_season_start_month
    , NULL::int as direct_div_season_start_day
    , NULL::int as direct_div_season_end_month
    , NULL::int as direct_div_season_end_day
    , NULL::int as storage_season_start_month
    , NULL::int as storage_season_start_day
    , NULL::int as storage_season_end_month
    , NULL::int as storage_season_end_day
    , NULL::double precision as max_storage_af
    , NULL::double precision as face_amount_af
    , NULL::double precision as max_rate_of_diversion_cfs
    , True as in_pod_basin
    , ST_SetSRID(ST_Point(p.lng, p.lat), 4326) as pod_geom
    , ST_ClosestPoint(streams.geom4326, ST_SetSRID(ST_Point(p.lng, p.lat), 4326)) as closest_pt
    , NULL as source_name
    , p.nhdplusid
    , streams.gnis_name
    , streams.geom4326 as stream_geom
    , NULL::double precision as dist
    , 3::smallint analysis_label_map -- Proposed POD
    , CASE
		WHEN f.nhdplusid = p.nhdplusid then f.nhd_order
		ELSE COALESCE(f.nhd_order, 0)+1
	END as nhd_order
FROM
		proposed p
LEFT JOIN LATERAL
	(SELECT nhd_order, nhdplusid from wr_records_upstream_of_proposed_basin order by nhd_order desc LIMIT 1) f ON TRUE
 -- join with proposed to see if the stream reach that the pod is on has any other WRs snapped to it
 -- in the case where a WR in the proposed basin is snapped to the same stream reach the proposed POD is on
 -- then assign the proposed POD the same nhd_order as the max in proposed
 -- this will later be caught and reordered & labeled properly in lower query
	JOIN
		cwat_data.nhdflowline streams
	on
		streams.nhdplusid = p.nhdplusid
)
, remaining_wr_records as (
SELECT
    application_number
    , pod_id
	, wr_water_right_id
	, year_diversion_commenced
	, use_codes
    , water_right_type
    , water_right_status
    , application_primary_owner
    , appl_pod
    , pod_type
    , pod_count
    , latitude
    , longitude
    , direct_div_season_start_month
    , direct_div_season_start_day
    , direct_div_season_end_month
    , direct_div_season_end_day
    , storage_season_start_month
    , storage_season_start_day
    , storage_season_end_month
    , storage_season_end_day
    , max_storage_af
    , face_amount_af
    , max_rate_of_diversion_cfs
    , in_pod_basin
    , pod_geom
    , closest_pt
    , source_name
    , nhdplusid
    , p.gnis_name
    , stream_geom
    , dist
    , riparian
	, CASE
        WHEN nhd.analysis_label_map IS NULL THEN 5 -- 'Within Project Extent'/'In Study Area'
        ELSE nhd.analysis_label_map
    END as analysis_label_map
    , COALESCE(in_pod_basin.max_nhd_order, 0) + 1 + nhd_order as nhd_order
FROM
	wr_related_to_streams p
CROSS JOIN
	(SELECT max(nhd_order) as max_nhd_order FROM wr_records_upstream_of_proposed_basin) in_pod_basin
LEFT JOIN
	ordered_nhds_above_most_downstream_wr_on_mainstem nhd
USING
	(nhdplusid)
WHERE
	not in_pod_basin
), wr_3_cases_appended as (
SELECT
	analysis_label_map
	, nhd_order
    , in_pod_basin
    , pod_geom
    , closest_pt
    , source_name
    , nhdplusid
    , gnis_name
    , stream_geom
    , dist
	, application_number
    , pod_id
    , water_right_type
    , water_right_status
    , application_primary_owner
    , appl_pod
    , pod_type
    , pod_count
    , latitude
    , longitude
    , direct_div_season_start_month
    , direct_div_season_start_day
    , direct_div_season_end_month
    , direct_div_season_end_day
    , storage_season_start_month
    , storage_season_start_day
    , storage_season_end_month
    , storage_season_end_day
    , max_storage_af
    , face_amount_af
    , max_rate_of_diversion_cfs
	, wr_water_right_id
	, year_diversion_commenced
	, use_codes
    , riparian
FROM
wr_records_upstream_of_proposed_basin
UNION
SELECT
	analysis_label_map
	, nhd_order
    , in_pod_basin
    , pod_geom
    , closest_pt
    , source_name
    , nhdplusid
    , gnis_name
    , stream_geom
    , dist
	, application_number
    , pod_id
    , water_right_type
    , water_right_status
    , application_primary_owner
    , appl_pod
    , pod_type
    , pod_count
    , latitude
    , longitude
    , direct_div_season_start_month
    , direct_div_season_start_day
    , direct_div_season_end_month
    , direct_div_season_end_day
    , storage_season_start_month
    , storage_season_start_day
    , storage_season_end_month
    , storage_season_end_day
    , max_storage_af
    , face_amount_af
    , max_rate_of_diversion_cfs
	, wr_water_right_id
	, year_diversion_commenced
	, use_codes
    , false::boolean as riparian
FROM
pod_record
UNION
SELECT
	analysis_label_map
	, nhd_order
    , in_pod_basin
    , pod_geom
    , closest_pt
    , source_name
    , nhdplusid
    , gnis_name
    , stream_geom
    , dist
	, application_number
    , pod_id
    , water_right_type
    , water_right_status
    , application_primary_owner
    , appl_pod
    , pod_type
    , pod_count
    , latitude
    , longitude
    , direct_div_season_start_month
    , direct_div_season_start_day
    , direct_div_season_end_month
    , direct_div_season_end_day
    , storage_season_start_month
    , storage_season_start_day
    , storage_season_end_month
    , storage_season_end_day
    , max_storage_af
    , face_amount_af
    , max_rate_of_diversion_cfs
	, wr_water_right_id
	, year_diversion_commenced
	, use_codes
    , riparian
FROM
remaining_wr_records
), combined_wr as (
SELECT
	row_number() over(order by appended.nhd_order, ST_LineLocatePoint(appended.stream_geom, appended.pod_geom))::int as order_upstream_to_downstream
	, appended.analysis_label_map
	, appended.nhd_order
    , appended.in_pod_basin
    , appended.pod_geom
    , appended.closest_pt
    , appended.source_name
    , appended.nhdplusid
    , appended.gnis_name
    , appended.stream_geom
    , appended.dist
	, appended.application_number
    , appended.pod_id
    , appended.water_right_type
    , appended.water_right_status
    , appended.application_primary_owner
    , appended.appl_pod
    , appended.pod_type
    , appended.pod_count
    , appended.latitude
    , appended.longitude
    , appended.direct_div_season_start_month
    , appended.direct_div_season_start_day
    , appended.direct_div_season_end_month
    , appended.direct_div_season_end_day
    , appended.storage_season_start_month
    , appended.storage_season_start_day
    , appended.storage_season_end_month
    , appended.storage_season_end_day
    , appended.max_storage_af
    , appended.face_amount_af
    , appended.max_rate_of_diversion_cfs
	, appended.wr_water_right_id
	, appended.year_diversion_commenced
	, appended.use_codes
    , appended.riparian
	, phy.area_sqmi as drainage_area_sqmi
	, phy.map_1991_2020_in as annual_precip_in
FROM
	wr_3_cases_appended appended
JOIN
	cwat_data.ws_physical_characteristics phy
USING
	(nhdplusid)
)
SELECT
	CASE
		WHEN analysis_label_map = 2 and order_upstream_to_downstream > (select order_upstream_to_downstream from combined_wr where analysis_label_map = 3) THEN 'Mainstem POA'::text
		WHEN analysis_label_map = 1 THEN 'Upstream of Mainstem POA'
		WHEN analysis_label_map = 2 THEN 'Upstream of POD'
		WHEN analysis_label_map = 3 THEN 'Proposed POD'
		WHEN analysis_label_map  = 4 THEN 'Mainstem POA'
		-- known cases of water rights that are in the tributaries between the most downstream water right
		-- and the pacific ocean or the next regulated river
		WHEN analysis_label_map  = 5 THEN 'Inside Project Extent'
	  END AS analysis_label -- eventually deprecating
	, order_upstream_to_downstream
-- special case when there are multiple WR potentially upstream and downstream of proposed POD location snapped to the same stream reach.
-- relabel the WR that are on the same stream reach as proposed POD location but downstream to 'Mainstem POA'
	, CASE
		WHEN analysis_label_map = 2 and order_upstream_to_downstream > (select order_upstream_to_downstream from combined_wr where analysis_label_map = 3) THEN 4::smallint
		ELSE analysis_label_map::smallint
	END AS analysis_label_map
    , application_number::varchar(15)
    , application_primary_owner::varchar(100)
    , appl_pod::varchar(25)
    , nhdplusid::bigint
    , source_name::varchar(55)
	, wr_water_right_id::bigint
	, year_diversion_commenced::smallint
    , water_right_type::varchar(28)
    , water_right_status::varchar(25)
	, use_codes::text[]
    , pod_type:: varchar(150)
    , pod_count::int
    , latitude::double precision
    , longitude::double precision
	, drainage_area_sqmi::double precision
	, annual_precip_in::double precision
    , direct_div_season_start_month::smallint
    , direct_div_season_start_day::smallint
    , direct_div_season_end_month::smallint
    , direct_div_season_end_day::smallint
    , storage_season_start_month::smallint
    , storage_season_start_day::smallint
    , storage_season_end_month::smallint
    , storage_season_end_day::smallint
    , max_storage_af::double precision
    , face_amount_af::double precision
    , max_rate_of_diversion_cfs::double precision
	, NULL::double precision as minimum_bypass_flow_cfs
    , riparian::boolean
	, NULL::text as comments
	, NULL::double precision as seasonal_demand_af
	, NULL::text as overwrite_seasonal_demand_af_justification
FROM
	combined_wr;
$BODY$;
