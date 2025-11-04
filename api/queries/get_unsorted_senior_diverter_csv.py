get_unsorted_senior_diverter_csv_query = """
    WITH proposed (
        nhdplusid
        , lat
        , lng
    ) AS (
        VALUES(%(nhdplusid)s::bigint, %(lat)s, %(lng)s)
    ),  wr_related_to_streams as (

    -- join wr with ordered streams in downstream basin
    SELECT
        -- WATER RIGHT fields:
        distinct on (pod.application_number)
        pod.application_number,
        pod.pod_id,
        pod.wr_water_right_id,
        COALESCE(
            pod.priority_date,
            pod.receipt_date,
            pod.application_acceptance_date
        ) as priority_date,
        pod.use_codes,
        pod.water_right_type,
        pod.water_right_status,
        pod.application_primary_owner,
        pod.appl_pod,
        array_to_string(pod.pod_type, ', ') as pod_type,
        pod.pod_count,
        pod.latitude,
        pod.longitude,
        extract(month from pod.direct_div_season_start) as direct_div_season_start_month,
        extract(day from pod.direct_div_season_start) as direct_div_season_start_day,
        extract(month from pod.direct_div_season_end) as direct_div_season_end_month,
        extract(day from pod.direct_div_season_end) as direct_div_season_end_day,
        extract(month from pod.storage_season_start) as storage_season_start_month,
        extract(day from pod.storage_season_start) as storage_season_start_day,
        extract(month from pod.storage_season_end) as storage_season_end_month,
        extract(day from pod.storage_season_end) as storage_season_end_day,
        pod.max_storage_af,
        CASE
            WHEN coalesce(pod.max_dd_ann_af, 0) > 0 AND coalesce(pod.max_dd_ann_af, 0) < coalesce(pod.face_value_af, 0) THEN coalesce(pod.max_dd_ann_af, 0)
            ELSE coalesce(pod.face_value_af, 0)
        END AS face_amount_af,
        CASE
            WHEN pod.max_diversion_rate_cfs IS NOT NULL THEN pod.max_diversion_rate_cfs
            ELSE pod.direct_diversion_rate_cfs
        END AS max_rate_of_diversion_cfs,
        -- POSITIONAL fields
        CASE
            WHEN nhdids_upstream_of_proposed_pod.up_nhdplusid IS NOT null THEN True
            ELSE False
        END AS in_pod_basin,
        CASE
            WHEN mainstem.nhdplusid IS NOT NULL THEN true
            ELSE False
        END AS mainstem,
        pod.geom4326 as pod_geom,
        CASE
                WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.closest_pt
                ELSE fuzzy_match_name_within_1600m.closest_pt
            END AS closest_pt,
        pod.source_name,
        CASE
                WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.nhdplusid
                ELSE fuzzy_match_name_within_1600m.nhdplusid
            END AS nhdplusid,
        CASE
                WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.gnis_name
                ELSE fuzzy_match_name_within_1600m.gnis_name
            END AS gnis_name,
        CASE
                WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.stream_geom
                ELSE fuzzy_match_name_within_1600m.stream_geom
            END AS stream_geom,
        CASE
                WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.dist
                ELSE fuzzy_match_name_within_1600m.dist
            END AS dist,
        pod.riparian as riparian,
        ST_LineLocatePoint(CASE
                    WHEN fuzzy_match_name_within_1600m.nhdplusid IS NULL THEN nearest_stream.stream_geom
                    ELSE fuzzy_match_name_within_1600m.stream_geom
                    END, pod.geom4326) as st_calc_distance
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
                streams.nhdplusid,
                streams.gnis_name,
                ST_Distance(streams.geog4326, pod.geom4326::geography) as dist,
                ST_ClosestPoint(streams.geom4326, pod.geom4326) as closest_pt,
                streams.geom4326 as stream_geom
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
                streams.nhdplusid,
                streams.gnis_name,
                ST_Distance(streams.geog4326, pod.geom4326::geography) as dist,
                ST_ClosestPoint(streams.geom4326, pod.geom4326) as closest_pt,
                streams.geom4326 as stream_geom
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
            CASE WHEN up.up_nhdplusid is NULL AND proposed.nhdplusid = %(nhdplusid)s THEN %(nhdplusid)s
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
        %(date)s::timestamptz > pod.foundry_date
    ORDER BY
        pod.application_number
    )
    SELECT * from wr_related_to_streams;
"""
