generate_watershed_for_poi_and_store_query = """
    WITH INPUT(lat, lng) AS (
        VALUES(%(lat)s, %(lng)s)
    ), point_to_analyse AS (
        SELECT
            ST_POINT(lng, lat, 4326) as pointgeom4326
        FROM
            input
    ), point_catchment AS (
        SELECT
            point_to_analyse.pointgeom4326,
            catchments.geom4326 as catchgeom4326,
            catchments.nhdplusid as nhdplusid
        FROM
            cwat_data.catchments
        JOIN
            point_to_analyse
        ON
            catchments.nhdplusid = %(nhdplusid)s::bigint
    ), split_on_point AS (
        SELECT
            pc.nhdplusid,
            pc.pointgeom4326,
            pc.catchgeom4326,
            nfl.geom4326 as flowlinegeom,
            (ST_Dump(ST_Split(nfl.geom4326,ST_Buffer(ST_CLOSESTPOINT(nfl.geom4326, pc.pointgeom4326), 0.0001)))).path[1] as r,
            (ST_Dump(ST_Split(nfl.geom4326,ST_Buffer(ST_CLOSESTPOINT(nfl.geom4326, pc.pointgeom4326), 0.0001)))).geom AS splitgeom4326
        FROM
            point_catchment pc
        JOIN
            cwat_data.nhdflowline nfl
        USING
            (nhdplusid)
        GROUP BY
            pc.nhdplusid,
            pc.pointgeom4326,
            pc.catchgeom4326,
            nfl.geom4326
    ), point_catchment_split AS (
        -- Note - I thought it'd be simpler to have multiple CTE's and have the query planner parse out whether to use one or the other if we are doing r=3
        -- or no r=3 (up or downstream) case
        SELECT
            sop.nhdplusid,
            sop.pointgeom4326,
            sop.catchgeom4326,
            sop.flowlinegeom,
            ST_Intersection(
                ST_Buffer(
                    ST_LineMerge(
                        ST_COLLECT(
                            ST_MAKELINE(
                                ST_STARTPOINT(sop.splitgeom4326),
                                ST_TRANSLATE(
                                    ST_STARTPOINT(sop.splitgeom4326),
                                    sin(ST_AZIMUTH(ST_POINTN(sop.splitgeom4326,2),ST_STARTPOINT(sop.splitgeom4326))) * -0.1,
                                    cos(ST_AZIMUTH(ST_POINTN(sop.splitgeom4326,2),ST_STARTPOINT(sop.splitgeom4326))) * -0.1
                                )
                            )
                        )
                    )::geography, 10000,
            'endcap=flat'),sop.catchgeom4326) AS split_catchment_ds
        FROM
            split_on_point sop
        WHERE
            r = 3
        GROUP BY
            sop.nhdplusid,
            sop.pointgeom4326,
            sop.catchgeom4326,
            sop.flowlinegeom
    ), split_segment_watershed AS (
        -- note: we could be done here, otherwise we have to handle other cases
        SELECT
            ST_Difference(upstream_geom, ST_Buffer(split_catchment_ds::geometry, 0.0001)) AS upstream_watershed
        FROM
            point_catchment_split
        JOIN
            cwat_data.ws_geoms_all
        USING
            (nhdplusid)
    ),  non_split AS (
        SELECT
            nhdplusid,
            pointgeom4326,
            catchgeom4326,
            flowlinegeom
        FROM
            split_on_point
        EXCEPT
            (SELECT
                nhdplusid,
                pointgeom4326,
                catchgeom4326,
                flowlinegeom
            FROM
                point_catchment_split
            )
    ), non_split_top_intersect AS (
        -- If we are closest closest to the top (start) of the catchment
        SELECT
            nhdplusid,
            pointgeom4326,
            catchgeom4326,
            flowlinegeom
        FROM
            non_split
        WHERE
            ST_DISTANCE(pointgeom4326, ST_STARTPOINT(flowlinegeom)) < ST_DISTANCE(pointgeom4326, ST_ENDPOINT(flowlinegeom))
    ), non_split_bottom_intersect AS (
        -- If we are closest closest to the bottom (start) of the catchment
        SELECT
            nhdplusid,
            pointgeom4326,
            catchgeom4326,
            flowlinegeom
        FROM
            non_split
        WHERE
            ST_DISTANCE(pointgeom4326, ST_STARTPOINT(flowlinegeom)) >= ST_DISTANCE(pointgeom4326, ST_ENDPOINT(flowlinegeom))
    ), bottom_intersect_watershed AS (
        -- Simple case, if at the bottom just take the calculated nhd watershed
        SELECT
            upstream_geom as upstream_watershed
        FROM
            non_split_bottom_intersect
        JOIN
            cwat_data.ws_geoms_all
        USING
            (nhdplusid)
    ), top_intersect_downstream_segments AS(
        -- These pointS need to intersect the top of their stream to make the catchment from that (best we can do I think)
        SELECT
            nhdplusid,
            pointgeom4326,
            catchgeom4326,
            flowlinegeom,
            ST_Intersection(
                ST_Buffer(
                    ST_LineMerge(
                        ST_COLLECT(
                            ST_MAKELINE(
                                ST_STARTPOINT(flowlinegeom),
                                ST_TRANSLATE(
                                    ST_STARTPOINT(flowlinegeom),
                                    sin(ST_AZIMUTH(ST_POINTN(flowlinegeom,2),ST_STARTPOINT(flowlinegeom))) * -0.1,
                                    cos(ST_AZIMUTH(ST_POINTN(flowlinegeom,2),ST_STARTPOINT(flowlinegeom))) * -0.1
                                )
                            )
                        )
                    )::geography, 10000,
            'endcap=flat'),catchgeom4326) AS split_catchment_ds
        FROM
            non_split_top_intersect
        GROUP BY
            nhdplusid,
            catchgeom4326,
            pointgeom4326,
            flowlinegeom
    ), top_intersect_watershed AS (
        SELECT
            ST_Difference(upstream_geom, ST_Buffer(split_catchment_ds::geometry, 0.0001)) AS upstream_watershed
        FROM
            top_intersect_downstream_segments
        JOIN
            cwat_data.ws_geoms_all
        USING
            (nhdplusid)
    ), watershed AS (
        SELECT
            upstream_watershed as watershed_geom4326
        FROM
            split_segment_watershed
        UNION ALL
            (SELECT * FROM bottom_intersect_watershed)
        UNION ALL
            (SELECT * FROM top_intersect_watershed)
    )
    INSERT INTO cwat_app.poi_watersheds
    SELECT
        %(cda_session_id)s::integer as session_id,
        %(poi_id)s::integer as poi_id,
        watershed_geom4326
    FROM
        watershed
    ON CONFLICT (session_id, poi_id)
    DO UPDATE
    SET watershed_geom4326 = excluded.watershed_geom4326;
"""