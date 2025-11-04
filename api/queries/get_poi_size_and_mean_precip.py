get_poi_size_and_mean_precip_query = """
WITH catchment AS (
    SELECT
        ST_MakeValid(watershed_geom4326) as geom4326,
        ST_Centroid(watershed_geom4326) as centroid
    FROM
        cwat_app.poi_watersheds
    WHERE
        session_id = %(cda_session_id)s
    AND
        poi_id = %(poi_id)s
), p_1991 as (
    SELECT
        geom4326,
        CASE
            WHEN (st_summarystats(rast)).mean IS NULL THEN catchment_val
            ELSE (st_summarystats(rast)).mean
        END AS avg_pp
    FROM
        (
        SELECT
            geom4326,
            ST_Clip(r.rast, catchment.geom4326) as rast,
            ST_Value(r.rast, centroid) as catchment_val
        FROM
            catchment
        JOIN
            cwat_staging.prism_ppt_30yr_normal_1991_2020_800m_annual r
        ON
            ST_Intersects(catchment.geom4326, r.rast)
        where
            ST_Clip(r.rast, catchment.geom4326) is not null
        ) clipped_raster
    WHERE
        catchment_val is not null
)
SELECT
p_1991.avg_pp* 0.0393701 as map_1991_2020_in,
ST_AREA(geom4326::geography)* 0.00000038610 as drainage_area_sqmi
FROM
p_1991
"""