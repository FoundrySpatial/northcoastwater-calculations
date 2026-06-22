get_session_information_query = f"""
WITH sessionInfo AS (
        SELECT
        	(wsr_session ->> 'selectedGage')::int as station_id,
			(wsr_session ->> 'seasonOfDiversionStart')::date as season_start,
			(wsr_session ->> 'seasonOfDiversionEnd')::date as season_end,
            (wsr_session -> 'meta' ->> 'created')::timestamp with time zone as created,
            (wsr_session -> 'meta' ->> 'modified')::timestamp with time zone as modified,
            (wsr_session ->> 'nhdId')::bigint as nhdID,
            (wsr_session ->> 'title')::text as title,
            (wsr_session ->> 'description')::text as descr,
            (wsr_session ->> 'requiresCda')::boolean as cdaBool,
            (wsr_session -> 'rateOfDiversion' ->> 'unit')::text as rodUnit,
            (wsr_session -> 'rateOfDiversion' ->> 'value')::real as rodVal,
            (wsr_session -> 'pointOfDiversion' -> 'geometry' ->> 'coordinates')::text as PODCoord,
            (wsr_session -> 'volumeOfDiversion' ->> 'unit')::text as vodUnit,
            (wsr_session -> 'volumeOfDiversion' ->> 'value')::real as vodVal,
			(wsr_session -> 'freezeDate')::text as freeze,
			(wsr_session ->> 'watershedArea')::float as watershedArea
        FROM
            cwat_app.project_sessions
        WHERE
            project_sessions.user_id = %(user_id)s
            AND
            project_sessions.id = %(session_id)s
        )
    SELECT
        stations.site_no,
        stations.station_name,
        stations.full_water_year_start,
        stations.full_water_year_end,
        stations.full_year_array,
        stations.number_of_full_years,
        phys.area_sqmi,
        sessionInfo.season_start,
        sessionInfo.season_end,
        sessionInfo.created,
        sessionInfo.modified,
        sessionInfo.nhdID,
        sessionInfo.title,
        sessionInfo.cdaBool,
        sessionInfo.rodUnit,
        sessionInfo.rodVal,
        sessionInfo.PODCoord,
        sessionInfo.vodUnit,
        sessionInfo.vodVal,
        sessionInfo.descr,
        sessionInfo.watershedArea as podarea,
        waterSheds.gnis_name,
        csv.when_raw_senior_diverters_modified as rawmodified,
        csv.when_csv_with_intermedidate_values_modified as intmodified,
		sessionInfo.freeze
    FROM
        sessionInfo
    JOIN
        cwat_data.stations as stations
    ON
        sessionInfo.station_id = stations.station_id
    JOIN
        cwat_data.ws_physical_characteristics as phys
    ON
        stations.nhdplusid = phys.nhdplusid
    JOIN
        cwat_data.ws_geoms_all as waterSheds
    ON
        sessionInfo.nhdID = waterSheds.nhdplusid
    JOIN
        cwat_app.senior_diverter_csv as csv
    ON
        csv.user_id = %(user_id)s
        AND
        csv.wsr_session_id = %(session_id)s;
"""