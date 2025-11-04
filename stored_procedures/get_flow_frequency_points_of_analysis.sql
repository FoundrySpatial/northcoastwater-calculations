-- stored procedure definition for cwat_app.get_flow_frequency_points_of_analysis

CREATE OR REPLACE FUNCTION cwat_app.get_flow_frequency_points_of_analysis(
	input_wsr_session_id bigint,
	input_user_id text,
	OUT direct_div_season_start date,
	OUT direct_div_season_end date,
	OUT application_number text,
	OUT seasonal_volume_af numeric,
	OUT rank bigint,
	OUT frequency numeric)
    RETURNS SETOF record
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
BEGIN
    RETURN query WITH wsr_summary AS
    (SELECT * from
        cwat_app.get_wsr_summary_by_session_id (
            input_wsr_session_id::bigint,
            input_user_id::text
))
        , points_of_flow_frequency_analysis AS (
            -- Here we are taking a subset of results that meets any of the following conditions:
            -- Is the proposed project’s POD(s)
            -- Is the senior POAs at which the estimate of unimpaired flow is the lowest
            -- Any other senior PODs at which the ratio is less than 50%
            SELECT
                *
            FROM
                wsr_summary
            WHERE
                analysis_label = 'Proposed POD'
            UNION
            (SELECT
                *
            FROM
                wsr_summary
            WHERE
                analysis_label != 'Proposed POD'
            ORDER BY
                percent_remaining_unappropriated_water_after_new_water_right ASC
            LIMIT 1
             )
        UNION
        SELECT
            *
        FROM
            wsr_summary
        WHERE
            percent_remaining_unappropriated_water_after_new_water_right < 50
        AND
            analysis_label != 'Proposed POD'
)

, user_info AS (
    SELECT
        (wsr_session ->> 'seasonOfDiversionStart')::date AS direct_div_season_start
        , (wsr_session ->> 'seasonOfDiversionEnd')::date AS direct_div_season_end
    FROM
        cwat_app.project_sessions
    WHERE
        id = input_wsr_session_id
)
, proposed_season_in_days AS (
    SELECT
        extract(month FROM generate_series(user_info.direct_div_season_start
                , user_info.direct_div_season_end
                , '1 day')) AS months
        , extract(day FROM generate_series(user_info.direct_div_season_start
                , user_info.direct_div_season_end
                , '1 day')) AS days,
        user_info.direct_div_season_start,
        user_info.direct_div_season_end
FROM
    user_info
)
, yearly_seasonal_volume AS (
    SELECT
        points_of_flow_frequency_analysis.application_number,
        p_days.direct_div_season_start,
        p_days.direct_div_season_end
        , CASE WHEN extract(month FROM d.datestamp) IN (10
            , 11
            , 12) THEN
            extract(year FROM d.datestamp) + 1
        ELSE
            extract(year FROM d.datestamp)
        END AS water_year
        ,
        --cfs*60 seconds * 60 minutes * 24 hours * # days in season
        avg(d.val) * 60 * 60 * 24 * count(*) * 2.29569e-5 * ratio1 AS seasonal_volume_af
    FROM
        cwat_data.stations s
        JOIN cwat_data.water_daily d USING (station_id)
        JOIN proposed_season_in_days p_days ON p_days.months = extract(month FROM d.datestamp)
            AND p_days.days = extract(day FROM d.datestamp)
        JOIN points_of_flow_frequency_analysis ON points_of_flow_frequency_analysis.gage_id = s.station_id
    WHERE
        parameter_id = 1 --currently not needed as there is only 1 parameter_id
    GROUP BY
        CASE WHEN extract(month FROM d.datestamp) IN (10 , 11 , 12) THEN
            extract(year FROM d.datestamp) + 1
        ELSE
            extract(year FROM d.datestamp)
        END , points_of_flow_frequency_analysis.ratio1 , points_of_flow_frequency_analysis.application_number, p_days.direct_div_season_start, p_days.direct_div_season_end
    ORDER BY
        seasonal_volume_af DESC
)
SELECT
    yearly_seasonal_volume.direct_div_season_start,
    yearly_seasonal_volume.direct_div_season_end,
    yearly_seasonal_volume.application_number
    , yearly_seasonal_volume.seasonal_volume_af::numeric
    , RANK() OVER (partition by yearly_seasonal_volume.application_number ORDER BY yearly_seasonal_volume.seasonal_volume_af DESC) AS rank
    , 1 - (RANK() OVER (partition by yearly_seasonal_volume.application_number ORDER BY yearly_seasonal_volume.seasonal_volume_af DESC))::numeric / (1 + count(*) OVER (partition by yearly_seasonal_volume.application_number))::numeric AS frequency
FROM
    yearly_seasonal_volume
GROUP BY
    water_year
    , yearly_seasonal_volume.seasonal_volume_af
    , yearly_seasonal_volume.application_number
    , yearly_seasonal_volume.direct_div_season_start, yearly_seasonal_volume.direct_div_season_end;
END;
$BODY$;

ALTER FUNCTION cwat_app.get_flow_frequency_points_of_analysis(bigint, text)
    OWNER TO foundry;
