-- stored procedure definition for cwat_app.get_wsr_gage_initial_calculations

CREATE OR REPLACE FUNCTION cwat_app.get_wsr_gage_initial_calculations(
	in_session_id integer,
	in_user_id text)
    RETURNS TABLE(site_no bigint, area_sqmi double precision, map_1991_2020_in real, diversion_season text, seasonal_flow_af double precision)
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$
BEGIN
    RETURN query
	WITH user_info(gage_id, seasonofdiversionstart, seasonofdiversionend) AS (
			SELECT
				(wsr_session ->> 'selectedGage')::bigint as gage_id,
			    (wsr_session ->> 'seasonOfDiversionStart')::date as season_start,
			    (wsr_session ->> 'seasonOfDiversionEnd')::date as season_end
			FROM cwat_app.project_sessions where id = in_session_id and user_id = in_user_id
		), proposed_season_in_days as (
			SELECT
				extract(month from generate_series(seasonofdiversionstart, seasonofdiversionend, '1 day')) as months,
				extract(day from generate_series(seasonofdiversionstart, seasonofdiversionend, '1 day')) as days
			FROM
				user_info
		), multi_year_monthly_volume as (
		SELECT
			extract(year from d.datestamp) as year,
			extract(month from d.datestamp) as month,
			sum(d.val*60*60*24) as flow_cubic_feet_per_month
		FROM
			cwat_data.stations s
		JOIN
			user_info
		ON
            s.station_id = user_info.gage_id
		JOIN
			cwat_data.water_daily d
		USING
			(station_id)
		JOIN
			proposed_season_in_days p_days
		ON
			p_days.months = extract(month from d.datestamp)
		AND
			p_days.days = extract(day from d.datestamp)
		WHERE
			parameter_id = 1 --currently not needed as there is only 1 parameter_id
		group by
			extract(year from d.datestamp),
			extract(month from d.datestamp)
		), mean_monthly_volume as (
		SELECT
			month,
			avg(flow_cubic_feet_per_month)*2.29569e-5 as monthly_avg
		FROM
			multi_year_monthly_volume
		GROUP BY
			month
		), seasonal_flow_volume as (
		SELECT
			sum(monthly_avg) as seasonal_flow_af
		FROM
			mean_monthly_volume
		)
			SELECT
				gage.site_no::bigint,
				up_params.area_sqmi,
				up_params.map_1991_2020_in,
				to_char(seasonofdiversionstart, 'Mon DD') || ' - ' || to_char(seasonofdiversionend, 'Mon DD') as diversion_season,
				seasonal_flow_volume.seasonal_flow_af
			FROM
				user_info
			JOIN
				cwat_data.stations gage
            ON
                gage.station_id = user_info.gage_id
			JOIN
				cwat_data.ws_physical_characteristics up_params
			ON
				gage.nhdplusid = up_params.nhdplusid
			CROSS JOIN
				seasonal_flow_volume;
END;
$BODY$;