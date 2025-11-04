-- stored procedure definition for cwat_app.unimpair_gage_timeseries

CREATE OR REPLACE FUNCTION cwat_app.unimpair_gage_timeseries(
	input_data jsonb,
	user_id_in text,
	session_id_in bigint)
    RETURNS text
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
BEGIN
        WITH session_gage_id AS(
		SELECT wsr_session ->> 'selectedGage' as gid
		FROM cwat_app.project_sessions
		WHERE user_id = user_id_in and id = session_id_in
		LIMIT 1
	), gage_data AS (
		SELECT wd.datestamp, wd.val
 		FROM cwat_data.water_daily as wd
		JOIN session_gage_id
		ON session_gage_id.gid::bigint = wd.station_id
	), json_table AS (
    	SELECT *
        FROM cwat_app.fill_missing_years(input_data)
		), gage_data_julian AS (
		SELECT
			datestamp,
			val,
			CASE
				WHEN EXTRACT(MONTH FROM datestamp) >= 10 THEN
					EXTRACT(DOY FROM datestamp) - EXTRACT(DOY FROM DATE_TRUNC('year', datestamp) + INTERVAL '9 months') + 1
				ELSE
					EXTRACT(DOY FROM datestamp) - EXTRACT(DOY FROM DATE_TRUNC('year', datestamp) - INTERVAL '3 months') + 366
			END AS julian_date,
			CASE
				WHEN EXTRACT(MONTH FROM datestamp) >= 10 THEN
					(EXTRACT(YEAR FROM datestamp))::int
				ELSE
					(EXTRACT(YEAR FROM datestamp) - 1)::int
			END AS water_year
		FROM
			gage_data
	), unimpaired_gage_table as (
		SELECT
			gd.datestamp,
			gd.val + jt.datac[gd.julian_date] as unimpaired_value
		FROM
			gage_data_julian as gd
		JOIN
			json_table as jt
		ON
			jt.datec = gd.water_year
	), formatted_json as (
		select jsonb_agg(jsonb_build_object('date', to_char(gt.datestamp, 'DD-MM-YYYY'), 'daily_flow', gt.unimpaired_value)) as jsonf
		from unimpaired_gage_table as gt
		LIMIT 1
	)
	INSERT INTO cwat_app.gage_senior_diverter_csv (user_id, cda_session_id, csv_data, raw_senior_diverters, csv_with_intermediate_values, unimpaired_gage_data)
	VALUES (user_id_in, session_id_in, '{}'::json, '{}'::json, '{}'::json, (SELECT jsonf from formatted_json))
	ON CONFLICT (user_id, cda_session_id)
	DO UPDATE
	SET unimpaired_gage_data = (SELECT jsonf from formatted_json)
	WHERE gage_senior_diverter_csv.cda_session_id = session_id_in and gage_senior_diverter_csv.user_id = user_id_in;
	RETURN 'COMPLETE';

END;
$BODY$;