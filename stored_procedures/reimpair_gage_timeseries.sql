-- stored procedure defintion for cwat_app.reimpair_gage_timeseries

CREATE OR REPLACE FUNCTION cwat_app.reimpair_gage_timeseries(
	input_data jsonb,
	user_id_in text,
	session_id_in bigint,
	poi_id_in bigint,
	includes_pod integer)
    RETURNS text
    LANGUAGE 'plpgsql'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$
BEGIN
	SET datestyle to 'dmy';
	WITH gage_json as(
		SELECT unimpaired_poi_data as json_data
		FROM cwat_app.daily_flow_data
		WHERE user_id = user_id_in and cda_session_id = session_id_in and poi_id = poi_id_in
	),gage_data AS (
		SELECT
			json_table.date::date as datestamp,
			json_table.daily_flow::double precision as val
		FROM
			gage_json,
			jsonb_to_recordset(gage_json.json_data) AS json_table(date text, daily_flow double precision)
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
	), impaired_gage_table as (
		SELECT
			gd.datestamp,
			GREATEST(gd.val - jt.datac[gd.julian_date], 0.0) as impaired_value
		FROM
			gage_data_julian as gd
		JOIN
			json_table as jt
		ON
			jt.datec = gd.water_year
		ORDER BY
		datestamp asc
	), formatted_json as (
		select jsonb_agg(jsonb_build_object('date', to_char(gt.datestamp, 'DD-MM-YYYY'), 'daily_flow', gt.impaired_value)) as jsonf
		from impaired_gage_table as gt
		LIMIT 1
	)
	UPDATE cwat_app.daily_flow_data
    SET
        diverters_impaired_poi_data = CASE
                                        WHEN includes_pod = 0 THEN COALESCE(formatted_json.jsonf, diverters_impaired_poi_data)
                                        ELSE diverters_impaired_poi_data
                                    END,
        pod_impaired_poi_data = CASE
                          WHEN includes_pod = 1 THEN COALESCE(formatted_json.jsonf, pod_impaired_poi_data)
                          ELSE pod_impaired_poi_data
                      END
    FROM formatted_json
    WHERE daily_flow_data.cda_session_id = session_id_in
          AND daily_flow_data.user_id = user_id_in
          AND daily_flow_data.poi_id = poi_id_in;
	SET datestyle to 'iso';
    RETURN 'COMPLETE';
END;
$BODY$;
