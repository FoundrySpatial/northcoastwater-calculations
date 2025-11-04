get_wsr_summary_data_query = """
WITH saved_senior_diverters_with_intermediate_values AS (
    SELECT
        *
    FROM
        jsonb_to_recordset((
            SELECT
                csv_with_intermediate_values
            FROM cwat_app.senior_diverter_csv
            WHERE
                wsr_session_id = %(wsr_session_id)s AND user_id = %(user_id)s)) AS results (
        order_upstream_to_downstream int,
        application_number text,
        analysis_label text,
        latitude double precision,
        longitude double precision,
        wr_seasonal_demand double precision,
        drainage_area_sqmi double precision,
        annual_precip_in double precision
        )
)
, saved_seasonal_demand_w_nhdplus AS (
	SELECT
		saved_csv.order_upstream_to_downstream,
        saved_csv.application_number,
        saved_csv.analysis_label,
        saved_csv.latitude,
        saved_csv.longitude,
        saved_csv.wr_seasonal_demand,
        drainage_area_sqmi,
        annual_precip_in
    FROM
        saved_senior_diverters_with_intermediate_values saved_csv
)
, user_info AS (
    SELECT
        (wsr_session -> 'volumeOfDiversion' ->> 'value')::numeric AS volume_of_diversion_value,
        (wsr_session ->> 'seasonOfDiversionStart')::date AS direct_div_season_start,
        (wsr_session ->> 'seasonOfDiversionEnd')::date AS direct_div_season_end,
        (wsr_session ->> 'selectedGage')::integer AS station_id
    FROM
        cwat_app.project_sessions
    WHERE
        id = %(wsr_session_id)s
)
, gage_initial_calculations AS (
	select
		gage.site_no,
		to_char(direct_div_season_start, 'Mon DD') || ' - ' || to_char(direct_div_season_end, 'Mon DD') as diversion_season
	from
        user_info
    JOIN
        cwat_data.stations gage
    ON
        gage.station_id = user_info.station_id
)
, initial_wsr_calcs AS (
    SELECT
        saved_csv.analysis_label,
        CASE WHEN saved_csv.analysis_label = 'Proposed POD' THEN
            saved_csv.analysis_label
        ELSE
            saved_csv.application_number
        END AS application_number,
        user_info.station_id,
        gage.diversion_season,
        CASE WHEN saved_csv.analysis_label = 'Proposed POD' THEN
        	0
   		 ELSE
        	saved_csv.wr_seasonal_demand
    	END AS seasonal_demand_before_new_water_right_af,
        COALESCE(
		SUM(
			CASE WHEN saved_csv.analysis_label = 'Proposed POD' THEN
				0
			ELSE
				saved_csv.wr_seasonal_demand
			END
		) OVER (ORDER BY saved_csv.order_upstream_to_downstream ASC ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
			0
		) AS seasonal_upstream_demand_af,
        user_info.volume_of_diversion_value AS additional_impairment_caused_by_new_water_right_af,
        drainage_area_sqmi,
        annual_precip_in
	FROM
    	saved_seasonal_demand_w_nhdplus saved_csv
    CROSS JOIN
		gage_initial_calculations gage
    CROSS JOIN
		user_info
    ORDER BY
        order_upstream_to_downstream
), staging_table as (
SELECT
    initial_wsr_calcs.analysis_label,
    initial_wsr_calcs.application_number,
    initial_wsr_calcs.station_id,
    initial_wsr_calcs.diversion_season,
    initial_wsr_calcs.seasonal_demand_before_new_water_right_af,
    initial_wsr_calcs.seasonal_upstream_demand_af,
    initial_wsr_calcs.additional_impairment_caused_by_new_water_right_af,
    initial_wsr_calcs.drainage_area_sqmi,
    initial_wsr_calcs.annual_precip_in
FROM
    initial_wsr_calcs
WHERE
    initial_wsr_calcs.analysis_label IN ('Proposed POD' , 'Mainstem POA', 'Downstream Flow Path')
)
SELECT
    staging_table.analysis_label::text,
    staging_table.application_number::text,
    staging_table.station_id::int,
    staging_table.diversion_season::text,
    staging_table.seasonal_demand_before_new_water_right_af::double precision,
    staging_table.seasonal_upstream_demand_af::double precision,
    staging_table.additional_impairment_caused_by_new_water_right_af::double precision,
    staging_table.drainage_area_sqmi::double precision as area_sqmi,
    staging_table.annual_precip_in::double precision
FROM
    staging_table;
"""