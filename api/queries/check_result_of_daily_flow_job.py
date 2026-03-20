check_result_of_daily_flow_job_query = f"""
SELECT
    job_status,
    output_data
FROM
    cwat_app.daily_flow_study_jobs
WHERE
    project_id = %(session_id)s
AND
    job_id = %(job_id)s
AND
    poi_id = %(poi_id)s;
"""