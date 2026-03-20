insert_daily_flow_job_data_into_db_query = f"""
UPDATE cwat_app.daily_flow_study_jobs
SET
    output_data = %(results)s,
    job_status = %(status)s
WHERE
    project_id = %(session_id)s
AND
    poi_id = %(poi_id)s
AND
    job_id = %(dfs_uuid)s;
"""