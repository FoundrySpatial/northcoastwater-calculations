create_dfs_job_entry_query = f"""
INSERT INTO cwat_app.daily_flow_study_jobs(
    project_id,
    poi_id,
    job_id,
    job_status,
    output_data
)
VALUES(
    %(session_id)s,
    %(poi_id)s,
    %(dfs_uuid)s,
    'In Progress',
    '{{}}'::jsonb
)
"""