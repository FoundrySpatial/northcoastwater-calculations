create_wsr_session_query = f"""
INSERT INTO cwat_app.project_sessions (user_id, wsr_session)
    VALUES (%(user_id)s, jsonb_build_object('status', 'In Progress (WSR)', 'hasEditedWaterRights', FALSE, 'meta', jsonb_build_object('created', statement_timestamp(), 'modified', statement_timestamp())))
RETURNING
    id AS session_id;
"""