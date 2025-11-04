import os
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv())

port = os.getenv("PGPORT")
user = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")
database = os.getenv("PGDATABASE")
host = os.getenv("PGHOST")

class Database:
    def __init__(self) -> None:
        self.pool = ThreadedConnectionPool(minconn=1, maxconn=10, host = host, database = database, user = user, password = password, port = port)

    def execute_as_dict(self, sql, args=[], fetch_one = False):
        """
        Execute sql on a self closing pool connection. Returns results as a dict.

        :param str sql - sql to execute
        :param list | dict args - args to pass to sql. Can use list for ordered args or dict for named args
        :param bool fetch_one - Set to true to return one row or None. Defaults to false
        :
        """
        connection = self.pool.getconn()
        try:
            with connection.cursor(cursor_factory=RealDictCursor) as conn:
                conn.execute(sql, args)
                results = None
                if fetch_one:
                    results = conn.fetchone()
                else:
                    results = conn.fetchall()
                connection.commit()
                return results
        except Exception as error:
            print(f"error in execute func: {error} (query is {sql})")
            raise Exception(500)
        finally:
            self.pool.putconn(connection)

    def execute(self, sql, args=[]):
        """
        Execute sql on a self closing pool connection. Returns no results.

        :param str sql - sql to execute
        :param list | dict args - args to pass to sql. Can use list for ordered args or dict for named args
        """
        connection = self.pool.getconn()
        try:
            with connection.cursor(cursor_factory=RealDictCursor) as conn:
                conn.execute(sql, args)
                connection.commit()
        except Exception as error:
            print(f"error in execute func: {error} (query is {sql})")
            raise error
        finally:
            self.pool.putconn(connection)


    def get_user_wsr_sessions(self, user_id):
        """
        Get all current sessions for the given user_id in the WSR module.
        """
        result = self.execute_as_dict("select * from cwat_app.get_user_wsr_sessions(%s)", [user_id])
        return result

    def get_wsr_session_by_id(self, user_id, id):
        """
        Get the current session for the given user_id in the WSR module.
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_user_wsr_session_by_id(%s, %s::integer)", [user_id, id], fetch_one=True)
        return result

    def create_wsr_session(self, **args):
        from queries.create_wsr_session import create_wsr_session_query
        result = self.execute_as_dict(create_wsr_session_query, args, fetch_one=True)
        return result

    def update_wsr_session_by_id(self, user_id, id, session_data):
        result = self.execute_as_dict(f"select update_wsr_session_by_id as modified from cwat_app.update_wsr_session_by_id(%s, %s, %s)", [user_id, id, json.dumps(session_data)], fetch_one=True)
        return result

    def delete_wsr_session_by_id(self, user_id, id):
        self.execute(f"delete from cwat_app.project_sessions where user_id = %s and id = %s", [user_id, id])
        self.execute(f"delete from cwat_app.senior_diverter_csv where user_id = %s and wsr_session_id = %s", [user_id, id])
        return

    def get_cda_session_by_id(self, user_id, id):
        """
        Retrieve a user's cda_session object from the database based on their given user_id and session id (id)
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_user_cda_session_by_id(%s, %s::integer)", [user_id, id], fetch_one=True)
        return result

    def create_cda_session(self, user_id, wsr_session_id):
        result = self.execute_as_dict(f"select * from cwat_app.create_cda_session(%s, %s)", [user_id, wsr_session_id], fetch_one=True)
        return result

    def update_cda_session_by_id(self, user_id, id, session_data):
        result = self.execute_as_dict(f"select update_cda_session_by_id as modified from cwat_app.update_cda_session_by_id(%s, %s, %s)", [user_id, id, json.dumps(session_data)], fetch_one=True)
        return result

    def get_nhd_id_by_lat_lng(self, lat, lng, distance):
        result = self.execute_as_dict(f"select get_nhd_id_by_latlng as data from cwat_app.get_nhd_id_by_latlng(%s::numeric, %s::numeric, %s::integer)", [lng, lat, distance], fetch_one=True)
        return result.get('data')

    def get_senior_diverter_csv_by_user_id(self, user_id, id):
        """Fetch the raw senior diverters and edited senior diverters for a users WSR.

        Args:
            user_id (str): The user id
            id (integer): The WSR session id
        """
        result = self.execute_as_dict(f"select csv_data, raw_senior_diverters, csv_with_intermediate_values from cwat_app.get_senior_diverter_csv_by_user_id(%s::text, %s::integer)", [user_id, id], fetch_one=True)
        return result

    def get_wsr_edited_senior_diverter_csv_by_user_id(self, user_id, id):
        """Fetch the user edited senior diverter csv from the wsr
        Args:
            user_id (str): The user id
            id (integer): The WSR session id
        """
        result = self.execute_as_dict(f"select csv_data from cwat_app.get_senior_diverter_csv_by_user_id(%s::text, %s::integer)", [user_id, id], fetch_one=True)
        return result.get('csv_data')

    def get_gage_edited_senior_diverter_csv_by_user_id(self, user_id, id):
        """Fetch the user edited senior diverter csv for gage data used in CDA

        Args:
            user_id (str): The user id
            id (integer): The WSR session id
        """
        result = self.execute_as_dict(f"select csv_data from cwat_app.get_gage_edited_senior_diverter_csv_by_user_id(%s::text, %s::integer)", [user_id, id], fetch_one=True)
        return result

    def get_wsr_start_date_by_id(self, user_id, id):
        """Fetches the frozen date from the wsr session object

         Args:
            user_id (str): The user id
            id (integer): The WSR session id
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_wsr_start_date_by_id(%s::text, %s::bigint)", [user_id, id], fetch_one=True)
        return result['get_wsr_start_date_by_id']

    def get_wsr_selected_gage_by_id(self, user_id, id):
        """Fetches the selected gage from the wsr session object

         Args:
            user_id (str): The user id
            id (integer): The WSR session id
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_wsr_selected_gage_by_id(%s::text, %s::bigint)", [user_id, id], fetch_one=True)
        return result['get_wsr_selected_gage_by_id']

    def get_raw_senior_diverter_csv(self, nhd_id, lat, lng, timestamp):
        """Fetch the raw water rights data for the given wsr session.

        Args:
            user_id (str): The users id.
            id (integer): The WSR session id.
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_raw_senior_diverters_by_session_id(%s::bigint, %s::double precision, %s::double precision, %s::timestamp with time zone)", [nhd_id, lat, lng, timestamp], fetch_one=False)
        return result

    def get_watershed_by_nhd_id(self, nhd_id):
        result = self.execute_as_dict(f"select watershed from cwat_app.get_watershed_by_nhd_id(%s::bigint)", [nhd_id], fetch_one=True)
        return result

    def get_downstream_watershed_by_nhd_id(self, nhd_id):
        result = self.execute_as_dict(f"select watershed from cwat_app.get_downstream_watershed_by_nhd_id(%s::bigint)", [nhd_id], fetch_one=True)
        return result

    def get_all_pods(self):
        results = self.execute_as_dict(f"select get_all_pods from cwat_app.get_all_pods()")
        # Un-nesting the column name; psycopg adds it when fetching all
        return [result.get('get_all_pods') for result in results]

    def get_pod_by_pod_id(self, pod_id):
        result = self.execute_as_dict(f"select pod from cwat_app.get_pod_by_pod_id(%s::integer)", [pod_id], fetch_one=True)
        return result and result.get('pod')

    def get_wsr_summary(self, **args):
        from queries.get_wsr_summary_data import get_wsr_summary_data_query
        result = self.execute_as_dict(get_wsr_summary_data_query, args, fetch_one=False)
        return result

    def get_session_information(self, user_id, session_id):
        result = self.execute_as_dict(f"select * from cwat_app.get_session_information(%s::text, %s::integer)", [user_id, session_id], fetch_one=True)
        return result

    def get_streampath_by_nhd_id(self, nhd_id):
        result = self.execute_as_dict(f"select * from cwat_app.get_downstream_flow_path(%s::bigint)", [nhd_id], fetch_one=True)
        return result

    def get_wsr_flow_frequency_points_of_analysis(self, user_id, session_id):
        result = self.execute_as_dict(f"select * from cwat_app.get_flow_frequency_points_of_analysis(%s::bigint, %s::text)", [session_id, user_id], fetch_one=False)
        return result

    def get_candidate_gages_by_nhd_id(self, nhd, lat, lng):
        result = self.execute_as_dict(f"select get_proposed_and_top_3_candidate_gages from cwat_app.get_proposed_and_top_3_candidate_gages(%s::bigint, %s::double precision, %s::double precision)", [nhd, lat, lng], fetch_one=True)
        return result.get('get_proposed_and_top_3_candidate_gages')

    def get_all_gage_names(self):
        results = self.execute_as_dict(f"select get_all_gage_names from cwat_app.get_all_gage_names()", [], fetch_one=False)
        return [result.get('get_all_gage_names') for result in results]

    def get_gage_streamflow(self, session_id, user_id):
        results = self.execute_as_dict(f"select * from cwat_app.get_gage_streamflow(%s::integer, %s::text)", [session_id, user_id], fetch_one=False)
        return results

    def get_gage_by_id(self, id):
        results = self.execute_as_dict(f"select * from cwat_app.get_gage_by_id(%s::integer)", [id], fetch_one=True)
        return results.get('get_gage_by_id')

    def get_initial_gage_calculations(self, session_id, user_id):
        results = self.execute_as_dict(f"select * from cwat_app.get_wsr_gage_initial_calculations(%s::integer, %s::text)", [session_id, user_id], fetch_one=False)
        return results

    def verify_streamreach(self, nhd_id):
        result = self.execute_as_dict(f"select * from cwat_app.verify_nhd_id(%s::bigint)", [nhd_id], fetch_one=True)
        return result.get('verify_nhd_id')

    def save_raw_senior_diverters(self, user_id, id, senior_diverter_data):
        """Saves the app-generated raw senior diverter data to the senior_diverter_csv table
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            senior_diverter_data: The tool-generated senior diverter data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select save_raw_senior_diverters from cwat_app.save_raw_senior_diverters(%s::text, %s::integer, %s::jsonb)", [user_id, id, senior_diverter_data], fetch_one=True)
        return result.get('save_raw_senior_diverters')

    def check_gage_diverters_exists(self, user_id, id):

        result = self.execute_as_dict(f'SELECT EXISTS(SELECT 1 FROM cwat_app.gage_senior_diverter_csv WHERE (cda_session_id=%s::integer AND user_id=%s::text))', [id, user_id], fetch_one=True)
        return result.get('exists')

    def save_raw_senior_diverters_gage(self, user_id, id, senior_diverter_data):
        """Saves the app-generated raw senior diverter data to the gage_senior_diverter_csv table
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            gage_senior_diverter_data: The tool-generated senior diverter data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select save_raw_senior_diverters_gage from cwat_app.save_raw_senior_diverters_gage(%s::text, %s::integer, %s::jsonb)", [user_id, id, senior_diverter_data], fetch_one=True)
        return result.get('save_raw_senior_diverters')

    def validate_senior_diverters_within_downstream_watershed(self, user_id, id, edited_csv_data):
        """Validated the user_uploaded senior diverter data so that each diverter is within its downstream watershed
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            edited_csv_data: The user-uploaded senior diverter data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select validate_senior_diverters_within_downstream_watershed from cwat_app.validate_senior_diverters_within_downstream_watershed(%s::text, %s::integer, %s::jsonb)", [user_id, id, edited_csv_data], fetch_one=True)
        return result.get('validate_senior_diverters_within_downstream_watershed')

    def save_edited_senior_diverters(self, user_id, id, edited_csv_data, intermediate_csv_data):
        """Saves the user-uploaded senior diverter data to the senior_diverter_csv table
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            edited_csv_data: The user-uploaded senior diverter data
            intermediate_csv_data: The tool-generated senior diverter calculation data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select save_edited_senior_diverters from cwat_app.save_edited_senior_diverters(%s::text, %s::integer, %s::jsonb, %s::jsonb)", [user_id, id, edited_csv_data, intermediate_csv_data], fetch_one=True)
        return result.get('save_edited_senior_diverters')

    def validate_gage_senior_diverters_within_upstream_watershed(self, user_id, id, edited_csv_data):
        """Validate the user_uploaded senior diverter gage data so that each diverter is within its upstream watershed
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            edited_csv_data: The user-uploaded senior diverter data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select validate_gage_senior_diverters_within_upstream_watershed from cwat_app.validate_gage_senior_diverters_within_upstream_watershed(%s::text, %s::integer, %s::jsonb)", [user_id, id, edited_csv_data], fetch_one=True)
        return result.get('validate_gage_senior_diverters_within_upstream_watershed')

    def save_gage_edited_senior_diverters(self, user_id, id, edited_csv_data, intermediate_csv_data):
        """Saves the user-edited gage senior diverter data to the gage_senior_diverter_csv table
        Args:
            user_id (string): The users id
            id (number): The wsr session id
            edited_csv_data: The user-uploaded senior diverter data
            intermediate_csv_data: The tool-generated senior diverter calculation data

        Returns:
            list or dict: If invalid, a list of the failing rows. Otherwise a dict with the newly inserted row's id.
        """
        result = self.execute_as_dict(f"select save_gage_edited_senior_diverters from cwat_app.save_gage_edited_senior_diverters(%s::text, %s::integer, %s::jsonb, %s::jsonb)", [user_id, id, edited_csv_data, intermediate_csv_data], fetch_one=True)
        return result.get('save_gage_edited_senior_diverters')

    def get_poi_ratio_data(self, nhdid):
        """
            Calls function get_poi_ratio_data for nhdid values of a poi or gage.
            Args:
                nhdid
            Returns:
                Dict: precipitation and drainage data for the watershed that the poi is located in.
        """
        result = self.execute_as_dict(f"select get_poi_ratio_data as data from cwat_app.get_poi_ratio_data(%s::bigint)", [nhdid], fetch_one=True)
        return result.get("data")

    def unimpair_gage_timeseries(self, user_id, id, ts_data):
        """
            Calls function unimpair_gage_timeseries for the id pair supplied and unimpairs the user selected gage with the data provided in ts_data
            Args:
                user_id (string): Users autorized id as supplied by flask
                id (int): the desired session's id
                ts_data (json): "compressed" timeseries data for the impairment caused by the diverters of the user selected gage on a daily basis of the json form {YYYY : [x, y, z, ...], YYYY: [x, y, z, ...], ...}
                    where the keys YYYY are each water year where changes occur in the diversions and the values are arrays containing doubles that represent how much impairment occurs for each day after the start of the water year such that the first entry is october 1st of YYYY and the last entry represents the impairment on September 31st of YYYY + 1
            Returns:
                String: "COMPLETE" to signal the computation is done
        """
        result = self.execute_as_dict(f"select * from cwat_app.unimpair_gage_timeseries(%s::jsonb, %s::text, %s::bigint)", [ts_data,user_id,id], fetch_one=True)
        return result.get("unimpair_gage_timeseries")

    def reimpair_gage_timeseries(self, user_id, id, poi_id, contains_pod, ts_data):
        """
            Calls the function reimpair_gage_timeseries to reimpair the unimpaired poi data with the diverters it has as it is currently an unimpaired estimation in the db
            Args:
                user_id (string): Users autorized id as supplied by flask
                id (int): the desired session's id
                ts_data (json): "compressed" timeseries data for the impairment caused by the diverters of the desired poi on a daily basis of the json form {YYYY : [x, y, z, ...], YYYY: [x, y, z, ...], ...}
                    where the keys YYYY are each water year where changes occur in the diversions and the values are arrays containing doubles that represent how much impairment occurs for each day after the start of the water year such that the first entry is october 1st of YYYY and the last entry represents the impairment on September 31st of YYYY + 1
                poi_id (int): the poi_id for the poi to be updated
                contains_pod (boolean): whether or not the impairment data contains the proposed point of diversion (changes what column data is inserted into)
            Returns:
                String: "COMPLETE" to signal the computation is done
        """
        result = self.execute_as_dict(f"select * from cwat_app.reimpair_gage_timeseries(%s::jsonb, %s::text, %s::bigint, %s::bigint, %s::int)", [ts_data, user_id, id, poi_id, contains_pod], fetch_one=True)
        return result.get("reimpair_gage_timeseries")


    def get_unimpaired_gage_data(self, user_id, id):
        """
            Calls function get_unimpaired_gage_data to fetch unimpaired data unimpaired by unimpair_gage_timeseries
            Args:
                user_id (string): The users id
                id (number): The wsr/cda session id
            Returns:
                json: json of the formatted as {[{"date": dd-mm-yyyy, daily_flow: x}, ...]}
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_unimpaired_gage_data( %s::text, %s::bigint)", [user_id,id], fetch_one=True)
        return result.get("get_unimpaired_gage_data")

    def get_gage_calculated_senior_diverters_by_user_id(self, user_id, id):
        """Fetch the edited senior diverters with calculated values for the CDA.

        Args:
            user_id (str): The user id
            id (integer): The CDA session id
        """
        result = self.execute_as_dict(f"select gage_senior_diverter_csv.csv_with_intermediate_values as data from cwat_app.gage_senior_diverter_csv where user_id = %s AND cda_session_id=%s", [user_id, id], fetch_one=True)
        return result.get("data")

    def save_unimpaired_poi_ts(self, user_id, id, poi_id, data):
        """
            Save the unimpaired poi time-series to the daily_flow_data table

            Args:
                user_id (str): The user id
                id (integer): The CDA session id
                poi_id (integer): The POI id
                data (dict): data to be saved
            Returns:
                json of row id {'id' : <id>}
        """
        result = self.execute_as_dict(f"select save_unimpaired_poi_ts from cwat_app.save_unimpaired_poi_ts(%s::text, %s::integer, %s::integer, %s::jsonb)", [user_id, id, poi_id, data], fetch_one=True)
        return result

    def get_poi_ts(self, user_id, id, poi_id):
        """
            Gets the calculated poi time series for daily flow study calculations.

            Args:
                user_id (str): User id
                id (integer): CDA session id
                poi_id: the POI id
            Returns:

        """
        query = f"select unimpaired_poi_data, diverters_impaired_poi_data, pod_impaired_poi_data from cwat_app.daily_flow_data where user_id = '{user_id}' and cda_session_id = {id} and poi_id = {poi_id};"
        result = self.execute_as_dict(query, fetch_one=True)
        return {'unimpaired' : result.get('unimpaired_poi_data'), 'impaired_with_diverters': result.get('diverters_impaired_poi_data'), 'impaired_with_pod' : result.get('pod_impaired_poi_data')}

    def get_gage_senior_diverter_csv_by_user_id(self, user_id, id):
        """Fetch the gage raw senior diverters and edited gage senior diverters for a users CDA session.

        Args:
            user_id (str): The user id
            id (integer): The CDA session id
        """
        result = self.execute_as_dict(f"select csv_data, raw_senior_diverters, csv_with_intermediate_values from cwat_app.gage_senior_diverter_csv where user_id = %s AND cda_session_id=%s", [user_id, id], fetch_one=True)
        if(result is None):
            return None
        return {'uploaded' : result.get('csv_data'), 'raw_diverters': result.get('raw_senior_diverters'), 'intermediate' : result.get('csv_with_intermediate_values')}

    def get_raw_gage_data(self, user_id, id):
        """
            Fetch the raw gage data from the cwat_data.water_daily table based on the given user and session ID's wsr selected gage.

            Args:
                user_id (str): The user id
                id (integer): CDA session id
        """
        result = self.execute_as_dict(f"select get_raw_gage_data from cwat_app.get_raw_gage_data(%s::text, %s::bigint)", [user_id, id], fetch_one=True)
        return result.get('get_raw_gage_data')

    def get_ws_bbox_by_nhdplusid(self, nhdplusid):
        """
            Fetch the bounding box of the watershed in cwat_data.ws_geoms_all table given the nhdplusid of the desired watershed.

            Args:
                nhdplusid (int): Bigint for watershed
        """
        result = self.execute_as_dict(f"select bbox_bounds from cwat_app.get_ws_bbox_by_nhdplusid(%s::bigint)", [nhdplusid], fetch_one=True)
        if(result is None):
            return None
        return(result.get('bbox_bounds'))

    def get_stream_by_name(self, search_name):
        """
            Get all the stream segments that fuzzy match the supplied search string.

            Args:
                search_name (string): Text to fuzzy match against the database.
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_stream_by_name(%s::text)", [search_name], fetch_one=True)
        return result

    def get_poi_data_by_nhd_id(self, nhd_id):
        """
            Get some front-end display data for the poi.

            Args:
                nhd_id -> poi stream reach nhd id
        """
        result = self.execute_as_dict(f"select poi_data from cwat_app.get_poi_data_by_nhd_id(%s::bigint)", [nhd_id], fetch_one=True)
        return result.get('poi_data')

    def get_unsorted_senior_diverter_csv(self, **args):
        """Fetch the unsorted (raw, bare-bones) water rights data for the given wsr session.

        Args:
            nhd_id - nhd id
            lat - latitude
            long - longitude
            timestamp - current time, used for limiting date
        """
        from queries.get_unsorted_senior_diverter_csv import get_unsorted_senior_diverter_csv_query
        result = self.execute_as_dict(get_unsorted_senior_diverter_csv_query, args, fetch_one=False)
        return result

    def get_depth_1_nhds_upstream_of_nhd_id(self, most_downstream_wr_nhd_id, nhd_id):
        """
        Get the depth 1 (no recursion) nhd ids upstream of the given nhd id.

        Args:
            most_downstream_wr_nhd_id - most downstream nhd id
            nhd_id - nhd id currently being analysed
        """
        result = self.execute_as_dict(f"select * from cwat_app.get_depth_1_nhds_upstream_of_nhd_id(%s::bigint, %s::bigint)", [most_downstream_wr_nhd_id, nhd_id], fetch_one=False)
        return result

    def update_project_status_by_id(self, id, status):
        """
        Update the project at the given id with the given status.

        Args:
            id - project (cda or wsr session) id
            status - status (1 - wsr, 2 - cda, 3 - complete)
        """
        self.execute(f"update cwat_app.project_sessions set project_status = (%s::integer) where id = (%s::integer)", [status, id])
        return True

    def get_project_status_by_id(self, id):
        """
        Get the project status at the given id.

        Args:
            id - project (cda or wsr session) id
        """
        result = self.execute_as_dict(f"select project_status from cwat_app.project_sessions where id = (%s::integer)", [id], fetch_one=True)
        return result.get('project_status')

    def lat_long_in_policy(self, lat, lng):
        """
        Checks whether a given lat and long are in the policy area given polygon from cwat_data.policy_boundary

        Args:
            lat - latitude
            lng - longitude
        """
        result = self.execute_as_dict(f"SELECT ST_Contains(geom4326, ST_Point((%s::numeric),(%s::numeric), 4326)) FROM cwat_data.policy_boundary", [lng, lat], fetch_one=True)
        return result.get('st_contains')

    def remove_wsr_senior_diverters_record(self, user_id, session_id):
        """
        Remove from wsr senior diverters table (user has cleared their file)

        Args:
            user_id - user ID
            session_id - session ID
        """
        self.execute(f"DELETE FROM cwat_app.senior_diverter_csv WHERE user_id = (%s::text) AND wsr_session_id = (%s::integer)", [user_id, session_id])


    def generate_watershed_for_pod_and_store(self, pod_geom, nhd_plus_id, session_id):
        """
        Uses the sql stored in generate_watershed_for_pod to create a watershed for the given POD

        Args:
            pod_geom - geojson of user POD point
            nhd_plus_id - user selected NHD ID
            session_id - project session inserting for
        """
        from queries.generate_watershed_for_pod import generate_watershed_for_pod_and_store_query
        self.execute(generate_watershed_for_pod_and_store_query, [pod_geom, nhd_plus_id, session_id])

    def get_gage_size_and_mean_precip(self, **args):
        """
        Uses the SQL stored in get_gage_size_and_mean_precip.py to get some gage data necessary for wsr session packages.
        """
        from queries.get_gage_size_and_mean_precip import get_gage_size_and_mean_precip_query
        data = self.execute_as_dict(get_gage_size_and_mean_precip_query, args, fetch_one=True)
        return data

    def get_diverter_size_and_mean_precip(self, **args):
        """
        Gets the diverter size and mean precipitation values to get diverter watershed data
        """
        from queries.get_diverter_size_and_mean_precip import get_diverter_size_and_mean_precip_query
        data = self.execute_as_dict(get_diverter_size_and_mean_precip_query, args, fetch_one=False)
        return data

    def get_pod_size_and_mean_precip(self, **args):
        """
        Gets the pod size and mean precipitation values to get pod watershed data
        """
        from queries.get_pod_size_and_mean_precip import get_pod_size_and_mean_precip_query
        data = self.execute_as_dict(get_pod_size_and_mean_precip_query, args, fetch_one=True)
        return data

    def generate_watershed_for_poi_and_store(self, **args):
        """
        Gets a watershed for a poi (point of interest) for the CDA
        """
        from queries.generate_watershed_for_poi_and_store import generate_watershed_for_poi_and_store_query
        self.execute(generate_watershed_for_poi_and_store_query, args)

    def get_poi_size_and_mean_precip(self, **args):
        """
        Gets the size of a poi and its mean precip values
        """
        from queries.get_poi_size_and_mean_precip import get_poi_size_and_mean_precip_query
        data = self.execute_as_dict(get_poi_size_and_mean_precip_query, args, fetch_one=True)
        return data

    def save_impaired_poi_timeseries(self, **args):
        """
        Saves given impaired poi timeseries's into the database.
        """
        from queries.save_impaired_poi_timeseries import save_impaired_poi_timeseries_query
        self.execute(save_impaired_poi_timeseries_query, args=args)

    def get_onstream_pod_upstream_diverters(self, **args):
        """
        For a point of onstream storage, get the list of diverters which are upstream of it.
        """
        from queries.get_onstream_pod_upstream_diverters import get_onstream_pod_upstream_diverters_query
        result = self.execute_as_dict(get_onstream_pod_upstream_diverters_query, args=args)
        return result