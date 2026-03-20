import json
from utils.cda_utils import (
    calculate_feb_median,
    get_senior_diverters_upstream_of_poi,
    impair_poi_time_series,
    scale_gage_ts_to_poi,
    calculate_cda_ratio,
    generate_senior_diverter_ts_poi,
    calculate_natural_flow_variability,
    calculate_instream_flows_reduction
)
from datetime import datetime
from database import Database

def generate_daily_flow_study_async(
    id,
    poi_id,
    dfs_uuid,
    user_id,
    session,
    poi_threshold,
    return_output = False
):
    """
    Generate a daily flow study, saving results in the database to be accessed via an api call asynchronously.

    Performs the following:
        1. Scale unimpaired gage time series to the POI using the poi proration ratio
        2. Get the POI senior diverters and re-impair the time-series
        3. Using generated time-series's, perform the spawning, rearing and passage daily flow study (calculate monthly data)
        4. Perform the natural flow variability daily flow study calculations (calculate 1.5-year instantaneous peaks)
        5. If the class II/III criteria is satisfied, perform monthly analysis of February median flow
        6. Bring all data together, store in user cda session and return to front end.

    Args:
        id - project id (int)
        poi_id - Point of Interest (POI) id (int)
        dfs_uuid - Daily Flow Study UUID (uuid)
        user_id - DB user_id (int)
        session - current user CDA session
        poi_threshold - some basic POI data (data filtered from session to given POI)
    """
    db = Database()
    daily_flow_study_results = {'poiId' : poi_id}
    poi = next((p for p in session['pointsOfInterest'] if p["id"] == poi_id), None)
    try:
        try:
            unimpaired_gage_data = db.get_unimpaired_gage_data(user_id, id)
            if(unimpaired_gage_data == None):
                if(not session['regionalCriteria']):
                    db.unimpair_gage_timeseries(user_id, id, json.dumps({1800: [0]*365}))
                    unimpaired_gage_data = db.get_unimpaired_gage_data(user_id, id)
                else:
                    raise Exception("An unimpaired gage time-series doesn't exist for the user.")
            if(not 'ratio' in poi_threshold):
                gage_ratio_raw = db.get_gage_size_and_mean_precip(wsr_session_id = id)
                #Get required data for user's pod
                wsr_session = db.get_wsr_session_by_id(user_id, id)
                pod_nhdid = wsr_session['session']['nhdId']
                if(pod_nhdid == None):
                    raise Exception("User does not have a selected point of diversion in session")
                poi_ratio_raw = db.get_poi_size_and_mean_precip(cda_session_id = id, poi_id = poi_id)
                poi_ratio_data = calculate_cda_ratio(poi_ratio_raw, gage_ratio_raw)
                poi_ratio_data['poiId'] = poi_id
                poi_threshold = poi_threshold | poi_ratio_data
        except Exception as e:
            raise Exception({"message": f'{str(e)}\nIssue with set up for daily flow study', 'status_code': 404})
        #Scale unimpaired gage time series to the POI
        try:
            unimpaired_poi_ts = scale_gage_ts_to_poi(unimpaired_gage_data, poi_threshold['ratio'])
            unimpaired_start_year = datetime.strptime(unimpaired_poi_ts.iloc[0]['date'], '%d-%m-%Y').year
            unimpaired_end_year = datetime.strptime(unimpaired_poi_ts.iloc[len(unimpaired_poi_ts.index)-1]['date'], '%d-%m-%Y').year
            daily_flow_study_results['yearsOfRecord'] = unimpaired_end_year-unimpaired_start_year
            db.save_unimpaired_poi_ts(user_id, id, poi_id, unimpaired_poi_ts.to_json(orient='records'))
        except Exception as e:
            raise Exception({"message": f'{str(e)}\nUnable to generate unimpaired poi time series', 'status_code': 400})
        #Get the senior diverters for the poi and impairg
        try:
            wsr_senior_diverters = db.get_senior_diverter_csv_by_user_id(user_id, id)['csv_data']
            if(wsr_senior_diverters == {}):
                raise Exception("No wsr senior diverters found - is the wsr section complete?")
            (upstream_senior_diverters, upstream_senior_diverters_with_pod) = get_senior_diverters_upstream_of_poi(wsr_senior_diverters, poi, session)
            currently_upstream = []
            onstream_storage_upstream_diverters = {}
            for diverter in upstream_senior_diverters_with_pod:
                if(diverter['analysis_label'] == 'Proposed POD'):
                    pod_upstream_diverters = db.get_proposed_pod_upstream_diverters(
                        session_id = id,
                        current_upstream_diverters = json.dumps(currently_upstream)
                    )
                else:
                    pod_upstream_diverters = db.get_onstream_pod_upstream_diverters(
                        water_right_id = int(diverter['wr_water_right_id']),
                        current_upstream_diverters = json.dumps(currently_upstream)
                    )
                pod_upstream_diverters = [int(x['order_upstream_to_downstream']) for x in pod_upstream_diverters]
                onstream_storage_upstream_diverters[diverter['order_upstream_to_downstream']] = pod_upstream_diverters
                currently_upstream.append({'order_upstream_to_downstream' : diverter['order_upstream_to_downstream'],
                                        'lat': diverter['latitude'],
                                        'lng': diverter['longitude']})
            gage_ratio_raw = db.get_gage_size_and_mean_precip(wsr_session_id = id)
            upstream_diverters_ts = generate_senior_diverter_ts_poi(
                upstream_senior_diverters,
                unimpaired_gage_data,
                onstream_storage_upstream_diverters,
                gage_ratio_raw,
                session
            )
            upstream_diverters_with_pod_ts = generate_senior_diverter_ts_poi(
                upstream_senior_diverters_with_pod,
                unimpaired_gage_data,
                onstream_storage_upstream_diverters,
                gage_ratio_raw,
                session
            )
            impair_result_diverters = impair_poi_time_series(
                unimpaired_poi_ts,
                upstream_diverters_ts
            )
            impair_result_diverters_with_pod = impair_poi_time_series(
                unimpaired_poi_ts,
                upstream_diverters_with_pod_ts
            )
            db.save_impaired_poi_timeseries(diverters = json.dumps(impair_result_diverters),
                                                diverters_with_pod = json.dumps(impair_result_diverters_with_pod),
                                                id = id,
                                                poi_id = poi_id)
        except Exception as e:
            raise Exception({"message": f'{str(e)}\nUnable to generate impaired time-series', 'status_code': 400})
        #Evaluation of reductions in instream flows needed for spawning, rearing, and passage
        try:
            # Get the data from the database
            poi_ts = {
                'unimpaired' : unimpaired_poi_ts,
                'impaired_with_diverters' : impair_result_diverters,
                'impaired_with_pod': impair_result_diverters_with_pod
            }
            season_start = session['seasonOfDiversionStart']
            season_end = session['seasonOfDiversionEnd']
            percentages = calculate_instream_flows_reduction(poi_ts, poi_threshold['minimumBypassFlow'], season_start, season_end)
            daily_flow_study_results['spawningPassage'] = percentages
        except Exception as e:
            raise Exception({"message": f'{str(e)}\nUnable to calculate data for spawning, rearing and passage daily flow study', 'status_code': 400})
        #Evaluations of reductions in instream flows needed for natural flow variability
        try:
            #Use the above poi_ts for this
            peaks_and_ratios = calculate_natural_flow_variability(poi_ts)
            daily_flow_study_results['naturalFlowVariability'] = peaks_and_ratios
        except Exception as e:
            raise Exception({"message": f'{str(e.__str__())}\nUnable to calculate data for natural flow variability daily flow study', 'status_code': 400})

        # February median analysis - only performed for class III POD/ class II POI
        try:
            if(session['podStreamClass'] == 3 and poi['class'] == 2):
                feb_median = calculate_feb_median(poi_ts['unimpaired'])
                febPercentages = calculate_instream_flows_reduction(poi_ts, feb_median, season_start, season_end)
                daily_flow_study_results['februaryMedian'] = febPercentages
        except Exception as e:
            raise Exception({"message": f'{str(e.__str__())}\nUnable to generate february median analysis', 'status_code': 400})

        #Reload the session here to allow for asynchronous processing
        session = db.get_cda_session_by_id(user_id, id)['session']
        if('dailyFlowData' in session and session['dailyFlowData'] != None):
            #If this has been done before (i.e session contains dfs data), append and save
            daily_flow_data = session['dailyFlowData']
            daily_flow_value = next((p for p in daily_flow_data if p["poiId"] == poi_id), None)
            if(daily_flow_value is not None):
                #If the poi already has an entry for this spot, then overwrite it
                index = daily_flow_data.index(daily_flow_value)
                daily_flow_data[index] = daily_flow_study_results
            else:
                #Otherwise append the data
                daily_flow_data.append(daily_flow_study_results)
            db.update_cda_session_by_id(user_id, id, {'dailyFlowData' : daily_flow_data})
        else:
            #Otherwise just make a new session entry for the daily flow study
            db.update_cda_session_by_id(user_id, id, {'dailyFlowData' : [daily_flow_study_results]})
        # Now we done - insert the data yay
        db.insert_daily_flow_job_data_into_db(
            session_id = id,
            poi_id = poi_id,
            dfs_uuid = str(dfs_uuid),
            results = json.dumps(daily_flow_study_results),
            status = 'Complete'
        )
        if(return_output):
            return daily_flow_study_results
    except Exception as e:
        # Error - update the db with this as well
        print(str(e))
        db.insert_daily_flow_job_data_into_db(
            session_id = id,
            poi_id = poi_id,
            dfs_uuid = str(dfs_uuid),
            results = json.dumps({}),
            status = 'Failed'
        )
