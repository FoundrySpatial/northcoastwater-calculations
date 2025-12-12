import json
from multiprocessing import Process
from utils.cda_output_package import generate_cda_output_package
import decimal
from flask import Blueprint, request, jsonify, g, current_app as app, Response, send_file
from utils.api_utils import check_query_params
from utils.wsr_csv_utils import (
    calculate_wsr_output_values,
    sort_and_format_unsorted_csv_data,
    validate_senior_diverter_csv,
    wsr_schema,
    get_intermediate_data_json_formatted,
    calc_wsr_intermediate_values,
    get_wsr_frequency_analysis,
    get_adjusted_csv_data,
    get_wsr_water_rights_json_formatted,
    format_wsr_summary_csv_names,
    get_wsr_summary_csv_format,
    add_wsr_flow_frequencies_to_zip,
    get_wsr_water_rights_csv_formatted,
    required_uploaded_csv_columns,
    get_wsr_intermediate_csv_formatted,
    format_session_data,
    generate_wsr_flow_frequency_points_of_analysis,
    generate_gage_timeseries_seasonal_means
)
from utils.cda_utils import (
    AFD_TO_CFS,
    calculate_cda_intermediate_values,
    calculate_feb_median,
    calculate_mbf,
    calculate_mcd,
    calculate_yearly_mean,
    generate_recurrence_intervals,
    get_senior_diverters_upstream_of_poi,
    impair_poi_time_series,
    overwrite_with_wsr_diverters,
    peaks_and_threshold,
    plot_and_find_instantaneous_peak_flow,
    scale_gage_ts_to_poi,
    validate_gage_senior_diverter_csv,
    calculate_cda_ratio,
    cda_schema,
    generate_senior_diverter_ts,
    generate_senior_diverter_ts_poi,
    calculate_natural_flow_variability,
    calculate_instream_flows_reduction
)
from authentication.guards import authorization_guard
import pandas as pd
import csv
from fixtures.projects import projects_fixture
from authentication.guards import authorization_guard
from werkzeug.exceptions import BadRequest
from jsonschema import validate, ValidationError
import zipfile
from datetime import datetime
from io import BytesIO
from utils.helpers import unit_converter
import dateutil.parser as date_parser
import pytz

projects = Blueprint('projects', __name__)

@projects.route('/', methods=['GET'])
def get_all_projects():
    """
    Get all projects
    """
    return jsonify(projects_fixture), 200

@projects.route('/<int:id>/status', methods=['PUT'])
@authorization_guard
@check_query_params(['status'])
def update_project_status_by_id(params, id):
    """
    Update this project status information
    Statuses:
        {
            0: 'Error', #deleted?
            1: 'In progress (WSR)',
            2: 'In progress (CDA)',
            3: 'Complete',
        }
    """
    # if this project exists, update the various columns with this data
    status,  = params
    if status is None:
        raise Exception({'message': "status not in request ", 'status_code': 400})
    app.db.update_project_status_by_id(id, status)
    return {'status' : int(status)}, 200

@projects.route('/<int:id>/status', methods=['GET'])
@authorization_guard
def get_project_status_by_id(id):
    """
    Fetch the project status from the db
    """
    status = app.db.get_project_status_by_id(id)
    return {'status': status}, 200

@projects.route('/statuses', methods=['GET'])
@authorization_guard
def get_all_statuses():
    return {
        0: 'Error',
        1: 'In progress (WSR)',
        2: 'In progress (CDA)',
        3: 'Complete',
    }


@projects.route('/wsr/sessions/<int:id>/water-rights', methods=['POST'])
@authorization_guard
def create_project_session_water_rights_csv(id):
    """
    Create a csv file from uploaded. Expects an uploaded file named 'user_demands'.
    """
    try:
        user_wsr_data = app.db.get_wsr_session_by_id(g.user_id, id)
        requires_cda = True
        if "requiresCda" in user_wsr_data['session'].keys():
            requires_cda = user_wsr_data['session']['requiresCda']
        df = None
        user_included_upload = request.mimetype == 'multipart/form-data'

        if (user_included_upload):
            #Handle if the user hasn't originally downloaded their file
            existing_files = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)
            if(existing_files is None):
                app.db.save_raw_senior_diverters(g.user_id, id, '{}')
            date = app.db.get_wsr_start_date_by_id(g.user_id, id)
            if(date is None):
                date = datetime.now(pytz.timezone('America/Vancouver'))
                app.db.update_wsr_session_by_id(g.user_id, id, {'freezeDate': str(date)})
            file = request.files['user_demands']
            df = pd.read_csv(file)
            csv_errors = validate_senior_diverter_csv(df, requires_cda)
            if (len(csv_errors) > 0):
                raise Exception({'message': csv_errors, 'status_code': 400})

        # If no upload included, set the data to just the user's proposed POD information (no senior diverters)
        else:
            df = pd.DataFrame(columns=required_uploaded_csv_columns)
            df.loc[0, 'analysis_label'] = 'Proposed POD'
            df.loc[0, "order_upstream_to_downstream"] = 1
            df.loc[0, "latitude"] = user_wsr_data['session']['pointOfDiversion']['geometry']['coordinates'][1]
            df.loc[0, "longitude"] = user_wsr_data['session']['pointOfDiversion']['geometry']['coordinates'][0]

            volume_of_diversion = user_wsr_data['session']['volumeOfDiversion']
            rate_of_diversion = user_wsr_data['session']['rateOfDiversion']

            if (volume_of_diversion['unit'] == 'acreFeet'):
                df.loc[0, "face_amount_af"] = volume_of_diversion['value']
            elif (volume_of_diversion['unit'] == 'gallons'):
                df.loc[0, "face_amount_af"] = unit_converter.gallons_to_af(volume_of_diversion['value'])

            if (rate_of_diversion['unit'] == 'acreFeet/s'):
                df.loc[0, "max_rate_of_diversion_cfs"] = unit_converter.af_to_cf(rate_of_diversion['value'])
            elif (rate_of_diversion['unit'] == 'cf/s'):
                df.loc[0, "max_rate_of_diversion_cfs"] = rate_of_diversion['value']

            start_date = date_parser.parse(user_wsr_data['session']['seasonOfDiversionStart'])
            end_date = date_parser.parse(user_wsr_data['session']['seasonOfDiversionEnd'])
            df.loc[0, "direct_div_season_start_month"] = start_date.month
            df.loc[0, "direct_div_season_start_day"] = start_date.day
            df.loc[0, "direct_div_season_end_month"] = end_date.month
            df.loc[0, "direct_div_season_end_day"] = end_date.day

            date = datetime.now(pytz.timezone('America/Vancouver'))
            existing_files = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)
            if(existing_files is None):
                app.db.save_raw_senior_diverters(g.user_id, id, '{}')
            app.db.update_wsr_session_by_id(g.user_id, id, {'freezeDate': str(date)})

        points_outside_watershed = app.db.validate_senior_diverters_within_downstream_watershed(g.user_id, id, df.to_json(orient='records'))
        if (points_outside_watershed != None):
            raise Exception({ "message": [f"application {row['application_number']} is outside the downstream watershed" for row in points_outside_watershed] , "status_code" : 400})

        form_data = pd.Series(dict(user_wsr_data['session']))
        intermediate_wsr_values = calc_wsr_intermediate_values(df.copy(), form_data)
        result = app.db.save_edited_senior_diverters(g.user_id, id, df.to_json(orient='records'), intermediate_wsr_values.to_json(orient='records'))

        # Update flag in session
        app.db.update_wsr_session_by_id(g.user_id, id, {'hasEditedWaterRights': True})
        return result, 200
    except BadRequest as error:
        raise Exception({'message': 'Could not read file.', 'status_code': 400})

@projects.route('/wsr/sessions/<int:id>/water-rights', methods=['GET'])
@authorization_guard
def get_project_session_water_rights(id):
    """
    Retrieve the water-rights for a proposed POD. Pass an 'edited' query param to fetch the user uploaded version.

    :param str edited - include to get the user edited version.
    """
    water_rights_csv_data = None

    water_rights_csv_data = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)
    if (water_rights_csv_data == None):
        raise Exception({"message": "You have not uploaded any water-rights yet.", "status_code": 400})

    if request.mimetype == 'text/csv':
        water_rights_csv = get_wsr_water_rights_csv_formatted(water_rights_csv_data.get('csv_data'), True)
        return Response(water_rights_csv,  mimetype='application/zip'), 200
    else:
        # modify the senior diverters so it's a geojson Feature Collection
        sd_geojson = get_wsr_water_rights_json_formatted(water_rights_csv_data.get('csv_data'))
        return jsonify(sd_geojson), 200

@projects.route('/wsr/sessions/<int:id>/water-rights', methods=['DELETE'])
@authorization_guard
def remove_existing_project_water_rights(id):
    """
    Removes the user-uploaded water rights and any reference to them in the database.
    Allows the user to re-select their POD.

    Args:
        id - wsr session id
    """
    #We need to delete the senior diverters
    app.db.remove_wsr_senior_diverters_record(g.user_id, id)
    result = app.db.update_wsr_session_by_id(g.user_id, id, {'hasEditedWaterRights': False})
    return result, 200

@projects.route('cda/sessions/<int:id>/gage-water-rights', methods=['GET'])
@authorization_guard
@check_query_params(['edited'])
def get_gage_water_rights(params, id):
    """
    Retrieve the raw water-rights for a selected gage. Pass an 'edited' query param to fetch the user uploaded version.

    :param str edited - include to get the user edited version.
    """
    ( edited ) = params
    edited = edited[0]
    if edited.lower() == "false":
        gage = app.db.get_wsr_selected_gage_by_id(g.user_id, id)
        nhd = gage["nhdplusid"]
        lat = gage["site_location"]["geometry"]["coordinates"][1]
        lng = gage["site_location"]["geometry"]["coordinates"][0]
        date = app.db.get_wsr_start_date_by_id(g.user_id, id)
        if(date is None):
            date = datetime.now(pytz.timezone('America/Vancouver'))
            app.db.update_wsr_session_by_id(g.user_id, id, {'freezeDate': str(date)})
        wsr_session = app.db.get_wsr_session_by_id(g.user_id, id)['session']
        water_shed = app.db.get_watershed_by_nhd_id(nhd)
        unsorted_water_rights_csv_data = app.db.get_unsorted_senior_diverter_csv(nhdplusid = nhd, lat = lat, lng = lng, date = date)
        # Handling for differently-sized watersheds for PODs
        water_rights_dicts = [dict(row) for row in unsorted_water_rights_csv_data]
        application_numbers = [row['application_number'] for row in water_rights_dicts]
        pod_diverter_rain_and_area = app.db.get_diverter_size_and_mean_precip(application_numbers = application_numbers)
        pod_diverter_rain_and_area_dicts = []
        for row in pod_diverter_rain_and_area:
            pod_diverter_rain_and_area_dicts.append(dict(row))
        pod_diverter_rain_and_area_dicts = [dict(row) for row in pod_diverter_rain_and_area]
        output = []
        for row in water_rights_dicts:
            output_dict = row
            matching_rain_area_row = next(
                    r for r in pod_diverter_rain_and_area_dicts if r['application_number'] == row['application_number']
                )
            output_dict['annual_precip_in'] = matching_rain_area_row['map_1991_2020_in']
            output_dict['drainage_area_sqmi'] = matching_rain_area_row['drainage_area_sqmi']
            for key in output_dict.keys():
                if(type(output_dict[key]) is decimal.Decimal):
                    # little bit of type handling
                    output_dict[key] = float(output_dict[key])
            output.append(output_dict)
        raw_water_rights_csv_data = sort_and_format_unsorted_csv_data(output, nhd, lat, lng, wsr_session)
        water_rights_csv_data = get_adjusted_csv_data(raw_water_rights_csv_data, water_shed, gage = True)
        wsr_senior_diverter_csv = app.db.get_wsr_edited_senior_diverter_csv_by_user_id(g.user_id, id)
        water_rights_csv_data  = overwrite_with_wsr_diverters(water_rights_csv_data, wsr_senior_diverter_csv)
        water_rights_json = get_intermediate_data_json_formatted(water_rights_csv_data, cda=True)
        app.db.save_raw_senior_diverters_gage(g.user_id, id, water_rights_json)
        if request.mimetype == 'text/csv':
            water_rights_csv = get_wsr_water_rights_csv_formatted(water_rights_csv_data, False)
            return Response(water_rights_csv,  mimetype='application/zip'), 200
        else:
            sd_geojson = get_wsr_water_rights_json_formatted(raw_water_rights_csv_data)
            return jsonify(sd_geojson), 200
    elif edited.lower() == "true":
        water_rights_csv_data = app.db.get_gage_edited_senior_diverter_csv_by_user_id(g.user_id, id)
        if (water_rights_csv_data == None):
            raise Exception({"message": "You have not uploaded any edited gage water-rights yet.", "status_code": 400})

        if request.mimetype == 'text/csv':
            water_rights_csv = get_wsr_water_rights_csv_formatted(water_rights_csv_data.get('csv_data'), True)
            return Response(water_rights_csv,  mimetype='application/zip'), 200
        else:
            # modify the senior diverters so it's a geojson Feature Collection
            sd_geojson = get_wsr_water_rights_json_formatted(water_rights_csv_data.get('csv_data'))
            return jsonify(sd_geojson), 200

    else:
        return Response(f"Edited must be true or false not: {edited}"), 422


@projects.route('/wsr/sessions/<int:id>/flow-frequency', methods=['GET'])
@authorization_guard
def get_project_session_flow_frequency(id):
    """
    Retrieve summary information of the proposed point of diversion.
    """
    wsr_flow_frequency_points_of_analysis = app.db.get_wsr_flow_frequency_points_of_analysis(g.user_id, id)

    if (len(wsr_flow_frequency_points_of_analysis) == 0):
        raise Exception({"message": "No points of analysis found for current application.", "status_code": 404})

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        df = pd.DataFrame.from_dict(wsr_flow_frequency_points_of_analysis)
        df.sort_values(by=['application_number', 'rank'], inplace=True)
        application_numbers = df.application_number.unique()

        for application_number in application_numbers:
            application_flow_frequency = df[df.application_number == application_number].copy()

            # Get formatted date information to add to filename. Season applies to the proposed POD and is the same across rows.
            season_start = application_flow_frequency.iloc[1]['direct_div_season_start']
            season_end = application_flow_frequency.iloc[1]['direct_div_season_end']
            season_start_formatted = datetime.combine(season_start, datetime.min.time()).strftime('%b%d')
            season_end_formatted = datetime.combine(season_end, datetime.min.time()).strftime('%b%d')
            filename = f'flow_frequency_analysis_{application_number}_{season_start_formatted}-{season_end_formatted}'

            chart = get_wsr_frequency_analysis(application_flow_frequency, application_number)
            zf.writestr(f'{filename}.png', chart.getvalue())

            demand_title = 'Proposed POD' if application_number == 'Proposed POD' else f"water right, {application_number}"
            application_flow_frequency.rename(columns={'seasonal_volume_af': f'Discharge, acre-ft, at {demand_title}'}, inplace=True)
            # Removing unneeded columns, season only used in filename
            formatted_application_flow_frequency_data = application_flow_frequency.drop(['direct_div_season_start', 'direct_div_season_end'], axis=1)

            water_rights_csv = formatted_application_flow_frequency_data.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
            zf.writestr(f'{filename}.csv', water_rights_csv)

    memory_file.seek(0)
    return send_file(memory_file, download_name='summary.zip', as_attachment=True)

@projects.route('/wsr/sessions/<int:id>/summary', methods=['GET'])
@authorization_guard
def get_project_session_summary(id):
    """
    Retrieve summary information of the proposed point of diversion.
    """
    wsr_summary = app.db.get_wsr_summary(wsr_session_id = id, user_id = g.user_id)
    if (len(wsr_summary) == 0):
        raise Exception({"message": "No summary data found for this project", "status_code": 404})

    raw_gage_timeseries = app.db.get_raw_gage_data(g.user_id, id)
    (_, average_seasonal_gage_flow) = generate_gage_timeseries_seasonal_means(raw_gage_timeseries, wsr_summary[0]['diversion_season'])

    # Since we have now changed the get_wsr_summary function to return less data, we need to return that data
    # First, calculate the gage values that will be the same for each of the rows
    gage_data = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
    wsr_summary_dicts = [dict(row) for row in wsr_summary]
    # Better performance when these are batched together
    pod_rain_and_area = app.db.get_pod_size_and_mean_precip(wsr_session_id = id)
    formatted_data = calculate_wsr_output_values(wsr_summary_dicts, gage_data, pod_rain_and_area, average_seasonal_gage_flow)

    summary_df = pd.DataFrame.from_dict(formatted_data)
    name_map = format_wsr_summary_csv_names(summary_df.loc[0, 'diversion_season'])
    summary_df.rename(columns=name_map, inplace=True)
    summary_df = summary_df[list(name_map.values())]

    min_percentage = summary_df.min()['Percentage of Remaining Unappropriated Water After Proposed POD']

    if request.mimetype == 'text/csv':
        summary_csv = summary_df.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
        return Response(summary_csv,  mimetype='application/zip'), 200
    else:
        json_data = {}
        json_data['summary'] = summary_df.to_json(orient = 'split')
        json_data['min_percentage'] = min_percentage
        return jsonify(json_data), 200

@projects.route('/wsr/sessions/<int:id>/initial-gage-calculations', methods=['GET'])
@authorization_guard
def get_project_session_gage_calculations(id):
    """
    Retrieve summary information of the proposed point of diversion.
    """
    initial_gage_calculations = app.db.get_initial_gage_calculations(id, g.user_id)

    if (len(initial_gage_calculations) == 0):
        raise Exception({ "message": "Gage not found", "status_code": 404 })

    if request.mimetype == 'text/csv':
        gage_calculations_csv = pd.DataFrame.from_dict(initial_gage_calculations).to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
        return Response(gage_calculations_csv,  mimetype='application/zip'), 200
    else:
        return initial_gage_calculations, 200

@projects.route('/wsr/sessions/<int:id>/package', methods=['GET'])
@authorization_guard
@check_query_params(['gage-id'])
def get_project_session_package(params, id):
    """
    Retrieve summary information of the proposed point of diversion.
    """
    wsr_summary = app.db.get_wsr_summary(wsr_session_id = id, user_id = g.user_id)
    if (len(wsr_summary) == 0):
        raise Exception({"message": "No summary data found for this project", "status_code": 404})
    # Since we have now changed the get_wsr_summary function to return less data, we need to return that data
    # First, calculate the gage values that will be the same for each of the rows
    gage_data = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
    wsr_summary_dicts = [dict(row) for row in wsr_summary]
    # Better performance when these are batched together
    pod_rain_and_area = app.db.get_pod_size_and_mean_precip(wsr_session_id = id)
    raw_gage_timeseries = app.db.get_raw_gage_data(g.user_id, id)
    (yearly_mean_gage_timeseries, average_gage_flow) = generate_gage_timeseries_seasonal_means(raw_gage_timeseries, wsr_summary_dicts[0]['diversion_season'])
    formatted_data = calculate_wsr_output_values(wsr_summary_dicts, gage_data, pod_rain_and_area, average_gage_flow)
    wsr_flow_frequency_points_of_analysis = generate_wsr_flow_frequency_points_of_analysis(formatted_data, yearly_mean_gage_timeseries)
    session_data = app.db.get_session_information(g.user_id, id)
    water_rights_csv_data = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)
    formated_session_data = format_session_data(session_data)
    user_uploaded_water_rights = get_wsr_water_rights_csv_formatted(water_rights_csv_data.get('csv_data'), True)
    app_generated_water_rights = get_wsr_water_rights_csv_formatted(water_rights_csv_data.get('raw_senior_diverters'), False)
    csv_data_with_intermediate_values = get_wsr_intermediate_csv_formatted(water_rights_csv_data.get('csv_with_intermediate_values'))
    gage_data = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
    wsr_summary_dicts = [dict(row) for row in wsr_summary]
    pod_rain_and_area = app.db.get_pod_size_and_mean_precip(wsr_session_id = id)
    streamflow_data = app.db.get_gage_streamflow(id, g.user_id)
    files = [
        {
            "data": get_wsr_summary_csv_format(formatted_data),
            "filename": "summary.csv"
        },
        {
            "data": streamflow_data,
            "filename": "gage_streamflow.csv"
        },
    ]
    file_like = BytesIO()
    with zipfile.ZipFile(file_like, mode='w') as zipFileByteObject:
        for file in files:
            csv_data = file["data"]
            if (type(csv_data) != pd.DataFrame):
                csv_data = pd.DataFrame.from_dict(csv_data)
            prepared_csv = csv_data.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
            zipFileByteObject.writestr(file["filename"], prepared_csv)
        add_wsr_flow_frequencies_to_zip(zipFileByteObject, wsr_flow_frequency_points_of_analysis)
        zipFileByteObject.writestr('wsr_project_information.txt', formated_session_data)
        zipFileByteObject.writestr('senior_diverters_edited.csv', user_uploaded_water_rights)
        zipFileByteObject.writestr('senior_diverters_unedited.csv', app_generated_water_rights)
        zipFileByteObject.writestr('senior_diverters_with_seasonal_demand_calculation.csv', csv_data_with_intermediate_values)
        # append some PDF files to this zip
        zipFileByteObject.write('./zipfiles/WSR-Senior-Diverters-HELP.pdf', 'WSR-Senior-Diverters-HELP.pdf')
        zipFileByteObject.write('./zipfiles/Watershed-Candidates-Calculations.pdf', 'Watershed-Candidates-Calculations.pdf')
        zipFileByteObject.write('./zipfiles/Water-Supply-Report-Output-Package.pdf', 'Water-Supply-Report-Output-Package.pdf')
    file_like.seek(0)
    return send_file(file_like,  mimetype='application/zip', as_attachment=True, download_name="csv"), 200

@projects.route('/wsr/sessions', methods=['GET'])
@authorization_guard
def get_all_projects_sessions():
    """
    Get all projects
    """
    result = app.db.get_user_wsr_sessions(g.user_id)
    return result, 200

@projects.route('/wsr/sessions', methods=['POST'])
@authorization_guard
def create_project_session():
    """
    Creates a new session object to save partly-completed proposals.
    """
    # session_id will be used on actual db logic, just setting it for now
    result = app.db.create_wsr_session(user_id = g.user_id)
    return result, 200

@projects.route('/wsr/sessions/<int:id>', methods=['PUT'])
@authorization_guard
def update_project_session(id):
    """
    Update a user session.
    """
    # Body contents below. Wrap in try/except for custom msg.
    try:
        validate(instance=request.json, schema=wsr_schema)
        result = app.db.update_wsr_session_by_id(g.user_id, id, request.json)
        data = request.json
        if("pointOfDiversion" in data.keys() and data['pointOfDiversion'] is not None):
            session = app.db.get_wsr_session_by_id(g.user_id, id)['session']
            nhd_plus_id = session['nhdId']
            app.db.generate_watershed_for_pod_and_store(json.dumps(data['pointOfDiversion']['geometry']), nhd_plus_id, id)
        return result, 200
    # Catch for json schema errors. Pass along the schema error message, may be useful for client side warnings.
    except ValidationError as error:
        raise Exception({"message": str(error), "status_code": 400})
    except BadRequest as error:
        raise Exception({"message": "Include JSON body content with the 'Content-Type' header set to 'application/json'.", "status_code": 400})

@projects.route('/wsr/sessions/<int:id>', methods=['GET'])
@authorization_guard
def get_project_session(id):
    """
    Get all data for a given session.
    """
    result = app.db.get_wsr_session_by_id(g.user_id, id)
    if result == None:
        raise Exception({'message': 'Not found', 'status_code': 404})
    return result, 200

@projects.route('/wsr/sessions/<int:id>', methods=['DELETE'])
@authorization_guard
def delete_project_session(id):
    """
    Get all data for a given session.
    """
    result = app.db.get_wsr_session_by_id(g.user_id, id)
    if result == None:
        raise Exception({'message': 'Not found', 'status_code': 404})
    # delete this session
    app.db.delete_wsr_session_by_id(g.user_id, id)
    return '', 200


@projects.route('/cda/sessions', methods=['POST'])
@authorization_guard
def create_cda_session():
    """
    Creates a new cda session object to save partly-completed cda proposals.
    """
    wsr_session_id = request.json
    try:
        wsr_session_id = int(wsr_session_id)
    except ValueError:
        raise Exception({'message': "WSR session id as an integer is required to create cda session", 'status_code': 400})
    # session_id will be used on actual db logic, just setting it for now
    result = app.db.create_cda_session(g.user_id, wsr_session_id)
    return result, 200

@projects.route('cda/sessions/<int:id>', methods=['GET'])
@authorization_guard
def get_project_cda_session(id):
    """
        Retrieves a user cda session from its ID.
    """
    result = app.db.get_cda_session_by_id(g.user_id, id)
    if result == None:
        raise Exception({'message': 'CDA Session Not found', 'status_code': 404})
    return result, 200

@projects.route('cda/sessions/<int:id>', methods=['PUT'])
@authorization_guard
def update_cda_session(id):
    """
        Uses stored procedure update_cda_session_by_id to update user's stored cda_session.
    """
    try:
        validate(instance=request.json, schema=cda_schema)
        result = app.db.update_cda_session_by_id(g.user_id, id, request.json)
        data = request.json
        if("pointsOfInterest" in data.keys()):
            for poi in data['pointsOfInterest']:
                app.db.generate_watershed_for_poi_and_store(cda_session_id = id, poi_id = poi['id'], lat = poi['lat'], lng = poi['long'], nhdplusid = poi['nhdId'])
        return result, 200
    # Catch for json schema errors. Pass along the schema error message, may be useful for client side warnings.
    except ValidationError as error:
        raise Exception({"message": str(error), "status_code": 400})
    except BadRequest as error:
        raise Exception({"message": "Include JSON body content with the 'Content-Type' header set to 'application/json'.", "status_code": 400})

@projects.route('/cda/sessions/<int:id>/gage-water-rights', methods=['POST'])
@authorization_guard
def process_uploaded_gage_water_rights_csv(id):
    """
    Create a csv file from uploaded. Expects an uploaded file named 'user_demands'. If no file uploaded just pass empty data to user.
    """
    try:
        user_included_upload = ('user_demands' in request.files)
        exists = app.db.check_gage_diverters_exists(g.user_id, id)
        if not exists and user_included_upload:
            gage = app.db.get_wsr_selected_gage_by_id(g.user_id, id)
            wsr_session = app.db.get_wsr_session_by_id(g.user_id, id)['session']
            nhd = gage["nhdplusid"]
            lat = gage["site_location"]["geometry"]["coordinates"][1]
            lng = gage["site_location"]["geometry"]["coordinates"][0]
            date = app.db.get_wsr_start_date_by_id(g.user_id, id)
            if(date is None):
                date = datetime.now(pytz.timezone('America/Vancouver'))
                app.db.update_wsr_session_by_id(g.user_id, id, {'freezeDate': str(date)})
            water_shed = app.db.get_watershed_by_nhd_id(nhd)
            unsorted_water_rights_csv_data = app.db.get_unsorted_senior_diverter_csv(nhd, lat, lng, date)
            raw_water_rights_csv_data = sort_and_format_unsorted_csv_data(unsorted_water_rights_csv_data, nhd, lat, lng, wsr_session)
            water_rights_csv_data = get_adjusted_csv_data(raw_water_rights_csv_data, water_shed, gage = True)
            wsr_senior_diverter_csv = app.db.get_wsr_edited_senior_diverter_csv_by_user_id(g.user_id, id)
            water_rights_csv_data  = overwrite_with_wsr_diverters(water_rights_csv_data, wsr_senior_diverter_csv)
            water_rights_json = get_intermediate_data_json_formatted(water_rights_csv_data, cda=True)
            app.db.save_raw_senior_diverters_gage(g.user_id, id, water_rights_json)

        if (user_included_upload):
            file = request.files['user_demands']
            df = pd.read_csv(file)
            csv_errors = validate_gage_senior_diverter_csv(df)
            if (len(csv_errors) > 0):
                raise Exception({'message': csv_errors, 'status_code': 400})

        # If no upload included, set the data to an empty dataframe (gage doesn't need data)
        else:
            date = app.db.get_wsr_start_date_by_id(g.user_id, id)
            if(date is None):
                date = datetime.now(pytz.timezone('America/Vancouver'))
                app.db.update_wsr_session_by_id(g.user_id, id, {'freezeDate': str(date)})
            app.db.save_raw_senior_diverters_gage(g.user_id, id, '{}')
            df = pd.DataFrame(columns=required_uploaded_csv_columns)

        points_outside_watershed = app.db.validate_gage_senior_diverters_within_upstream_watershed(g.user_id, id, df.to_json(orient='records'))
        if (points_outside_watershed != None):
            raise Exception({ "message": [f"application {row['application_number']} is outside the gage's upstream watershed" for row in points_outside_watershed] , "status_code" : 400})
        intermediate_wsr_values = calculate_cda_intermediate_values(df.copy())
        result = app.db.save_gage_edited_senior_diverters(g.user_id, id, df.to_json(orient='records'), intermediate_wsr_values.to_json(orient='records'))
        # Update flag in session
        app.db.update_cda_session_by_id(g.user_id, id, {'hasGageEditedWaterRights': True})
        return result, 200
    except BadRequest as error:
        raise Exception({'message': 'Could not read file.', 'status_code': 400})

@projects.route('cda/sessions/<int:id>/thresholds', methods=['GET'])
@authorization_guard
def calculate_cda_thresholds(id):
    """
        Get Thresholds values (ratios, maximum cumulative diversion, minimum bypass flow) for the cda
        steps:
            1. Calculate senior diverter timeseries from the user-uploaded Gage's diverters
            2. Calculate proration ratios using gage/poi nhdIds, get_poi_ratio_data and calculate_cda_ratio
            3. Estimate the mean annual unimpaired flow at the gage by unimpairing it
            4. Calculate minimum bypass flow.
            5. Use peaks over threshold/ yearly study to get 1.5-yr instantaneous peak flow.
            6. Calculate maximum cumulative diversion
            7. Return the relevant data as json object to be displayed as a table to the front-end
        Threshold table data saved to cda_session object, and returned to front end to display in the table.
        Initially sets "regional" data to the same as the output data, user can overwrite the values but not the regional values
    """
    # Set up, get wsr gage and cda session
    thresholds_data = []
    try:
        gage = app.db.get_wsr_selected_gage_by_id(g.user_id, id)
        session = app.db.get_cda_session_by_id(g.user_id, id)['session']
        pois = []
        if("pointsOfInterest" in session):
            pois = session['pointsOfInterest']
    except Exception as e:
        raise Exception({"message": f'{str(e.__str__())}\nIssue with getting session and gage', 'status_code': 404})
    #Step 1 - calculate unimpaired data series
    try:
        senior_diverters = app.db.get_gage_calculated_senior_diverters_by_user_id(g.user_id, id)
        if(senior_diverters == None):
            raise Exception("User has not uploaded Senior Diverters!")
        gage_start_date = int(gage['historic_water_record'].split('-')[0])
        gage_diversion_timeseries = generate_senior_diverter_ts(senior_diverters, gage_start_date)
        result = app.db.unimpair_gage_timeseries(g.user_id, id, json.dumps(gage_diversion_timeseries))
        if(result != "COMPLETE"):
            raise Exception({"message" : "Could not successfully unimpair gage timeseries", "status_code": 500})
    except Exception as e:
        raise Exception({"message": f'{str(e.__str__())}\nIssue with calculating unimpaired gage data series', 'status_code': 400})
    #Step 2: Calculate poi proration ratios
    try:
        #Get required data for gage
        gage_ratio_raw = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
        gage_ratio_data = {"poiId": -2,
                           "drainageArea": gage_ratio_raw['drainage_area_sqmi'],
                           "averagePrecipitation": gage_ratio_raw['map_1991_2020_in']}
        thresholds_data.append(gage_ratio_data)
        #Get required data for user's pod
        wsr_session = app.db.get_wsr_session_by_id(g.user_id, id)
        pod_nhdid = wsr_session['session']['nhdId']
        if(pod_nhdid == None):
            raise Exception("User does not have a selected point of diversion in session")
        pod_ratio_raw = app.db.get_pod_size_and_mean_precip(wsr_session_id = id)
        pod_ratio_data = calculate_cda_ratio(pod_ratio_raw, gage_ratio_raw)
        pod_ratio_data['poiId'] = -1
        thresholds_data.append(pod_ratio_data)
        #Calculate ratios for poi's
        for poi in pois:
            poi_ratio_raw = app.db.get_poi_size_and_mean_precip(cda_session_id = id, poi_id = poi['id'])
            poi_ratio_data = calculate_cda_ratio(poi_ratio_raw, gage_ratio_raw)
            poi_ratio_data['poiId'] = poi['id']
            thresholds_data.append(poi_ratio_data)
    except Exception as e:
        raise Exception({"message": f'{str(e)}\nLikely did not have necessary poi data' , "status_code": 404})
    # Get the mean unimpaired yearly flow and calculate minimum bypass flow
    try:
        unimpaired_gage_data = app.db.get_unimpaired_gage_data(g.user_id, id)
        gage_yearly_mean_cfs = calculate_yearly_mean(unimpaired_gage_data)
        gage_february_median = calculate_feb_median(unimpaired_gage_data)
        gage_yearly_mean_af = gage_yearly_mean_cfs/AFD_TO_CFS * 365
        #position 0 = gage
        thresholds_data[0]['meanAnnualUnimpairedVolumeAf'] = gage_yearly_mean_af
        thresholds_data[0]['meanAnnualUnimpairedVolumeCfs'] = gage_yearly_mean_cfs
        for i in range(1, len(thresholds_data)):
            if(i > 1 and int(pois[i-2]['class']) == 3):
                continue
            thresholds_data[i]['meanAnnualUnimpairedVolumeAf'] = gage_yearly_mean_af * thresholds_data[i]['ratio']
            poi_mean_cfs = gage_yearly_mean_cfs * thresholds_data[i]['ratio']
            thresholds_data[i]['meanAnnualUnimpairedVolumeCfs'] = poi_mean_cfs
            minimum_bypass_flow = calculate_mbf(thresholds_data[i]['drainageArea'], poi_mean_cfs)
            #Check if the poi is class 2 and calculate february median, override as necessary
            if(i == 1 and session['podStreamClass'] == 2):
                pod_february_median = gage_february_median * thresholds_data[i]['ratio']
                thresholds_data[i]['februaryMedian'] = pod_february_median
                if(pod_february_median > minimum_bypass_flow):
                    minimum_bypass_flow = pod_february_median
            if(i > 1 and int(pois[i-2]['class']) == 2):
                poi_february_median = gage_february_median * thresholds_data[i]['ratio']
                thresholds_data[i]['februaryMedian'] = poi_february_median
                if(poi_february_median > minimum_bypass_flow):
                    minimum_bypass_flow = poi_february_median
            thresholds_data[i]['minimumBypassFlow'] = minimum_bypass_flow
            thresholds_data[i]['minimumBypassFlowRegional'] = minimum_bypass_flow
    except Exception as e:
        raise Exception({"message": f'{str(e)}\nUnable to calculate minimum bypass flow.' , "status_code": 400})
    try:
        #Implemented with peaks-over-threshold for now
        peaks_thresholds = peaks_and_threshold(unimpaired_gage_data)
        recurrence_intervals = generate_recurrence_intervals(peaks_thresholds['peaks'], peaks_thresholds['record_years'])
        instantaneous_flow = plot_and_find_instantaneous_peak_flow(peaks_thresholds['peaks'], recurrence_intervals)['curve_fit']
        #Don't worry about calculation for class 3 pod's
        if(session['podStreamClass'] != 3):
            thresholds_data[1]['maximumCumulativeDiversion'] = calculate_mcd(instantaneous_flow, thresholds_data[i]['ratio'])
            thresholds_data[1]['maximumCumulativeDiversionRegional'] = thresholds_data[1]['maximumCumulativeDiversion']
        for i in range(2, len(thresholds_data)):
            if(pois[i-2]['anadromy'] != 'Above'):
                thresholds_data[i]['maximumCumulativeDiversion'] = calculate_mcd(instantaneous_flow, thresholds_data[i]['ratio'])
                thresholds_data[i]['maximumCumulativeDiversionRegional'] = thresholds_data[i]['maximumCumulativeDiversion']
    except Exception as e:
        raise Exception({"message": f'{str(e)}\nUnable to calculate maximum cumulative diversion' , "status_code": 400})

    app.db.update_cda_session_by_id(g.user_id, id, {"thresholdTableData" : thresholds_data})

    return thresholds_data, 200

@projects.route('cda/sessions/<int:id>/poi/<int:poi_id>/daily-flow', methods=['GET'])
@authorization_guard
def daily_flow_study(id, poi_id):
    """
        The big one : Daily flow study. The following is done:
        1. Scale unimpaired gage time series to the POI using the poi proration ratio
        2. Get the POI senior diverters and re-impair the time-series
        3. Using generated time-series's, perform the spawning, rearing and passage daily flow study (calculate monthly data)
        4. Perform the natural flow variability daily flow study calculations (calculate 1.5-year instantaneous peaks)
        5. If the class II/III criteria is satisfied, perform monthly analysis of February median flow
        6. Bring all data together, store in user cda session and return to front end.
    """
    #Initial set up, get the session and associated POI data
    daily_flow_study_results = {'poiId' : poi_id}
    try:
        session = app.db.get_cda_session_by_id(g.user_id, id)['session']
        if(not ('thresholdTableData' in session) or session['thresholdTableData'] == None):
            raise Exception("User must have calculated threshold data to perform daily flow study.")
        unimpaired_gage_data = app.db.get_unimpaired_gage_data(g.user_id, id)
        if(unimpaired_gage_data == None):
            if(not session['regionalCriteria']):
                app.db.unimpair_gage_timeseries(g.user_id, id, json.dumps({1800: [0]*365}))
                unimpaired_gage_data = app.db.get_unimpaired_gage_data(g.user_id, id)
            else:
                raise Exception("An unimpaired gage time-series doesn't exist for the user.")
        poi = next((p for p in session['pointsOfInterest'] if p["id"] == poi_id), None)
        if(poi == None):
            raise Exception(f"No poi at given index : {poi_id}")
        poi_threshold = next((p for p in session['thresholdTableData'] if p["poiId"] == poi_id), None)
        if(poi_threshold == None):
            raise Exception(f"Poi at given index doesn't have associated threshold data : {poi_id}")
        if(not 'ratio' in poi_threshold):

            gage_ratio_raw = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
            #Get required data for user's pod
            wsr_session = app.db.get_wsr_session_by_id(g.user_id, id)
            pod_nhdid = wsr_session['session']['nhdId']
            if(pod_nhdid == None):
                raise Exception("User does not have a selected point of diversion in session")
            poi_ratio_raw = app.db.get_poi_size_and_mean_precip(cda_session_id = id, poi_id = poi_id)
            poi_ratio_data = calculate_cda_ratio(poi_ratio_raw, gage_ratio_raw)
            poi_ratio_data['poiId'] = poi['id']
            poi_threshold = poi_threshold | poi_ratio_data
    except Exception as e:
        raise Exception({"message": f'{str(e)}\nIssue with set up for daily flow study', 'status_code': 404})
    #Scale unimpaired gage time series to the POI
    try:
        unimpaired_poi_ts = scale_gage_ts_to_poi(unimpaired_gage_data, poi_threshold['ratio'])
        unimpaired_start_year = datetime.strptime(unimpaired_poi_ts.iloc[0]['date'], '%d-%m-%Y').year
        unimpaired_end_year = datetime.strptime(unimpaired_poi_ts.iloc[len(unimpaired_poi_ts.index)-1]['date'], '%d-%m-%Y').year
        daily_flow_study_results['yearsOfRecord'] = unimpaired_end_year-unimpaired_start_year
        app.db.save_unimpaired_poi_ts(g.user_id, id, poi_id, unimpaired_poi_ts.to_json(orient='records'))
    except Exception as e:
        raise Exception({"message": f'{str(e)}\nUnable to generate unimpaired poi time series', 'status_code': 400})
    #Get the senior diverters for the poi and impair
    try:
        wsr_senior_diverters = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)['csv_data']
        if(wsr_senior_diverters == {}):
            raise Exception("No wsr senior diverters found - is the wsr section complete?")
        (upstream_senior_diverters, upstream_senior_diverters_with_pod) = get_senior_diverters_upstream_of_poi(wsr_senior_diverters, poi, session)
        currently_upstream = []
        onstream_storage_upstream_diverters = {}
        for diverter in upstream_senior_diverters_with_pod:
            if(diverter['analysis_label'] == 'Proposed POD'):
                pod_upstream_diverters = app.db.get_proposed_pod_upstream_diverters(
                    session_id = id,
                    current_upstream_diverters = json.dumps(currently_upstream)
                )
            else:
                pod_upstream_diverters = app.db.get_onstream_pod_upstream_diverters(
                    water_right_id = int(diverter['wr_water_right_id']),
                    current_upstream_diverters = json.dumps(currently_upstream)
                )
            pod_upstream_diverters = [int(x['order_upstream_to_downstream']) for x in pod_upstream_diverters]
            onstream_storage_upstream_diverters[diverter['order_upstream_to_downstream']] = pod_upstream_diverters
            currently_upstream.append({'order_upstream_to_downstream' : diverter['order_upstream_to_downstream'],
                                       'lat': diverter['latitude'],
                                       'lng': diverter['longitude']})
        gage_ratio_raw = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)
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
        app.db.save_impaired_poi_timeseries(diverters = json.dumps(impair_result_diverters),
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
    session = app.db.get_cda_session_by_id(g.user_id, id)['session']
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
        app.db.update_cda_session_by_id(g.user_id, id, {'dailyFlowData' : daily_flow_data})
    else:
        #Otherwise just make a new session entry for the daily flow study
        app.db.update_cda_session_by_id(g.user_id, id, {'dailyFlowData' : [daily_flow_study_results]})

    return daily_flow_study_results, 200

@projects.route('cda/sessions/<int:id>/package', methods=['GET'])
@authorization_guard
@check_query_params(['email'])
def generate_cda_session_package(params, id):
    """
        Generates the cda session package and zips together for the user. The steps are as follows:
        1. Get the gage_senior_diverter_csv data and format into a nice format
        2. Gets the raw gage data, unimpaired gage data, and poi daily time series amounts
        3. Re-runs and generates peaks over threshold data for generation of tables and an output plot
        4. Generates the threshold table data in a user-friendly table format
        5. Gets the senior diverter and daily flow study time series for each POI and formats into csvs.
        6. Generates the overall daily flow study summary output csvs in a user-friendly format
    """
    email, = params
    try:
        #Get necessary data from database first off
        gage_csvs = app.db.get_gage_senior_diverter_csv_by_user_id(g.user_id, id)
        raw_gage_timeseries = app.db.get_raw_gage_data(g.user_id, id)
        if(raw_gage_timeseries == None):
            raise Exception(f"Unable to find gage time series data for selected gage.")
        unimpaired_gage_data = app.db.get_unimpaired_gage_data(g.user_id, id)
        if(unimpaired_gage_data == None):
            raise Exception("Must have uploaded gage diverters for output package formatting.")
        poi_time_seriess = {}
        poi_unimpaired = {}
        cda_session = app.db.get_cda_session_by_id(g.user_id, id)['session']
        wsr_senior_diverters = app.db.get_senior_diverter_csv_by_user_id(g.user_id, id)['csv_data']
        if(wsr_senior_diverters == {}):
            raise Exception("No wsr senior diverters found - is the wsr section complete?")
        diverters_upstream_of_onstream_storage = {}
        for poi in cda_session['pointsOfInterest']:
            poi_time_seriess[poi['id']] = app.db.get_poi_ts(g.user_id, id, poi['id'])
            poi_unimpaired[poi['id']] = poi_time_seriess[poi['id']]['unimpaired']
            (upstream_senior_diverters, upstream_senior_diverters_with_pod) = get_senior_diverters_upstream_of_poi(wsr_senior_diverters, poi, cda_session)
            currently_upstream = []
            onstream_storage_upstream_diverters = {}
            for diverter in upstream_senior_diverters_with_pod:
                if(diverter['analysis_label'] == 'Proposed POD'):
                    pod_upstream_diverters = app.db.get_proposed_pod_upstream_diverters(
                        session_id = id,
                        current_upstream_diverters = json.dumps(currently_upstream)
                    )
                else:
                    pod_upstream_diverters = app.db.get_onstream_pod_upstream_diverters(
                        water_right_id = int(diverter['wr_water_right_id']),
                        current_upstream_diverters = json.dumps(currently_upstream)
                    )
                pod_upstream_diverters = [int(x['order_upstream_to_downstream']) for x in pod_upstream_diverters]
                onstream_storage_upstream_diverters[diverter['order_upstream_to_downstream']] = pod_upstream_diverters
                currently_upstream.append({'order_upstream_to_downstream' : diverter['order_upstream_to_downstream'],
                                        'lat': diverter['latitude'],
                                        'lng': diverter['longitude']})
            diverters_upstream_of_onstream_storage[poi['id']] = onstream_storage_upstream_diverters
        gage_ratio_raw = app.db.get_gage_size_and_mean_precip(wsr_session_id = id)

    except Exception as e:
        raise Exception({"message": f'{str(e)}\nUnable to get data from the database.', 'status_code': 404})
    # Generate output package from given data
    if(email != 'test'):
        generate_output_process = Process(
            target = generate_cda_output_package,
            args = (gage_csvs, raw_gage_timeseries,unimpaired_gage_data, poi_unimpaired, poi_time_seriess,cda_session,wsr_senior_diverters,diverters_upstream_of_onstream_storage, gage_ratio_raw, email,id)
        )
        generate_output_process.start()
    return "emailing", 202