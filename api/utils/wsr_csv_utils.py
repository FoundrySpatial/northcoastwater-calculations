import sys
import pandas as pd
import numpy as np
import datetime
import pytz
from collections import namedtuple
from utils.helpers import is_integer, is_float_in_range
from matplotlib.figure import Figure
import scipy.optimize as opt
from sklearn.metrics import r2_score
from io import BytesIO
import csv
from flask import current_app as app
from shapely.geometry import Point, Polygon
from psycopg2.extras import RealDictRow
# import matplotlib.pyplot as plt

AFD_TO_CFS = 1.5125/3

MONTH_TO_NUMBER = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

MAX_SENIOR_DIVERTERS = 4500
Range = namedtuple('Range', ['start', 'end'])
policy_season = Range(start=datetime.datetime(
    2018, 12, 15), end=datetime.datetime(2019, 3, 31))
frost_season = Range(start=datetime.datetime(
    2019, 3, 15), end=datetime.datetime(2019, 4, 30))
non_policy_season = Range(start=datetime.datetime(
    2019, 4, 1), end=datetime.datetime(2019, 12, 14))
direct_div_date_cols = ["direct_div_season_start_month", "direct_div_season_end_month",
                        "direct_div_season_start_day", "direct_div_season_end_day"]
storage_date_cols = ["storage_season_start_month", "storage_season_end_month",
                     "storage_season_start_day", "storage_season_end_day"]

intermediate_table_dropped_columns = ["direct_div_season_start_month", "direct_div_season_start_day", "direct_div_season_end_month", "direct_div_season_end_day",
    "storage_season_start_month", "storage_season_start_day", "storage_season_end_month", "storage_season_end_day"]

required_uploaded_csv_columns = [
    "analysis_label",
    "order_upstream_to_downstream",
    "application_number",
    "appl_pod",
    "wr_water_right_id",
    "water_right_type",
    "water_right_status",
    "application_primary_owner",
    "pod_type",
    "pod_count",
    "source_name",
    "latitude",
    "longitude",
    "drainage_area_sqmi",
    "annual_precip_in",
    "use_codes",
    "priority_date",
    "direct_div_season_start_month",
    "direct_div_season_start_day",
    "direct_div_season_end_month",
    "direct_div_season_end_day",
    "storage_season_start_month",
    "storage_season_start_day",
    "storage_season_end_month",
    "storage_season_end_day",
    "max_storage_af",
    "face_amount_af",
    "max_rate_of_diversion_cfs",
    "minimum_bypass_flow_cfs",
    "seasonal_demand_af",
    "overwrite_seasonal_demand_af_justification",
    "comments",
]

intermediate_data_columns = [
    "analysis_label",
    "order_upstream_to_downstream",
    "application_number",
    "appl_pod",
    "wr_water_right_id",
    "water_right_type",
    "water_right_status",
    "application_primary_owner",
    "pod_type",
    "pod_count",
    "source_name",
    "latitude",
    "longitude",
    "drainage_area_sqmi",
    "annual_precip_in",
    "use_codes",
    "priority_date",
    "max_storage_af",
    "face_amount_af",
    "max_rate_of_diversion_cfs",
    "minimum_bypass_flow_cfs",
    "seasonal_demand_af",
    "overwrite_seasonal_demand_af_justification",
    "comments",
    "storage_season_start",
    "storage_season_end",
    "days_of_storage",
    "direct_div_season_start",
    "direct_div_season_end",
    "days_of_diversion",
    "diversion_amount_af",
    "diversion_per_day_af",
    "overlapping_days_of_proposed_and_policy_season",
    "overlapping_days_of_storage_and_policy_season",
    "overlapping_days_of_storage_and_proposed_season",
    "overlapping_proposed_and_days_of_direct_diversion",
    "overlapping_days_of_direct_diversion_and_policy_season",
    "overlapping_days_of_proposed_and_frost_season",
    "frost_demand_af",
    "wr_seasonal_demand_af",
]


wsr_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://example.com/example.json",
    "type": "object",
    "default": {},
    "title": "WSR Schema",
    "definitions": {
        "LatType": {
            "type": "number",
            "minimum": -90,
            "maximum": 90
        },
        "LongType": {
            "type": "number",
            "minimum": -180,
            "maximum": 180
        },
        "UnitValue": {
            "type": "object",
            "properties": {
                "unit": {
                    "type": "string"
                },
                "value": {
                    "type": "number"
                },
            }
        },
        "MetaType": {
            "type": "object",
            "properties": {
                "created": {
                    "type": "string"
                },
                "modified": {
                    "type": "string"
                }
            }
        },
        "Point": {
            "type": ["object" , "null"],
            "properties": {
                "lat": { "$ref": "#/definitions/LatType" },
                "long": { "$ref": "#/definitions/LongType" }
            }
        },
    },
    "properties": {
        "meta": { "$ref": "#/definitions/MetaType" },
        "status": {"type": "string"},
        "title": {"type": "string"},
        "requiresCda": {"type": "boolean"},
        "hasMinBypassThreshold": {"type": "boolean"},
        "hasSeniorDiverters": {"type": "boolean"},
        "description": {"type": "string"},
        "pointOfDiversion": { "$ref": "#/definitions/Point" },
        "seasonOfDiversionStart": {"type": ["string", "null"]},
        "seasonOfDiversionEnd": {"type": ["string", "null"]},
        "volumeOfDiversion": {"$ref": "#/definitions/UnitValue"},
        "rateOfDiversion": {"$ref": "#/definitions/UnitValue"},
        "minBypassThreshold": {"$ref": "#/definitions/UnitValue"},
        "selectedGage" : {"type": "integer"},
        "freezeDate": {"type": "string"},
    },
    "additionalProperties": True,
}

def has_full_row(row, cols):
    return not row[cols].isnull().values.any()

def format_wsr_summary_csv_names (proposed_diversion_season):
    """ Function to map sql names to csv titles.

    Args:
        proposed_diversion_season (str): String of the proposed PODs season

    Returns:
        dict: Dictionary of name mapping
    """
    return  {
        "application_number": "POD (Application ID)",
        "gage_area_sqmi": "Watershed Area Above Gage (sq mi)",
        "gage_map_1991_2020_in": "Avg Annual precip of wshd above Gage (in)",
        "gage_seasonal_flow_af": "Avg Unappropriated Seasonal Flow volume recorded at the Gage (AF) during proposed season",
        "area_sqmi": "Watershed Area Above POD (sq mi)",
        "map_1991_2020_in": "Avg Annual precip of wshd above POD (in)",
        "ratio1": "Streamflow Scaling Ratio",
        "diversion_season": "Diversion Season",
        "seasonal_unimpaired_flow_volume_af": "Seasonal Unappropriated Flow Volume (AF)",
        "seasonal_demand_before_new_water_right_af": f"{proposed_diversion_season} Seasonal Demand Before Proposed POD (AF)",
        "seasonal_upstream_demand_af": f"{proposed_diversion_season} Upstream Demand Before Proposed POD (AF)",
        "remaining_unimpaired_discharge_before_new_water_right_af": "Remaining unappropriated flow, (AF) Before Proposed POD",
        "percent_remaining_unappropriated_water_before_new_water_right": "Percentage of remaining unappropriated water Before Proposed POD",
        "additional_impairment_caused_by_new_water_right_af": "Additional Impairment Caused By Proposed POD (AF)",
        "remaining_unimpaired_discharge_after_new_water_right_af": "Remaining Unappropriated flow, (AF) After Proposed POD",
        "percent_remaining_unappropriated_water_after_new_water_right": "Percentage of Remaining Unappropriated Water After Proposed POD",
        "percent_change_caused_by_new_water_right": "Percent Change Caused By Proposed POD",
        "ratio_of_project_total_diversion_to_impaired_streamflow": "Ratio of Project Demand to Remaining Unappropriated Water Supply at Diverter"
    }

def get_wsr_summary_csv_format(raw_data):
    summary_df = pd.DataFrame.from_dict(raw_data)
    name_map = format_wsr_summary_csv_names(summary_df.loc[0, 'diversion_season'])
    summary_df.rename(columns=name_map, inplace=True)
    return summary_df[list(name_map.values())]

def format_output_dates(row, column):
    if(pd.isna(row[column]) or row[column] == ''):
        return None
    else:
        values = row[column].split('-')
        return f"{values[1]}/{values[0]}/{values[2]}"

def get_wsr_intermediate_csv_formatted(raw_data):
    df = pd.DataFrame.from_dict(raw_data)
    df.rename(columns={"wr_seasonal_demand":"wr_seasonal_demand_af"}, inplace=True)
    df = df[[*intermediate_data_columns]]
    date_columns = ['storage_season_start', 'storage_season_end', 'direct_div_season_start', 'direct_div_season_end']
    for column in date_columns:
        df[column] = df.apply(
            lambda x: format_output_dates(x, column), axis=1)
    return df.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)

def get_wsr_water_rights_csv_formatted(raw_data, edited=False):
    df = pd.DataFrame.from_dict(raw_data)
    if(df.empty):
        #If dataframe is empty (happens occasionally for gage cases), return empty df with columns
        df = pd.DataFrame(columns = required_uploaded_csv_columns)
        return df.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
    df = df[[*required_uploaded_csv_columns]]
    if not edited:
        df['priority_date'] = df['priority_date'].apply(lambda x: '' if x is None else x)
        df['wr_water_right_id'] = df['wr_water_right_id'].fillna(-1).astype(int)
        df['wr_water_right_id'] = df['wr_water_right_id'].apply(lambda x: '' if x == -1 else x)
        df['use_codes'] = df['use_codes'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    return df.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)

def filter_out_status_wrs(row):
    """
    Filter out water rights with status that indicates it is not diverting water
    """
    status = row['water_right_status']
    return status !='Cancelled' and status != 'Closed' and status != 'Completed' and status != 'Rejected' and status != 'Revoked' and status != 'Withdrawn'

def filter_out_uns(row):
    """
    Filter out UN's (water rights not necessary for analysis)
    """
    application_id = row['application_number']
    return application_id is None or not application_id.startswith('UN')

def add_comment_if_wr_inactive(row):
    """
    Inform user that inactive water right will require further investigation
    """
    if(row['water_right_status'] == 'Inactive'):
        if(pd.isna(row['comments'])):
            row['comments'] = ''
        return (row['comments'] + "Warning: Inactive water right. Look into EWRIMS data for more information.").strip()
    return row['comments']

def add_comment_if_wr_riparian_case(row):
    """
    Edge case with riparian data -> if a riparian water right, comment if storage exists.
    """
    if(row['riparian'] and (row['max_storage_af'] > 0 or pd.notna(row['storage_season_start_month']))):
        if(pd.isna(row['comments'])):
            row['comments'] = ''
        return (row['comments'] + " Warning: Riparian water right with storage, ensure this is accurate.").strip()
    return row['comments']

def is_point_inside_polygon(row, polygon):
    """
    Helper function to see if a row is inside a given polygon (when it sohuld not be)
    Args:
        row: a dataframe row with longitude and latitude columns and an analysis label
        polygon: a polygon to check the existance of the row argument in it
    Returns:
        A boolean indicating the existance of a row in the supplied polygon and if it is a mainstem poa or within project extent
    """
    point = Point(row['longitude'], row['latitude'])
    return (polygon.contains(point)) and (row['analysis_label'] == "Inside Project Extent" or row['analysis_label'] == "Mainstem POA")

def get_adjusted_csv_data(raw_data, water_shed, gage = False):
    """
    Function to re-label and move inaccuratley labled points by our DFS SQL endpoint that are on isolated stream reaches or snapped to an outside stream despite being within the upstream polygon.

    Also adds comments for edge cases and does some error checking on SQL CSV.
    Args:
        raw_data: raw csv data as returned by the database
        water_shed: a watershed multipolygon for the proposed NHDID (just upstream)
        gage: a boolean for if the supplied data is for a gage instead of a pod. If this is the case then downstream and the pod itself will be ignored as all that matters is upstream demand for impairment
    Returns:
        Returns two values the first is a Boolean if bad points existed within the supplied raw data for the given polygon
        Second is edited data that only exists if there was bad points in the data. This was done to speedup runtime in the common case where there is no bad points.
        Also to eliminate the likely hood of unintended side effects occuring by running this on data with no bad points (there should be none but better safe then sorry)
    """
    # Turn off a warning for modifying a copy of a spliced dataframe
    pd.options.mode.chained_assignment = None

    df = pd.DataFrame.from_dict(raw_data)

    if(df.empty):
        return []
    #Filter out values with water_right_status that indicate unused water right
    df = df[df.apply(lambda x : filter_out_status_wrs(x), axis=1)]
    df = df[df.apply(lambda x: filter_out_uns(x), axis = 1)]
    df.reset_index(drop=True, inplace=True)
    df['comments'] = df.apply(lambda x : add_comment_if_wr_inactive(x), axis=1)
    #Check for weird riparian edge cases
    df['comments'] = df.apply(lambda x : add_comment_if_wr_riparian_case(x), axis=1)
    #Do a check for points in Frost/Irrigation which are in the wrong season
    #Calculate diversion season values
    diversion_values = df.copy()

    diversion_values = diversion_values.replace({pd.NA: np.nan})
    diversion_values.fillna(np.nan, inplace=True)
    diversion_values["direct_div_season_start"] = diversion_values.apply(
        lambda x: get_start_date(x, direct_div_date_cols), axis=1)
    diversion_values["direct_div_season_end"] = diversion_values.apply(
        lambda x: get_end_date(x, direct_div_date_cols), axis=1)
    diversion_values["storage_season_start"] = diversion_values.apply(
        lambda x: get_start_date(x, storage_date_cols), axis=1)
    diversion_values["storage_season_end"] = diversion_values.apply(
        lambda x: get_end_date(x, storage_date_cols), axis=1)
    df['comments'] = diversion_values.apply(
        lambda x: add_frost_irrigation_comments(x), axis=1
    )
    del diversion_values

    # Will error out if the watershed geojson is not of the expected format first 0 is for first polygon in multipolygon and second 0 is for outer ring
    # Will fail if our watersheds include inner holes (which none currently do)
    ws_poly = Polygon(water_shed['watershed']['geometry']['coordinates'][0][0])

    # detect bad points and give a temp Boolean field that will be dropped later
    df['IsBadPoint'] = df.apply(is_point_inside_polygon, axis=1, polygon=ws_poly)
    df_bad_points = df[df['IsBadPoint']]
    df_good_points = df[~df['IsBadPoint']]
    # No bad points we are done here
    if df_bad_points.empty:
        df = df.drop('IsBadPoint', axis=1)
        #Reindex here
        df['order_upstream_to_downstream'] = df.index + 1
        if gage:
            values_to_drop = ['Proposed POD', 'Downstream Flow Path', 'Inside Project Extent', 'Upstream of Downstream Flow Path']
            df = df[~df['analysis_label'].isin(values_to_drop)]
            df['analysis_label'] = df['analysis_label'].apply(
                lambda x: "Upstream of Gage")
        converted_list = [RealDictRow(row) for row in df.to_dict(orient='records')]
        return converted_list

    # Free up this memory now that it has been split
    del df

    # Fix inaccurate fields
    df_bad_points['comments'] = 'Likely out of order, this WR is on an isolated stream reach and therefore cannot be accurately ordered' + df_bad_points['comments']
    df_bad_points['analysis_label'] = 'Upstream of POD'
    df_bad_points['analysis_label_map'] = 2

    # find proposed POD at correct index
    good_index_points = df_good_points.reset_index(drop=True)
    index_to_insert = good_index_points.loc[good_index_points['analysis_label'] == "Proposed POD"].index[0]

    # Concatenate the dataframesbakc into the final DF above proposed POD
    result_df = pd.concat([df_good_points.iloc[:index_to_insert], df_bad_points, df_good_points.iloc[index_to_insert:]], ignore_index=True)

    # Fix uncessesary column and the index that got messed up from moving rows around
    result_df = result_df.drop('IsBadPoint', axis=1)
    result_df = result_df.reset_index(drop=True)
    result_df['order_upstream_to_downstream'] = result_df.index + 1
    # Another weird hack to fix the fact that pandas does not like reading ints with holes in it makes all columns of type "Object" which would be inefficient normally
    # But should be fine given no more processing is happening (also the dataframes cant get very large)
    result_df = result_df.replace(np.nan, None)
    result_df.fillna(-999999, inplace=True)
    result_df = result_df.convert_dtypes()
    result_df = result_df.replace(-999999, None)
    result_df = result_df.replace({np.nan: None})

    if gage:
        values_to_drop = ['Proposed POD', 'Downstream Flow Path', 'Inside Project Extent', 'Upstream of Downstream Flow Path']
        result_df = result_df[~result_df['analysis_label'].isin(values_to_drop)]
        result_df['analysis_label'] = result_df['analysis_label'].apply(
                lambda x: "Upstream of Gage")
    # mock data comming from database using psycopg2 datatype so the pipeline remains the same
    converted_list = [RealDictRow(row) for row in result_df.to_dict(orient='records')]
    return converted_list

def get_intermediate_data_json_formatted(raw_data, cda = False):
    df = pd.DataFrame.from_dict(raw_data)
    if(cda and 'analysis_label' in df.columns):
        df['analysis_label'] = df['analysis_label'].apply(
            lambda x: "Upstream of Gage")
    # Ensure priority dates aren't turned into times since unix epoch or anything crazy like that
    if('priority_date' in df.columns):
        df['priority_date'] = df['priority_date'].apply(str)
    return df.to_json(orient='records')

def get_wsr_water_rights_json_formatted(raw_data):
    sd_geojson = {
        'type': 'FeatureCollection',
        'features': []
    }
    label_mapping = {
        'Upstream of Downstream Flow Path': 1,
        'Upstream of POD': 2,
        'Proposed POD': 3,
        'Downstream Flow Path': 4,
        'Inside Project Extent': 5
    }
    for idx, seniordiv in enumerate(raw_data):
        seniordiv['priority_date'] = str(seniordiv['priority_date'])
        if (seniordiv['analysis_label'] in label_mapping):
            seniordiv['analysis_label_map'] = label_mapping[seniordiv['analysis_label']]
        else:
            seniordiv['analysis_label_map'] = 1 # assume it's upstream of Downstream Flow Path
        sd_geojson['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [
                    seniordiv['longitude'], seniordiv['latitude']
                ]
            },
            'id': idx,
            'properties': seniordiv.copy()
        })
    return sd_geojson

def format_json_units(original_dict, keys_to_change):
    """Function to change a dictionary value to a sub-dictionary with vcalue and unit.
    original_dict - dictionary to change
    keys_to_change - list of the values to change, e.g
        [
            {
                "key_name": "volume_af",
                "unit_name": "af",
                "new_key_name": "volume",
            },
            ...
        ]
    """
    dict = original_dict.copy()
    for change in keys_to_change:
        dict[change["new_key_name"]] = {
            "unit": change["unit_name"],
            "value": dict[change["key_name"]]
        }
        dict.pop(change["key_name"])
    return dict

#Rules for seasonal_demand_af if user populated: overwrite_seasonal_demand_af_justification must be populated
def seasonal_demand_populated_check(row):
    if(pd.notna(row['seasonal_demand_af'])):
        if(pd.notna(row['overwrite_seasonal_demand_af_justification']) and row['overwrite_seasonal_demand_af_justification'] != ""):
            return False
        else:
            return True
    else:
        return False

def has_valid_date(row, prefix, errors):
    try:
        datetime.datetime(year=2018, month=int(row.loc[f'{prefix}_season_start_month']), day=int(
            row.loc[f'{prefix}_season_start_day']))
        datetime.datetime(year=2018, month=int(
            row.loc[f'{prefix}_season_end_month']), day=int(row.loc[f'{prefix}_season_end_day']))
        return errors
    except (ValueError, TypeError):
        return errors + [{"error": f'A {prefix} date is invalid.'}]

def get_wsr_frequency_analysis(application_flow_frequency_raw, application_number):
    application_flow_frequency = application_flow_frequency_raw.copy()

    def polynomial(x, a, b, c, d, e):
        return a * x + b * x**2 + c * x**3 + d * x**4 + e

    # Adding a fitted curve for the given points
    optimizedParameters, pcov = opt.curve_fit(polynomial, application_flow_frequency['frequency'].astype(float), application_flow_frequency['seasonal_volume_af'].astype(float))
    application_flow_frequency['fitted_curve'] = optimizedParameters[0]*application_flow_frequency['frequency'].astype(float)+optimizedParameters[1]*application_flow_frequency['frequency'].astype(float)**2+optimizedParameters[2]*application_flow_frequency['frequency'].astype(float)**3+optimizedParameters[3]*application_flow_frequency['frequency'].astype(float)**4+optimizedParameters[4]

    r2 = r2_score(application_flow_frequency['seasonal_volume_af'], application_flow_frequency['fitted_curve'])
    r2 = r2.round(4)
    fig = Figure(figsize=(10, 7), dpi=80)
    ax = fig.subplots()
    ax.plot(application_flow_frequency['frequency'], application_flow_frequency['fitted_curve'], label="fit", color = 'green')
    ax.scatter(application_flow_frequency['frequency'],
        application_flow_frequency['seasonal_volume_af'],
        linewidth=2)

    fig.gca().invert_xaxis()
    fig.suptitle("Flow frequency analysis at {} \n $y={}*x+{}*x^2+{}*x^3+{}*x^4+{}$ \n $R^2 = {}$".format(application_number,optimizedParameters[0].round(2),optimizedParameters[1].round(2),optimizedParameters[2].round(2),optimizedParameters[3].round(2),optimizedParameters[4].round(2), r2))
    fig.supxlabel('Frequency of Occurrence')
    fig.supylabel('Discharge, acre-ft')
    ax.grid(color='grey', linestyle='-', linewidth=0.25)

    buf = BytesIO()
    fig.savefig(buf, format="png",  dpi='figure')
    return buf

def validate_senior_diverter_csv(df, requires_cda):
    """Validate uploaded csv against business logic.

    Args:
        df (pd.DataFrame): A dataframe of the uploaded csv data.

    Returns:
        list: A list of found errors.
    """
    errors = []

    if not set(required_uploaded_csv_columns).issubset(df.columns):
        return [f"Not all required columns provided. Please include all of {required_uploaded_csv_columns}"]

    df[['max_storage_af', 'face_amount_af', 'max_rate_of_diversion_cfs', 'minimum_bypass_flow_cfs']] = df[[
        'max_storage_af', 'face_amount_af', 'max_rate_of_diversion_cfs', 'minimum_bypass_flow_cfs']].fillna(0)

    df = df.copy()

    all_analysis_labels_correct = df["analysis_label"].isin([
        "Upstream of POD",
        "Proposed POD",
        "Downstream Flow Path",
        "Upstream of Downstream Flow Path",
        "Upstream of POA",
        "Inside Project Extent"
    ]).all()

    if (not all_analysis_labels_correct):
        errors.append("Not all analysis labels are in the list (Upstream of POD, Proposed POD, Downstream Flow Path, Upstream of Downstream Flow Path, Inside Project Extent)")

    if (df[df["analysis_label"] == "Proposed POD"].shape[0] != 1):
        errors.append("Exactly one entry must be labelled as \"Proposed POD\"")

    if (not df['order_upstream_to_downstream'].is_unique):
        errors.append("order_upstream_to_downstream values are not unique")

    if (not all(is_integer(val) for val in df["order_upstream_to_downstream"])):
        errors.append("order_upstream_to_downstream values are not all integers")

    if (not pd.to_numeric(df['latitude'], errors='coerce').notnull().all()):
        errors.append("Not all latitudes are numeric")

    if (not pd.to_numeric(df['longitude'], errors='coerce').notnull().all()):
        errors.append("Not all longitudes are numeric")
    # Ignore the Proposed POD row for all other checks
    df = df.loc[df.analysis_label != 'Proposed POD']

    # Only the proposed POD can be missing application number
    if (df[df.analysis_label != 'Proposed POD'].application_number.isnull().values.any()):
        errors.append("Ensure no application numbers are null")

    errors = errors + validate_csv_month_fields(df, 'direct_div')
    errors = errors + validate_csv_month_fields(df, 'storage')

    if (not all(is_float_in_range(val, 0, 30000000) for val in df["max_storage_af"])):
        errors.append("All max_storage_af fields must be between 0 and 30000000")

    if (not all(is_float_in_range(val, 0, 30000000) for val in df["face_amount_af"])):
        errors.append(
            {"error": "All non-empty face_amount_af fields must be between 0 and 30000000"})

    if (not all(is_float_in_range(val, 0, 150000) for val in df["max_rate_of_diversion_cfs"].dropna())):
        errors.append("All non-empty max_rate_of_diversion_cfs fields must be between 0 and 150000")

    #Only check minimum_bypass_flow_cfs if requires_cda = True
    if(requires_cda and not all(is_float_in_range(val, 0, 150000) for val in df["minimum_bypass_flow_cfs"].dropna())):
       errors.append("All non-empty minimum_bypass_flow_cfs fields must be between 0 and 150000")

    #Validation for wr_water_right_id
    if(df['wr_water_right_id'].isnull().values.any()):
        errors.append("All wr_water_right_id fields must be populated")

    #Validation for use_codes
    if(df['use_codes'].isnull().values.any()):
        errors.append("All use_codes fields must be populated")

    direct_div_cols = ["direct_div_season_start_month", "direct_div_season_start_day",
                       "direct_div_season_end_month", "direct_div_season_end_day"]
    storage_cols = ["storage_season_start_month", "storage_season_start_day",
                    "storage_season_end_month", "storage_season_end_day"]

    for index, row in df.iterrows():
        has_full_direct_div_season = has_full_row(row, direct_div_cols)
        has_full_storage_season = has_full_row(row, storage_cols)

        if (not has_full_direct_div_season and not has_full_storage_season):
            errors.append(f'Either a full set of direct div or storage dates are required for each row.')
            break
        if (not has_full_storage_season and row['max_storage_af'] != 0):
            errors.append(f"If maximum storage is set, you must provide a full set of storage season dates")

        if (has_full_storage_season):
            errors = has_valid_date(row, 'storage', errors)
        if (has_full_direct_div_season):
            errors = has_valid_date(row, 'direct_div', errors)

        if(seasonal_demand_populated_check(row)):
            errors.append("If seasonal_demand_af is populated, justification must be supplied in overwrite_seasonal_demand_justification")

    return errors

def validate_csv_month_fields(df, prefix):
    """Validate month fields are in an acceptable range if not NaN

    Args:
        df (pd.DataFrame): the dataframe of csv data
        prefix (str): the column name prefix, e.g direct_div

    Returns:
        list: A list of found errors
    """
    errors = []
    if not all(is_integer(val) and (int(val) <= 12 and int(val) >= 1) for val in df[f"{prefix}_season_start_month"].dropna()):
        errors.append(
            {"error": f"{prefix}_season_start_month must be an integer between 1 and 12"})
    if not all(is_integer(val) and (int(val) <= 12 and int(val) >= 1) for val in df[f"{prefix}_season_end_month"].dropna()):
        errors.append(
            {"error": f"{prefix}_season_end_month must be an integer between 1 and 12"})
    if not all(is_integer(val) and (int(val) <= 31 and int(val) >= 1) for val in df[f"{prefix}_season_start_day"].dropna()):
        errors.append(
            {"error": f"{prefix}_season_start_day must be an integer between 1 and 31"})
    if not all(is_integer(val) and (int(val) <= 31 and int(val) >= 1) for val in df[f"{prefix}_season_end_day"].dropna()):
        errors.append(
            {"error": f"{prefix}_season_end_day must be an integer between 1 and 31"})
    return errors

def get_month_date_range_overlap(range_a, range_b):
    """Function to determine the overlap between two date ranges where the year does not matter.
    For ease of use with other libraries, dates are setup so that they will be within 2019 if they
    dont span two years, and from 2018 to 2019 otherwise

    Args:
        range_a (Range): a range
        range_b (Range): a range
    """

    def get_range_overlap(range_a, range_b):
        """Gets the overlap of two ranges within the same year

        Args:
            range_a (Range): a range
            range_b (Range): a range

        """
        latest_start = max(range_a.start,
                        range_b.start)
        earliest_end = min(range_a.end, range_b.end)
        delta = (earliest_end - latest_start).days + 1
        overlap = max(0, delta)
        return overlap

    range_a_spans_year = range_a.start.year == 2018
    range_b_spans_year = range_b.start.year == 2018
    if (range_a_spans_year and not range_b_spans_year):
        first_overlap = get_range_overlap(range_a, range_b)
        range_a_next_year = Range(
            start=range_a.start.replace(year=2019),
            end=range_a.start.replace(year=2020)
        )
        second_overlap = get_range_overlap(range_a_next_year, range_b)
        return first_overlap + second_overlap

    if (not range_a_spans_year and range_b_spans_year):
        first_overlap = get_range_overlap(range_a, range_b)
        range_b_next_year = Range(
            start=range_b.start.replace(year=2019),
            end=range_b.start.replace(year=2020)
        )
        second_overlap = get_range_overlap(range_a, range_b_next_year)
        return first_overlap + second_overlap

    if ((not range_a_spans_year and not range_b_spans_year) or (range_a_spans_year and range_b_spans_year)):
        return get_range_overlap(range_a, range_b)

def get_row_date_overlap(row, proposed_range, prefix):
    if (pd.isnull(row[f"{prefix}_season_start"]) or pd.isnull(row[f"{prefix}_season_end"]) or proposed_range == 0):
        return 0

    senior_diverter_diversion_range = Range(start=pd.to_datetime(
        row[f"{prefix}_season_start"], dayfirst=True), end=pd.to_datetime(row[f"{prefix}_season_end"], dayfirst=True))

    return get_month_date_range_overlap(senior_diverter_diversion_range, proposed_range)

def calculate_frost_demand_amount(row, proposed_range):
    """
        Frost demand occurs for 8 hours a day every other day at the senior diverter's max_rate_of_diversion.
        This happens between March 15th and April 30th.
    """
    if(row['max_storage_af'] == row['face_amount_af'] and row['overlapping_days_of_storage_and_policy_season'] > 0):
        # Max Storage and Face amount are equal -> assume this means that all diversions go to storage
        # This implies that the frost diversions would happen in the policy season if storage is in policy season
        frost_range = Range(start=pd.to_datetime(
            '15-03-2019', dayfirst=True), end=pd.to_datetime('31-03-2019', dayfirst=True))
    else:
        # Otherwise, use the full frost range
        frost_range = Range(start=pd.to_datetime(
            '15-03-2019', dayfirst=True), end=pd.to_datetime('30-04-2019', dayfirst=True))
    overlap = get_month_date_range_overlap(frost_range, proposed_range)
    # Every other day
    overlap = overlap // 2
    if (not 'Frost Protection' in row['use_codes']):
        return 0
    elif (row['overlapping_proposed_and_days_of_direct_diversion'] == 0):
        return 0
    elif (overlap == 0):
        return 0
    else:
        return overlap * 10 * 3600 * row['max_rate_of_diversion_cfs'] / 43560

def calculate_seasonal_demand_amount(row):
    """
        Calculates the wr_seasonal_demand (overall demand overlap with proposed season)

        https://foundryspatial.atlassian.net/wiki/spaces/CAL/pages/1777664023/Senior+Diverters+Seasonal+Demand
        Represents calculation of number 15 in above document
        Splits in and outside of policy season
    """
    #If seasonal_demand_af has been set for this row by the user, use that value
    if('seasonal_demand_af' in row and pd.notna(row['seasonal_demand_af'])):
        return row['seasonal_demand_af']
    # with no seasonal overlap set to zero
    if (row["overlapping_proposed_and_days_of_direct_diversion"] == 0 and row["overlapping_days_of_storage_and_proposed_season"] == 0):
        return 0
    total_demand_af = 0
    #Inside of policy season - our calculations only allow entirely inside or entirely outside of policy season
    if(row["overlapping_days_of_proposed_and_policy_season"] > 0):
        #Storage amount
        if(row["overlapping_days_of_storage_and_proposed_season"] > 0 and row["overlapping_days_of_storage_and_policy_season"] > 0):
            total_demand_af += row["max_storage_af"] * row["overlapping_days_of_storage_and_proposed_season"] / row["overlapping_days_of_storage_and_policy_season"]
        #Add frost (0 if "Frost Protection" not in use_codes)
        total_demand_af += row['frost_demand_af']
        #If "Frost Protection" and "Irrigation" aren't the only use_codes, add diversion
        if (not (len(row["use_codes"]) in [1,2] and all(use_code in ['Frost Protection', 'Irrigation'] for use_code in row["use_codes"]))):
            total_demand_af += row["overlapping_proposed_and_days_of_direct_diversion"] * row["diversion_per_day_af"]
    #Outside of policy season
    else:
        #Only if no storage happens in policy season we will include with overlap
        if(row["overlapping_days_of_storage_and_policy_season"] == 0):
            total_demand_af +=  (row["max_storage_af"] * row["overlapping_days_of_storage_and_proposed_season"] / row["days_of_storage"])
        # Add calculated frost demand if it occurs outside of the policy season
        total_demand_af += row['frost_demand_af']
        #If Irrigation and Frost Protection are only use codes, the diversion only happens outside of policy season
        if("Irrigation" in row['use_codes'] and len(row["use_codes"]) in [1,2] and all(use_code in ['Frost Protection', 'Irrigation'] for use_code in row["use_codes"])):
            #calculate diversion days outside of the policy season
            diversion_days_outside_of_policy = row['days_of_diversion'] - row['overlapping_days_of_direct_diversion_and_policy_season']
            if(diversion_days_outside_of_policy > 0):
                irrigation_diversion_per_day = row['diversion_per_day_af'] *  row['days_of_diversion']/diversion_days_outside_of_policy
                total_demand_af += irrigation_diversion_per_day*row['overlapping_proposed_and_days_of_direct_diversion']
        #otherwise take diversion overlap with diversion_per_day
        else:
            total_demand_af += row["overlapping_proposed_and_days_of_direct_diversion"] * row["diversion_per_day_af"]
    if(row['face_amount_af'] < total_demand_af):
        return row['face_amount_af']
    return total_demand_af


def get_start_date(row, date_cols):
    for col in date_cols:
        if (np.isnan(row[col])):
            return np.nan
    start_date = datetime.datetime(
        2019, int(row[date_cols[0]]), int(row[date_cols[2]]))
    end_date = datetime.datetime(
        2019, int(row[date_cols[1]]), int(row[date_cols[3]]))
    if (end_date <= start_date):
        start_date = start_date.replace(year=2018)
    return start_date.strftime('%d-%m-%Y')

def get_end_date(row, date_cols):
    for col in date_cols:
        if (np.isnan(row[col])):
            return np.nan
    start_date = datetime.datetime(
        2019, int(row[date_cols[0]]), int(row[date_cols[2]]))
    end_date = datetime.datetime(
        2019, int(row[date_cols[1]]), int(row[date_cols[3]]))
    if (start_date == end_date):
        end_date = end_date - datetime.timedelta(days=1)
        if (end_date == datetime.datetime(2018, month=12, day=31)):
            end_date = end_date.replace(year=2019)
    return end_date.strftime('%d-%m-%Y')

#Analyse use codes only if not proposed POD
def format_use_codes(row):
    if(row["analysis_label"] != "Proposed POD"):
        codes = row['use_codes'].split(',')
        codes = [code.strip() for code in codes]
        return codes
    else:
        return []

def add_frost_irrigation_comments(row):
    """
    Add comments for edge cases with Frost and Irrigation.
    Checks if there is no overlap with the diverter's season and the frost/Irrigation season.
    """
    additional_comments = ""
    if(isinstance(row['use_codes'], float) and pd.isna(row['use_codes'])):
        row['use_codes'] = []
    if(pd.isna(row['comments'])):
        row['comments'] = ""
    if ('Frost Protection' in row['use_codes'] and get_row_date_overlap(row, frost_season, 'direct_div') == 0):
        #Don't add same comments multiple times
        if("Warning: Frost Protection in use_codes but senior diverter's diversion season has no overlap with frost season." not in row['comments']):
            additional_comments += " Warning: Frost Protection in use_codes but senior diverter's diversion season has no overlap with frost season."
    if ('Irrigation' in row['use_codes'] and get_row_date_overlap(row, non_policy_season, 'direct_div') == 0 and get_row_date_overlap(row, non_policy_season, 'storage') == 0):
        if("Warning: Irrigation in use codes but senior diverter's diversion and storage season has no overlap with summer irrigation season." not in row['comments']):
            additional_comments += " Warning: Irrigation in use codes but senior diverter's diversion and storage season has no overlap with summer irrigation season."
    return (row['comments'] + additional_comments).strip()


def calc_wsr_intermediate_values(senior_diverters_df, form_data_df):
    """
        Creating the generated "intermediate" values for CWAT, expected for data output

        Reference for fields:
        https://foundryspatial.atlassian.net/wiki/spaces/CAL/pages/1777664023/Senior+Diverters+Seasonal+Demand

        Numbers below correspond to numbering in above file
    """
    calculated_intermediate_values = senior_diverters_df.copy()

    #Turn use codes into a workable array
    calculated_intermediate_values['use_codes'] = calculated_intermediate_values.apply(format_use_codes, axis=1)

    calculated_intermediate_values = calculated_intermediate_values.replace({
                                                                            pd.NA: np.nan})
    calculated_intermediate_values.fillna(np.nan, inplace=True)

    #1. Create direct_div_season_start
    calculated_intermediate_values["direct_div_season_start"] = calculated_intermediate_values.apply(
        lambda x: get_start_date(x, direct_div_date_cols), axis=1)
    #2. Create direct_div_season_end
    calculated_intermediate_values["direct_div_season_end"] = calculated_intermediate_values.apply(
        lambda x: get_end_date(x, direct_div_date_cols), axis=1)
    #3. Create storage_season_start
    calculated_intermediate_values["storage_season_start"] = calculated_intermediate_values.apply(
        lambda x: get_start_date(x, storage_date_cols), axis=1)
    #4. Create storage_season_end
    calculated_intermediate_values["storage_season_end"] = calculated_intermediate_values.apply(
        lambda x: get_end_date(x, storage_date_cols), axis=1)

    #5. Create days_of_diversion
    calculated_intermediate_values['days_of_diversion'] = ((pd.to_datetime(calculated_intermediate_values['direct_div_season_end'], dayfirst=True) - pd.to_datetime(
        calculated_intermediate_values['direct_div_season_start'], dayfirst=True)).dt.days + 1).fillna(value=0)

    #6. Create days_of_storage
    calculated_intermediate_values['days_of_storage'] = (pd.to_datetime(calculated_intermediate_values['storage_season_end'],
        dayfirst=True) - pd.to_datetime(calculated_intermediate_values['storage_season_start'], dayfirst=True)).dt.days + 1

    #7. Create diversion_amount_af
    calculated_intermediate_values['diversion_amount_af'] = calculated_intermediate_values.apply(
        lambda row: row['face_amount_af'] - row['max_storage_af'] if row['face_amount_af'] > row['max_storage_af'] else 0, axis=1)

    #8. Create diversion_per_day_af
    calculated_intermediate_values['diversion_per_day_af'] = calculated_intermediate_values.apply(
        lambda row: row['diversion_amount_af'] / row['days_of_diversion'] if row['days_of_diversion'] >0  else 0, axis=1)

    proposed_season_range = None
    if (pd.isnull(form_data_df.loc["seasonOfDiversionStart"]) or pd.isnull(form_data_df.loc["seasonOfDiversionEnd"])):
        proposed_season_range = 0
    else:
        proposed_season_range = Range(start=pd.to_datetime(
            form_data_df["seasonOfDiversionStart"], dayfirst=True).tz_localize(None), end=pd.to_datetime(form_data_df["seasonOfDiversionEnd"], dayfirst=True).tz_localize(None))

    #9. Create overlapping_proposed_and_days_of_direct_diversion
    calculated_intermediate_values['overlapping_proposed_and_days_of_direct_diversion'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, proposed_season_range, 'direct_div'), axis=1)

    #10. Create overlapping_days_of_storage_and_policy_season
    calculated_intermediate_values['overlapping_days_of_storage_and_policy_season'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, policy_season, 'storage'), axis=1)

    #11. Create overlapping_proposed_and_storage
    calculated_intermediate_values['overlapping_days_of_storage_and_proposed_season'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, proposed_season_range, 'storage'), axis=1)

    #12. Create overlapping_days_of_direct_diversion_and_policy_season
    calculated_intermediate_values['overlapping_days_of_direct_diversion_and_policy_season'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, policy_season, 'direct_div'), axis=1)

    #13. Create overlapping_days_of_proposed_and_policy_season
    overlapping_days_of_proposed_and_policy_season = get_month_date_range_overlap(policy_season, proposed_season_range)

    calculated_intermediate_values = calculated_intermediate_values.assign(
        overlapping_days_of_proposed_and_policy_season=overlapping_days_of_proposed_and_policy_season)

    #14. Create overlapping_days_of_proposed_and_frost_season
    overlapping_days_of_proposed_and_frost_season = get_month_date_range_overlap(frost_season, proposed_season_range)

    calculated_intermediate_values = calculated_intermediate_values.assign(
        overlapping_days_of_proposed_and_frost_season=overlapping_days_of_proposed_and_frost_season)

    #15. Create frost_demand_af
    calculated_intermediate_values['frost_demand_af'] = calculated_intermediate_values.apply(
        lambda x: calculate_frost_demand_amount(x, proposed_season_range), axis=1)

    #Add comment if edge cases exist in frost and Irrigation cases
    calculated_intermediate_values['comments'] = calculated_intermediate_values.apply(
        lambda x: add_frost_irrigation_comments(x), axis=1
    )

    #16. Create wr_seasonal_demand
    calculated_intermediate_values['wr_seasonal_demand'] = calculated_intermediate_values.apply(
        lambda x: calculate_seasonal_demand_amount(x), axis=1)

    #Re-format use codes back into comma-separated list
    calculated_intermediate_values['use_codes'] = calculated_intermediate_values['use_codes'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else '')

    return calculated_intermediate_values.drop(intermediate_table_dropped_columns, axis=1)



def add_wsr_flow_frequencies_to_zip(zf, raw_data):
    """Helper to add flow frequency data to a zip archive.

    Args:
        zf (_type_): The zip archive to add to.
        raw_data (_type_): Raw flow frequency data.
    """
    df = pd.DataFrame.from_dict(raw_data)
    df.sort_values(by=['application_number', 'rank'], inplace=True)
    application_numbers = df.application_number.unique()

    for application_number in application_numbers:
        application_flow_frequency = df[df.application_number == application_number].copy()

        # Get formatted date information to add to filename. Season applies to the proposed POD and is the same across rows.
        season_start = application_flow_frequency.iloc[1]['direct_div_season_start']
        season_end = application_flow_frequency.iloc[1]['direct_div_season_end']
        filename = f'flow_frequency_analysis_{application_number}_{season_start}-{season_end}'

        chart = get_wsr_frequency_analysis(application_flow_frequency, application_number)
        zf.writestr(f'{filename}.png', chart.getvalue())

        demand_title = 'Proposed POD' if application_number == 'Proposed POD' else f"water right, {application_number}"
        application_flow_frequency.rename(columns={'seasonal_volume_af': f'Discharge, acre-ft, at {demand_title}'}, inplace=True)
        # Removing unneeded columns, season only used in filename
        formatted_application_flow_frequency_data = application_flow_frequency.drop(['direct_div_season_start', 'direct_div_season_end'], axis=1)

        water_rights_csv = formatted_application_flow_frequency_data.to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
        zf.writestr(f'{filename}.csv', water_rights_csv)


def format_session_data(raw_data):
    """Helper function to reformat raw data recieved by sql query to get session data and various other bits of data meant to end up in the wsr_project_information.txt file in the output package

    Args:
        raw_data: raw data returned by the query dictionary of values to be used
    """
    # processing year array to make it readable
    full_water_year_array = raw_data.get('full_water_year_array', ["no full water years available for selected gage"])
    formatted_years = [str(year) for year in full_water_year_array]
    formatted_years_str = ', '.join(formatted_years)

    # processing coord list so it is readable
    # eval can be a security risk but because podcoord is a key value pair from a geojson no bad strings can make it here
    podcoord = raw_data.get('podcoord', "[0,0]")
    podcoord_list = eval(podcoord)
    longitude, latitude = podcoord_list


    # processing date and timestamp fields into the desired format
    raw_timestamp = raw_data.get('rawmodified', 'Unedited senior diverters CSV have not yet been downloaded')
    int_timestamp = raw_data.get('intmodified', 'Project edited senior diverters CSV has not yet been uploaded')
    created_timestamp = raw_data.get('created', 'No Created Date??')
    modified_timestamp = raw_data.get('modified', 'No Modified Date??')
    season_end_date = raw_data.get('season_end', 'Season end not supplied')
    season_start_date = raw_data.get('season_start', "No season start supplied")
    freeze_str = 'Data not frozen, up to date with current date'
    freeze_date = raw_data.get('freeze', freeze_str)
    vancouver_timezone = pytz.timezone('America/Vancouver')


    if isinstance(season_start_date, datetime.datetime):
        season_start_str = season_start_date.strftime('%B %d, %Y')
    else:
        season_start_str = season_start_date

    if isinstance(season_end_date, datetime.datetime):
        season_end_str = season_end_date.strftime('%B %d, %Y')
    else:
        season_end_str = season_end_date

    if isinstance(created_timestamp, datetime.datetime):
        created_datetime = created_timestamp.astimezone(vancouver_timezone)
        created_datetime_str = created_datetime.strftime('%B %d, %Y %H:%M:%S %Z')
    else:
        created_datetime_str = created_timestamp

    if isinstance(modified_timestamp, datetime.datetime):
        modified_datetime = modified_timestamp.astimezone(vancouver_timezone)
        modified_datetime_str = modified_datetime.strftime('%B %d, %Y %H:%M:%S %Z')
    else:
        modified_datetime_str = modified_timestamp

    if isinstance(raw_timestamp, datetime.datetime):
        raw_timestamp = raw_timestamp.astimezone(vancouver_timezone)
        raw_timestamp_str = raw_timestamp.strftime('%B %d, %Y %H:%M:%S %Z')
    else:
        raw_timestamp_str = raw_timestamp

    if isinstance(int_timestamp, datetime.datetime):
        int_timestamp = int_timestamp.astimezone(vancouver_timezone)
        int_timestamp_str = int_timestamp.strftime('%B %d, %Y %H:%M:%S %Z')
    else:
        int_timestamp_str = int_timestamp
    if freeze_date != freeze_str:
        freeze_date = datetime.datetime.strptime(freeze_date.replace('"',''), "%Y-%m-%d %H:%M:%S.%f%z")
        freeze_date = freeze_date.astimezone(vancouver_timezone)
        freeze_date_str = freeze_date.strftime('%B %d, %Y %H:%M:%S %Z')
    else:
        freeze_date_str = freeze_date

    # rounding values to their desired decimal places
    raw_data['podarea'] = round(raw_data.get('podarea', 0), 3)
    raw_data['area_sqmi'] = round(raw_data.get('area_sqmi', 0), 3)
    longitude = round(longitude,7)
    latitude = round(latitude,7)

    # formatting data
    project_section = f"""Project Data:
    Project name: {raw_data.get('title', 'No title')}
    Project description: {raw_data.get('descr', 'No description supplied')}
    Project created on: {created_datetime_str}
    Project last modified on: {modified_datetime_str}
    Project unedited senior diverters CSV last downloaded on: {raw_timestamp_str}
    Project edited senior diverters CSV last uploaded on: {int_timestamp_str}
    Proposed longitude: {longitude}
    Proposed latitude: {latitude}
    Proposed POD NHDPlusID: {raw_data.get('nhdid', "No proposed NHDPlusID")}
    Proposed POD NHD basin name: {raw_data.get('podname', "Unnamed Basin")}
    Proposed POD drainage area miles squared: {raw_data.get('podarea', 0)}
    Proposed season start date: {season_start_str}
    Proposed season end date: {season_end_str}
    Proposed rate of diversion: {raw_data.get('rodval', "No supplied Rate of diversion value")} {raw_data.get('rodunit', 'No supplied rate of diversion units')}
    Proposed volume of diversion: {raw_data.get('vodval', "No supplied volume of diversion value")} {"Acre-Feet" if raw_data.get('vodunit', 'No supplied volume of diversion units') == 'acreFeet' else raw_data.get('vodunit', 'No supplied volume of diversion units')}
    Requires CDA: {raw_data.get('cdabool', "Project created before CDA was an option")}
    """
    date_section = f"""Streamflow & Senior Diverters data current as of:
    {freeze_date_str}
    """
    station_section = f"""Gage Data:
    Gage data source: United States Geological Survey (USGS)
    Gage data last updated: Dec 12th, 2023
    Station name: {raw_data.get('station_name', 'Unnamed Station')}
    Site number: {raw_data.get('site_no', 'No site number')}
    Drainage area miles squared: {raw_data.get('area_sqmi', 'Drainage area unavailable')}
    First full water year: {raw_data.get('full_water_year_start', 'First year unavilable for selected gage')}
    Last full water year: {raw_data.get('full_water_year_end', 'Last year unavilable for selected gage')}
    Number of full water years: {raw_data.get('numb_of_full_years', "Number of years unavailble for selected gage")}
    List of full water years: {formatted_years_str}
    """
    returnval = f"{date_section}\n{project_section}\n{station_section}"
    return returnval

def build_ordered_nhds_upstream(upstream_df, analysis_idx, tree_depth, mainstem_depth):
    """
    Creates a list of upstream ordered nhd ids from the most downstream nhd id
    """
    if(upstream_df.iloc[analysis_idx,  upstream_df.columns.get_loc('analysis_label_map')] is not None):
        return
    upstream_df.iloc[analysis_idx, upstream_df.columns.get_loc('analysis_label_map')] = 4 if upstream_df.iloc[analysis_idx]['is_mainstem'] else 1
    upstream_df.iloc[analysis_idx, upstream_df.columns.get_loc('tree_depth')] = tree_depth
    upstream_df.iloc[analysis_idx, upstream_df.columns.get_loc('mainstem_depth')] = mainstem_depth
    next_analysis_cases = upstream_df[upstream_df['tonode'] == upstream_df.iloc[analysis_idx, upstream_df.columns.get_loc('fromnode')]]
    next_analysis_cases.apply(
        lambda row, upstream_df, tree_depth, mainstem_depth:
        build_ordered_nhds_upstream(upstream_df,
                                    row.name,
                                    tree_depth=tree_depth+1,
                                    mainstem_depth = mainstem_depth + 1 if row['is_mainstem'] else mainstem_depth
                                    ),
        axis = 1,
        upstream_df = upstream_df,
        tree_depth = tree_depth,
        mainstem_depth = mainstem_depth)

def analysis_label_map_to_str(row, min_mainstem_order):
    """
    Performs logical mapping of analysis labels to strings

    Args:
        row - current row to be run on
        min_mainstem_order - minimum mainstem, used for comparison
    """
    if(row['analysis_label_map'] == 2 and row['order_upstream_to_downstream'] > min_mainstem_order):
        return 'Downstream Flow Path'
    elif(row['analysis_label_map'] == 1):
        return 'Upstream of Downstream Flow Path'
    elif(row['analysis_label_map'] == 2):
        return 'Upstream of POD'
    elif(row['analysis_label_map'] == 3):
        return 'Proposed POD'
    elif(row['analysis_label_map'] == 4):
        return 'Downstream Flow Path'
    elif(row['analysis_label_map'] == 5):
        return 'Inside Project Extent'

def sort_using_nhd_list_and_build_output(upstream_df, unsorted_df, nhd, pod_lat, pod_long):
    """
    Perform sorting on unsorted diverters and add some formatting required for output csvs.
    Split apart for unit testing and readability purposes

    Args:
        upstream_df - df of nhds upstream of lowest mainstem diverter nhd id
        unsorted_df - df of unsorted senior diverters
        nhd - nhd id of pod
        pod_lat - latitude of pod
        pod_long - longitude of pod
    """
    upstream_df.dropna(subset=['tree_depth'], inplace=True)
    upstream_df.sort_values(['mainstem_depth', 'tree_depth'], inplace=True)
    upstream_df['nhd_order'] = list(range(len(upstream_df.index)))
    above_pod = unsorted_df[unsorted_df['in_pod_basin']]
    downstream_pods = unsorted_df[~unsorted_df['in_pod_basin']]
    del unsorted_df
    above_pod = above_pod.merge(upstream_df, on='nhdplusid', how='left')
    above_pod.sort_values(['nhd_order', 'st_calc_distance'], inplace=True, ascending=[False, True])
    isolated_stream_reach = above_pod[pd.isna(above_pod['nhd_order'])]
    above_pod = above_pod[pd.notna(above_pod['nhd_order'])]
    above_pod['analysis_label_map'] = [2] * len(above_pod.index)
    isolated_stream_reach['analysis_label_map'] =  [5] * len(isolated_stream_reach.index)
    pod_data = {}
    pod_data['latitude'] = pod_lat
    pod_data['longitude'] = pod_long
    pod_data['nhdplusid'] = nhd
    pod_data['riparian'] = False
    pod_data['analysis_label_map'] = 3
    for col in above_pod.columns:
        if col not in pod_data:
            pod_data[col] = None
    pod_df = pd.DataFrame.from_dict([pod_data])
    pod_df = pod_df[above_pod.columns]
    downstream_pods = downstream_pods.merge(upstream_df, on='nhdplusid', how='left')
    downstream_pods.sort_values(['nhd_order', 'st_calc_distance'], inplace=True, ascending=[False, True])
    downstream_pods['analysis_label_map'] = downstream_pods['analysis_label_map'].apply(lambda x: x if pd.notna(x) else 5)
    overall_output = pd.concat([above_pod, isolated_stream_reach, pod_df, downstream_pods])
    overall_output['order_upstream_to_downstream'] = list(range(1, len(overall_output.index)+1))
    min_mainstem_order = min(overall_output[overall_output['analysis_label_map'] == 4]['order_upstream_to_downstream']) if(not overall_output[overall_output['analysis_label_map'] == 4].empty) else max(overall_output['order_upstream_to_downstream'])
    overall_output['analysis_label'] = overall_output.apply(lambda row, min_mainstem_order: analysis_label_map_to_str(row, min_mainstem_order),
                                                                                  min_mainstem_order = min_mainstem_order, axis=1)
    overall_output['analysis_label_map'] = overall_output.apply(lambda row, min_mainstem_order:
                                                                row['analysis_label_map'] if(not (row['analysis_label_map'] == 2 and row['order_upstream_to_downstream'] > min_mainstem_order )) else 3,
                                                                min_mainstem_order=min_mainstem_order,
                                                                axis=1)
    overall_output['comments'] = [''] * len(overall_output.index)
    overall_output['seasonal_demand_af'] = [None] * len(overall_output.index)
    overall_output['overwrite_seasonal_demand_af_justification'] = [None] * len(overall_output.index)
    overall_output['minimum_bypass_flow_cfs'] = [None] * len(overall_output.index)
    overall_output = overall_output[required_uploaded_csv_columns + ['riparian']]
    overall_output = overall_output.fillna(np.nan).replace([np.nan], [None])
    converted_list = [RealDictRow(row) for row in overall_output.to_dict(orient='records')]
    return converted_list

def sort_and_format_unsorted_csv_data(unsorted_csv_data, nhd, pod_lat, pod_long):
    """
    Sorts the unformatted csv data from get_unsorted_senior_diverters_by_session_id.
    Reformats into type expected by senior diverter csv functions.
    Args:
        unsorted_csv_data - the unsorted csvs
        nhd - pod NHD id
        pod_lat - pod latitude
        pod_long - pod longitude
    """
    unsorted_df = pd.DataFrame.from_dict(unsorted_csv_data)
    if unsorted_df.empty:
        return []
    mainstem_drainage = unsorted_df[unsorted_df['mainstem']]['drainage_area_sqmi']
    downstream_idx = mainstem_drainage.idxmax() if(len(mainstem_drainage) > 0) else None
    if(pd.isna(downstream_idx)):
        most_downstream_nhd = nhd
    else:
        #.item() to convert from numpy int64 to python int
        most_downstream_nhd = unsorted_df.iloc[downstream_idx]['nhdplusid'].item()
    raw_upstream_nhds = app.db.get_depth_1_nhds_upstream_of_nhd_id(most_downstream_nhd, nhd)
    upstream_df = pd.DataFrame.from_dict(raw_upstream_nhds)
    upstream_df['tree_depth'] = None
    upstream_df['mainstem_depth'] = None
    upstream_df['analysis_label_map'] = None
    #Base case for recursion - lowest mainstem idx
    lowest_mainstem_idx = upstream_df[upstream_df['is_mainstem']]['hydroseq'].idxmin()
    #Larger recursion limit needed for large watersheds ( I feel dirty doing this )
    sys.setrecursionlimit(10000)
    build_ordered_nhds_upstream(upstream_df, lowest_mainstem_idx, tree_depth=0, mainstem_depth=0)
    sys.setrecursionlimit(1000)
    return sort_using_nhd_list_and_build_output(upstream_df, unsorted_df, nhd, pod_lat, pod_long)

def calculate_wsr_output_values(wsr_summary_dicts, gage_data, pod_rain_and_area, average_gage_flow):
    """
    Calculate values:
        - ratio1
        - seasonal_unimpaired_flow_volume_af
        - remaining_unimpaired_discharge_before_new_water_right_af
        - remaining_unimpaired_discharge_after_new_water_right_af
    Using wsr summary, gage data, and pod watershed data from the database
    """
    output_dicts = []
    for row in wsr_summary_dicts:
        output_dict = row
        output_dict['gage_area_sqmi'] = gage_data['drainage_area_sqmi']
        output_dict['gage_map_1991_2020_in'] = gage_data['map_1991_2020_in']
        if(output_dict['analysis_label'] == 'Proposed POD'):
            # Handle the pod watershed stats differently
            output_dict['area_sqmi'] = pod_rain_and_area['drainage_area_sqmi']
            output_dict['annual_precip_in'] = pod_rain_and_area['map_1991_2020_in']
            output_dict['map_1991_2020_in'] = pod_rain_and_area['map_1991_2020_in']
        else:
            output_dict['map_1991_2020_in'] = row['annual_precip_in']
        # Moved these calculations into python for simplicity (good opportunity for refactoring)
        output_dict['gage_seasonal_flow_af'] = average_gage_flow
        output_dict['ratio1'] = (output_dict['area_sqmi'] / gage_data['drainage_area_sqmi']) * (output_dict['annual_precip_in'] /  gage_data['map_1991_2020_in'])
        output_dict['seasonal_unimpaired_flow_volume_af'] = output_dict['ratio1'] * output_dict['gage_seasonal_flow_af']
        output_dict['remaining_unimpaired_discharge_before_new_water_right_af'] = output_dict['seasonal_unimpaired_flow_volume_af'] - output_dict['seasonal_upstream_demand_af']
        output_dict['percent_remaining_unappropriated_water_before_new_water_right'] = output_dict['remaining_unimpaired_discharge_before_new_water_right_af'] / output_dict['seasonal_unimpaired_flow_volume_af'] * 100
        output_dict['remaining_unimpaired_discharge_after_new_water_right_af'] = output_dict['remaining_unimpaired_discharge_before_new_water_right_af'] - output_dict['additional_impairment_caused_by_new_water_right_af']
        output_dict['percent_remaining_unappropriated_water_after_new_water_right'] = output_dict['remaining_unimpaired_discharge_after_new_water_right_af'] / output_dict['seasonal_unimpaired_flow_volume_af'] * 100
        output_dict['percent_change_caused_by_new_water_right'] = output_dict['percent_remaining_unappropriated_water_after_new_water_right'] - output_dict['percent_remaining_unappropriated_water_before_new_water_right']
        output_dict['ratio_of_project_total_diversion_to_impaired_streamflow'] = output_dict['additional_impairment_caused_by_new_water_right_af'] / output_dict['remaining_unimpaired_discharge_before_new_water_right_af']
        output_dicts.append(output_dict)
    return output_dicts

def generate_wsr_flow_frequency_points_of_analysis(formatted_data, yearly_mean_gage_timeseries):
    """
        Get the wsr flow frequency points of analysis. Filter to Proposed POD, senior POD with water available percentage the lowest
        and any others where the percentage is less than 50%.
    """
    vals_to_format = []
    percentages = [x['percent_remaining_unappropriated_water_after_new_water_right'] for x in formatted_data if x['analysis_label'] != "Proposed POD"]
    min_percentage = 0
    if(len(percentages) > 0):
        min_percentage = min(percentages)
    for row in formatted_data:
        if(row['analysis_label'] == 'Proposed POD'):
            vals_to_format.append(row)
        if(row['percent_remaining_unappropriated_water_after_new_water_right'] < 50 or
           row['percent_remaining_unappropriated_water_after_new_water_right'] == min_percentage):
            vals_to_format.append(row)

    yearly_mean_gage_timeseries = yearly_mean_gage_timeseries['daily_flow']

    # We apply a linear calculation to these things, so the sorted order will remain the same
    sorted_gage_yearly_means_desc = {k: v for k, v in sorted(yearly_mean_gage_timeseries.items(), key=lambda item: item[1], reverse = True)}
    num_gage_years = len(sorted_gage_yearly_means_desc)
    output_vals = []
    for row in vals_to_format:
        i = 1
        for gage_year in sorted_gage_yearly_means_desc.keys():
            output_dict = {}
            season_data = row['diversion_season'].split(' - ')
            output_dict['direct_div_season_start'] = season_data[0]
            output_dict['direct_div_season_end'] = season_data[1]
            output_dict['application_number'] = row['application_number']
            output_dict['seasonal_volume_af'] = sorted_gage_yearly_means_desc[gage_year] * row['ratio1']
            output_dict['rank'] = i
            # Weibull Formula (policy section B2.2 3)
            output_dict['frequency'] = 1 - (i / (num_gage_years + 1))
            i +=  1
            output_vals.append(output_dict)
    return output_vals

def filter_gage_dates_to_season(row, season_start_month, season_start_day, season_end_month, season_end_day, spans_year):
    date_split = row['date'].split("-")
    month = int(date_split[1])
    day = int(date_split[0])
    is_in_season = False
    if(spans_year):
        is_in_season = (month > season_start_month or (month == season_start_month and day >= season_start_day)) or (month < season_end_month or month == season_end_month and day <= season_end_day)
    else:
        is_in_season = (month > season_start_month or (month == season_start_month and day >= season_start_day)) and (month < season_end_month or month == season_end_month and day <= season_end_day)
    return is_in_season

def generate_gage_timeseries_seasonal_means(raw_gage_timeseries, diversion_season):
    """
    Generate the average of gage flow for each year.
    """
    diversion_season_split = diversion_season.split(" - ")
    start_split = diversion_season_split[0].split(' ')
    end_split = diversion_season_split[1].split(' ')
    season_start_month = MONTH_TO_NUMBER[start_split[0]]
    season_start_day = int(start_split[1])
    season_end_month = MONTH_TO_NUMBER[end_split[0]]
    season_end_day = int(end_split[1])
    spans_year = False
    if(season_end_month < season_start_month or
       season_end_month == season_start_month and season_end_day < season_start_day):
        spans_year = True
    gage_df = pd.DataFrame.from_dict(raw_gage_timeseries)
    gage_df['water_year'] = gage_df['date'].apply(lambda date: generate_water_year(date))
    boolean_series = gage_df.apply(lambda row, season_start_month, season_start_day, season_end_month, season_end_day, spans_year:
                            filter_gage_dates_to_season(row, season_start_month, season_start_day, season_end_month, season_end_day, spans_year),
                            season_start_month = season_start_month,
                            season_start_day = season_start_day,
                            season_end_month = season_end_month,
                            season_end_day = season_end_day,
                            spans_year = spans_year,
                            axis = 1
                        )

    gage_df = gage_df[boolean_series]
    # Output data is in Acre-feet per day. Convert the units here
    gage_df['daily_flow'] = gage_df['daily_flow'] / AFD_TO_CFS
    # Return the average for each water year, and the overall average
    return (gage_df.groupby('water_year').sum(numeric_only=True).to_dict(), gage_df['daily_flow'].sum(numeric_only=True)/(gage_df['water_year'].nunique()))


def generate_water_year(date):
    """
        Given a date of the form dd-mm-yyyy, find the water year from october - september
    """
    date_split = date.split('-')
    month = int(date_split[1])
    year = int(date_split[2])
    if(month >= 10):
        return year + 1
    else:
        return year
