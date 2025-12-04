from datetime import datetime, date as dt_date, timedelta
import math
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from utils.helpers import is_integer, is_float_in_range
from psycopg2.extras import RealDictRow
from utils.wsr_csv_utils import (
    Range,
    format_use_codes,
    generate_water_year,
    get_end_date,
    get_month_date_range_overlap,
    get_row_date_overlap,
    get_start_date,
    has_full_row,
    has_valid_date,
    required_uploaded_csv_columns,
    seasonal_demand_populated_check,
    validate_csv_month_fields,
    direct_div_date_cols,
    storage_date_cols,
    policy_season
)
#lookback/forward value for the sliding window functionality for peaks over threshold
SLIDING_WINDOW_LOOKBACK = 2
#CDA Schema - Defines the input shape for editing the cda_sessions object
# TODO: Update as necessary with new schema entries as CDA is built
cda_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "http://example.com/example.json",
    "type": "object",
    "default": {},
    "title": "CDA Schema",
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
            "type": "object",
            "properties": {
                "lat": { "$ref": "#/definitions/LatType" },
                "long": { "$ref": "#/definitions/LongType" }
            }
        },
        "StreamClass": {
            "type": "integer",
            "enum": [1, 2, 3]
        },
        "Anadromy": {
            "type": "string",
            "enum": ["Above", "Below or at"]
        },
        "PointOfInterest": {
            "type": "object",
            "properties": {
                "lat": { "$ref": "#/definitions/LatType" },
                "long": { "$ref": "#/definitions/LongType" },
                "class": { "$ref": "#/definitions/StreamClass"},
                "anadromy": { "$ref": "#/definitions/Anadromy"},
                "name": { "type" : "string"},
                "description" : { "type" : "string"},
                "upstreamWaterRight" : {"type" : "string"},
                "streamName": {"type": "string"},
                "nhdId": {"type": "integer"},
                "upstreamArea": {"type" : "number"},
                "id" : {"type" : "integer"},
            },
            "additionalProperties" : False,
            "required": ["id"]
        },
        "NaturalFlowVariability" : {
            "type" : "object",
            "properties" : {
                "instantaneousPeakUnimpaired" : {"type" : "number"},
                "instantaneousPeakDiverters" : {"type" : "number"},
                "instantaneousPeakPod" : {"type": "number"},
                "ratioUnimpairedDiverters" : {"type": "number"},
                "ratioUnimpairedPod" : {"type" : "number"},
                "ratioDifference" : {"type" : "number"},
                "underReductionLimit" : {"type" : "boolean"}
            }
        },
        "MonthlyImpairment": {
            "type": "object",
            "properties" : {
                "percentage" : {"type" : "integer"},
                "waterAvailable": {"type": "boolean"}
            }
        },
        "SpawningRearingPassage": {
            "type":  "object",
            "properties" : {
                "unimpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "diverterImpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "podImpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "percentageImpaired" : {"type": "array",
                                            "items": {
                                                "$ref": "#/definitions/MonthlyImpairment"
                                                }
                                            },
                "percentagePod" : {"type": "array",
                                            "items": {
                                                "$ref": "#/definitions/MonthlyImpairment"
                                                }
                                            }
            }
        },
        "FebruaryMedian": {
            "type":  "object",
            "properties" : {
                "unimpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "diverterImpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "podImpairedExceedances" : {"type": "array",
                                            "items": {
                                                "type": "integer"
                                                }
                                            },
                "percentageImpaired" : {"type": "array",
                                            "items": {
                                                "$ref": "#/definitions/MonthlyImpairment"
                                                }
                                            },
                "percentagePod" : {"type": "array",
                                            "items": {
                                                "$ref": "#/definitions/MonthlyImpairment"
                                                }
                                            }
            }
        },
        "ThresholdsRow": {
            "type": "object",
            "properties": {
                #put -2 in here to indicate gage, -1 for POD
                "poiId": { "type": "integer"},
                "drainageArea": { "type": "number", "minimum": 0},
                "averagePrecipitation": { "type": "number", "minimum": 0},
                "meanAnnualUnimpairedVolumeAf" : {"type": "number", "minimum": 0},
                "meanAnnualUnimpairedVolumeCfs": {"type": "number", "minimum": 0},
                "ratio": {"type": "number", "minimum": 0},
                "minimumBypassFlow" : {"type": "number"},
                "maximumCumulativeDiversion" : {"type": "number"},
                "minimumBypassFlowRegional": {"type": "number"},
                "maximumCumulativeDiversionRegional": {"type": "number"},
                "februaryMedian" : {"type" : "number"}
            },
            "additionalProperties" : False,
            "required" : ["poiId"]
        },
        "dailyFlowPOI": {
            "type" : "object",
            "properties" : {
                "poiId": { "type": "integer"},
                "yearsOfRecord" : { "type": "integer"},
                "spawningPassage": {"$ref" : "#/definitions/SpawningRearingPassage"},
                "naturalFlowVariability": {"$ref" : "#/definitions/NaturalFlowVariability"},
                "februaryMedian" : {"$ref": "#/definitions/FebruaryMedian"}
            }
        }
    },
    "properties": {
        "meta": { "$ref": "#/definitions/MetaType" },
        "status": {"type": "string"},
        "finishedWSR" : {"type": "boolean"},
        "gageHasSeniorDiverters" : {"type": "boolean"},
        "hasGageEditedWaterRights": {"type": "boolean"},
        "pointOfDiversion": { "$ref": "#/definitions/Point" },
        "podStreamClass": {"$ref": "#/definitions/StreamClass"},
        "seasonOfDiversionStart": {"type": ["string", "null"]},
        "seasonOfDiversionEnd": {"type": ["string", "null"]},
        "volumeOfDiversion": {"$ref": "#/definitions/UnitValue"},
        "rateOfDiversion": {"$ref": "#/definitions/UnitValue"},
        "regionalCriteria": {"type": "boolean"},
        "pointsOfInterest": {"type": "array",
                             "items": {
                                 "$ref": "#/definitions/PointOfInterest"
                                }
                             },
        "thresholdTableData": {"type": "array",
                               "items": {
                                   "$ref": "#/definitions/ThresholdsRow"
                               }
        },
        "dailyFlowData" : {"type" : "array",
                           "items" : {
                               "$ref": "#/definitions/dailyFlowPOI"
                           }
        }
    },
    "additionalProperties": True,
}
AFD_TO_CFS = 1.5125/3
GALLONS_TO_AF = 325851

def overwrite_with_wsr_diverters(gage_senior_diverters, wsr_senior_diverters):
    """
    If there are any wsr senior diverters which are in the gage_senior_diverters
    (shared watersheds), overwrite these with the values from the wsr.
    TODO: if the tool expands to more watersheds, look into using more efficient methods.
    Args:
        gage_senior_diverters - gage senior diverters
        wsr_senior_diverters - wsr senior diverters
    Returns:
        gage_senior_diverters which are edited
    """
    gage_df = pd.DataFrame.from_dict(gage_senior_diverters)
    wsr_df = pd.DataFrame.from_dict(wsr_senior_diverters)

    if(gage_df.empty):
        return

    gage_wr_ids = gage_df['wr_water_right_id'].tolist()
    for index, row in wsr_df.iterrows():
        if(row['wr_water_right_id'] in gage_wr_ids):
            gage_df_row_comments = gage_df[gage_df['wr_water_right_id'] == row['wr_water_right_id']]['comments'].tolist()[0]
            for col in gage_df.columns:
                if(col in row):
                    gage_df.loc[gage_df['wr_water_right_id'] == row['wr_water_right_id'], col] = row[col]
            gage_df.loc[gage_df['wr_water_right_id'] == row['wr_water_right_id'], 'comments'] = "Data Imported from WSR Senior Diverters. " + gage_df_row_comments
            #Reset 'analysis_label' to "Upstream of Gage"
            gage_df.loc[gage_df['wr_water_right_id'] == row['wr_water_right_id'], 'analysis_label'] = "Upstream of Gage"
    gage_df['use_codes'] = gage_df['use_codes'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    converted_list = [RealDictRow(row) for row in gage_df.to_dict(orient='records')]
    return converted_list

def validate_priority_date(date):
    """
        Ensure the "priority_date" field in a gage senior diverter csv has the right shape and is in a valid date range.
        Args:
            date - user-uploaded priority date
    """
    try:
        priority_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError as e:
        # Incorrectly formatted user upload
        return False
    except TypeError as e:
        # likewise, incorrectly formatted upload
        return False
    return priority_date > datetime.strptime("1800-01-01", "%Y-%m-%d") and priority_date < datetime.strptime("2100-01-01", "%Y-%m-%d")

def validate_gage_senior_diverter_csv(df):
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

    # For CDA, only upstream diversions are returned
    all_analysis_labels_correct = df["analysis_label"].isin([
        "Upstream of Gage",
    ]).all()

    if (not all_analysis_labels_correct):
        errors.append("Not all analysis labels are \"Upstream of Gage\"")

    if (not df['order_upstream_to_downstream'].is_unique):
        errors.append("order_upstream_to_downstream values are not unique")

    if (not all(is_integer(val) for val in df["order_upstream_to_downstream"])):
        errors.append("order_upstream_to_downstream values are not all integers")

    if (not pd.to_numeric(df['latitude'], errors='coerce').notnull().all()):
        errors.append("Not all latitudes are numeric")

    if (not pd.to_numeric(df['longitude'], errors='coerce').notnull().all()):
        errors.append("Not all longitudes are numeric")

    # Only the proposed POD can be missing application number
    if (df.application_number.isnull().values.any()):
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

    if(not all(is_float_in_range(val, 0, 150000) for val in df["minimum_bypass_flow_cfs"].dropna())):
       errors.append("All non-empty minimum_bypass_flow_cfs fields must be between 0 and 150000")

    #Validation for priority_date field, only evaluated if requires_cda = True
    if(not all(validate_priority_date(val) for val in df["priority_date"])):
        errors.append("All priority_date fields must be populated with a date of the form YYYY-MM-DD between 1800 and 2100")

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

def calculate_cda_seasonal_demand_amount(row):
    """
        Calculates the wr_seasonal_demand (overall demand)
        For cda, just take the total demand amount as required by the document
    """
    #If seasonal_demand_af has been set for this row by the user, use that value
    if('seasonal_demand_af' in row and pd.notna(row['seasonal_demand_af'])):
        return row['seasonal_demand_af']
    total_demand_af = 0
    #Storage amount
    total_demand_af += row["max_storage_af"]
    #Add frost (0 if "Frost Protection" not in use_codes)
    total_demand_af += row['frost_demand_af']
    #Add diversion
    total_demand_af += row["diversion_amount_af"]
    if(row['face_amount_af'] < total_demand_af):
        return row['face_amount_af']
    return total_demand_af


def calculate_cda_frost_demand_amount(row):
    """
        Frost demand occurs for 8 hours a day every other day at the senior diverter's max_rate_of_diversion.
        This happens between March 15th and April 30th.
    """
    if(row['max_storage_af'] == row['face_amount_af'] and row['overlapping_days_of_storage_and_policy_season'] > 0):
        # Max Storage and Face amount are equal -> assume this means that all diversions go to storage
        # This implies that the frost diversions would happen in the policy season if storage is in policy season
        frost_range = Range(start=pd.to_datetime(
            '15-03-2019', dayfirst=True), end=pd.to_datetime('31-03-2019', dayfirst=True))
        storage_range = Range(start=pd.to_datetime(
            row[f"storage_season_start"], dayfirst=True), end=pd.to_datetime(row[f"storage_season_end"], dayfirst=True))
        overlap = get_month_date_range_overlap(frost_range, storage_range)
    else:
        frost_range = Range(start=pd.to_datetime(
            '15-03-2019', dayfirst=True), end=pd.to_datetime('30-04-2019', dayfirst=True))
        diversion_range = Range(start=pd.to_datetime(
            row[f"direct_div_season_start"], dayfirst=True), end=pd.to_datetime(row[f"direct_div_season_end"], dayfirst=True))
        overlap = get_month_date_range_overlap(frost_range, diversion_range)
    # Every other day
    overlap = overlap // 2
    if (not 'Frost Protection' in row['use_codes']):
        return 0
    else:
        return overlap * 10 * 3600 * row['max_rate_of_diversion_cfs'] / 43560


def calculate_cda_intermediate_values(senior_diverters_df):
    """
        Creating the generated "intermediate" values for CWAT CDA gage diverters calculation, required to unimpair gage

        Reference for fields:
        https://foundryspatial.atlassian.net/wiki/spaces/CAL/pages/1777664023/Senior+Diverters+Seasonal+Demand

        Numbers below correspond to numbering in above file. Note that for this case (CDA), not all of the above are calcualted.
        Most importantly, the calcualtions for "proposed season overlaps" are irrelevant as this is a yearly study/ unimpairment.
    """
    #handle if we have an empty dataframe (no uploaded diverters)
    #Ensure it has the same shape for downstream functionality
    if senior_diverters_df.empty:
        calculated_intermediate_values = senior_diverters_df.copy()
        calculated_intermediate_values["direct_div_season_start"] = []
        calculated_intermediate_values["direct_div_season_end"] = []
        calculated_intermediate_values["storage_season_start"] = []
        calculated_intermediate_values["storage_season_end"] = []
        calculated_intermediate_values['days_of_diversion'] = []
        calculated_intermediate_values['days_of_storage'] = []
        calculated_intermediate_values['diversion_amount_af'] = []
        calculated_intermediate_values['diversion_per_day_af'] = []
        calculated_intermediate_values['overlapping_days_of_storage_and_policy_season'] = []
        calculated_intermediate_values['overlapping_days_of_direct_diversion_and_policy_season'] = []
        calculated_intermediate_values['frost_demand_af'] = []
        calculated_intermediate_values['wr_seasonal_demand'] = []
        return calculated_intermediate_values
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

    #10. Create overlapping_days_of_storage_and_policy_season
    calculated_intermediate_values['overlapping_days_of_storage_and_policy_season'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, policy_season, 'storage'), axis=1)

    #12. Create overlapping_days_of_direct_diversion_and_policy_season
    calculated_intermediate_values['overlapping_days_of_direct_diversion_and_policy_season'] = calculated_intermediate_values.apply(
        lambda x: get_row_date_overlap(x, policy_season, 'direct_div'), axis=1)

    #15. Create frost_demand_af
    calculated_intermediate_values['frost_demand_af'] = calculated_intermediate_values.apply(
        lambda x: calculate_cda_frost_demand_amount(x), axis=1)

    #16. Create wr_seasonal_demand
    calculated_intermediate_values['wr_seasonal_demand'] = calculated_intermediate_values.apply(
        lambda x: calculate_cda_seasonal_demand_amount(x), axis=1)

    #Re-format use codes back into comma-separated list
    calculated_intermediate_values['use_codes'] = calculated_intermediate_values['use_codes'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else '')

    return calculated_intermediate_values

def calculate_yearly_mean(unimpaired_gage_data):
    """
        Calculates the yearly mean of the supplied data
        Args:
            unimpaired_gage_data -> gage data that has been unimpaired from senior diverters
        Returns:
            yearly mean value of the supplied data
    """
    df = pd.DataFrame.from_dict(unimpaired_gage_data)
    df = df.fillna(method='ffill')

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    total_daily_flow = df['daily_flow'].sum()
    # Calculate the mean of all of the daily records
    return total_daily_flow / len(df['date'])

def calculate_feb_median(data):
    """
        Calculates the median of the values in february
        Args:
            data -> a time series json to calculate the february median for expected to be estimated impaired poi data
        Returns:
            February median value
    """
    df = pd.DataFrame.from_dict(data)
    df = df.fillna(method='ffill')

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    february_data = df[(df['date'].dt.month == 2)]
    median_february_flow = february_data['daily_flow'].median()
    return median_february_flow

def calculate_cda_ratio(poi_data, gage_data):
    """
        Calculate cda poi proration ratio, which is simply calculated using the precipitation and drainage area values.
        Args:
            poi_data - precipitation and drainage area from the poi
            gage_data - precipitation and drainage from the gage
        Returns:
            dict of ratio, precipitation and drainage area for the poi (returned to front-end)
    """
    ratio = (poi_data['drainage_area_sqmi']/gage_data['drainage_area_sqmi'])*(poi_data['map_1991_2020_in']/gage_data['map_1991_2020_in'])
    #Returning as dict so we have the needed data for the front end
    return {"ratio" : ratio,
            "averagePrecipitation": poi_data['map_1991_2020_in'],
            "drainageArea": poi_data['drainage_area_sqmi'] }

def calculate_mbf(wsda, Qm):
    """
        Calculates the minimum bypass flow

        Args:
            wsda - Stands for Water Shed Drainage Area, is the area used to determine which formula to use (in square miles)
            Qm - the mean annual unimpaired flow (in cubic feet per second)
        Returns:
            mbf - the minimum bypass flow given these two values
    """

    if(wsda <= 1):
        return 9.0*Qm
    elif(wsda < 321):
        return 8.8*Qm*((wsda)**(-0.47))
    else:
        return 0.6*Qm

def peaks_and_threshold(unimpaired_gage_data, output_package = False):
    """
        Calculate Peaks and Thresholds value for the "peaks over threshold implementation"
        Args:
            unimpaired_gage_data -> gage data that has been unimpaired from senior diverters
            output_package -> if running as part of the output package, return an unsorted array of peaks to be given to the user
        Returns:
            peaks - array of peaks
            threshold - value used as a threshold
    """
    df = pd.DataFrame.from_dict(unimpaired_gage_data)
    df = df.fillna(method='ffill')

    unthresholded_peaks = []
    #Go through and find peaks using the sliding window (max in window = peak)
    for index, row in df.iterrows():
        current_flow = float(row['daily_flow'])
        start = max(index-SLIDING_WINDOW_LOOKBACK, 0)
        end = min(index+SLIDING_WINDOW_LOOKBACK+1, len(df))
        is_peak = True
        for i in range(start, end):
            if(index != i and float(df.iloc[i]['daily_flow']) >= current_flow):
                is_peak = False
                break
        if(is_peak):
            unthresholded_peaks.append(current_flow)
    # Calculate threshold index (triple the total number of years)
    # Use a set to ensure that gauge data with gaps is taken into consideration
    all_dates = pd.to_datetime(df['date'], format='%d-%m-%Y')
    # Take only January to avoid water year overlap (cut out first 3 months of "water year")
    all_dates = all_dates[all_dates.dt.month == 1]
    years = set({i for i in all_dates.dt.year})
    triple_years = min(3*(len(years)), len(unthresholded_peaks)-1)
    if(output_package):
        unsorted_peaks = unthresholded_peaks.copy()
    #sort and limit the unthresholded peaks
    unthresholded_peaks.sort(reverse=True)
    #Output at least one peak
    if triple_years == 0:
        triple_years = 1
    thresholded_peaks = unthresholded_peaks[0:triple_years]
    if(output_package):
        return {
            "threshold" : thresholded_peaks[-1],
            "peaks" : thresholded_peaks,
            "record_years": len(years),
            "unsorted_peaks": unsorted_peaks
        }
    return {
        "threshold" : thresholded_peaks[-1],
        "peaks" : thresholded_peaks,
        "record_years": len(years)
    }

def generate_recurrence_intervals(peaks, record_length):
    """
        Generate recurrence intervals from given threshold_peaks_data.
        Recurrence intervals are defined by the Weibull formula.
            T = (N+1)/m
        Args:
            peaks : calculated peaks from above function
            record_length : length of gage record (already has 1 added to it)
        Returns:
            Array of recurrence intervals which correspond to existing array of records
    """
    recurrence_intervals = []
    for i in range(len(peaks)):

        recurrence_intervals.append(record_length/(i+1))
    return recurrence_intervals

def plot_and_find_instantaneous_peak_flow(peaks, recurrence_intervals):
    """
        Find instantaneous peak flow in two ways:
            1. read the 1.5-year recurrence value from the recurrence_intervals plot
            2. create a log-log plot of peaks vs recurrence intervals and estimate from a line of best fit
        Args:
            peaks (x axis)
            recurrence intervals (y axis)
    """
    instantaneous_peak_flows = {}
    #Read value from table:
    first_index_below = 0
    while(recurrence_intervals[first_index_below] > 1.5):
        first_index_below = first_index_below+1
    if(first_index_below == 0):
        instantaneous_peak_flows['table_derived'] = peaks[first_index_below]
    else:
        #take average of values surrounding 1.5
        instantaneous_peak_flows['table_derived'] = (peaks[first_index_below] + peaks[first_index_below-1])/2

    #Don't perform curve fit if only 1 data point
    if(len(peaks) == 1 and len(recurrence_intervals) == 1):
        instantaneous_peak_flows['curve_fit'] = peaks[0]
    else:
        a, b = np.polyfit(np.log10(recurrence_intervals), np.log10(peaks), 1)
        instantaneous_peak_flows['curve_fit'] = 10**(a*np.log10(1.5)+b)
    return instantaneous_peak_flows

def convert_date_to_index(date):
    #Using mod 365 in case there is an end date after october 1
    return (date-dt_date(2018, 10, 1)).days % 365

def generate_yearly_ts_from_row(row):
    """
        Generating the yearly time series for the storage, diversion and frost data for a year.
        Using "water year" from October 1 to September 30. Assume leap year doesn't exist.
        Return array of 365 days worth of data.
    """
    total_demand = [0] * 365
    #Storage
    if(row['days_of_storage'] != None and row['days_of_storage'] > 0):
        if(row['overlapping_days_of_storage_and_policy_season'] > 0):
            #Assume all storage happens in the policy season if overlap
            storage_start = datetime.strptime(row['storage_season_start'], "%d-%m-%Y").date()
            storage_end = datetime.strptime(row['storage_season_end'], "%d-%m-%Y").date()
            both_sides_of_policy_season_case = storage_start.year != policy_season.start.year and storage_end > (policy_season.start.replace(year=2019))
            two_seasons=False
            if(both_sides_of_policy_season_case):
                #This case indicates that we have a storage season that starts at the start of one year and continues into the storage season of the next year
                #In this case, unless we have a full 12 months, we will need two seasons to bridge the gap
                is_full_year = storage_start == datetime(2019, 1, 1).date() and storage_end == datetime(2019, 12, 31).date()
                if(is_full_year):
                    start_date = policy_season.start
                    end_date = policy_season.end + timedelta(days=1)
                elif(storage_start > policy_season.end):
                    #Case where we only have a december section of policy season
                    start_date = policy_season.start
                    end_date = storage_end
                else:
                    two_seasons = True
                    season_one_start = storage_start
                    season_one_end = policy_season.end + timedelta(days=1)
                    season_two_start = policy_season.start.replace(year=2019)
                    season_two_end = storage_end
            else:
                if(policy_season.start > storage_start):
                    start_date = policy_season.start
                else:
                    start_date = storage_start
                if(policy_season.end > storage_end):
                    end_date = storage_end
                else:
                    #Make storage go until EOD March 31
                    end_date = policy_season.end + timedelta(days = 1)
            if(not two_seasons):
                storage_delta = (end_date-start_date).days
                storage_per_day = row['max_storage_af']/storage_delta
                first_index = convert_date_to_index(start_date)
                #Assume storage happens at a constant rate over its season
                for i in range(storage_delta):
                    year_index = (i+first_index) % 365
                    total_demand[year_index] = total_demand[year_index] + storage_per_day
            else:
                storage_delta = (season_one_end-season_one_start).days + (season_two_end-season_two_start).days
                storage_per_day = row['max_storage_af']/storage_delta
                first_season_index = convert_date_to_index(season_one_start)
                second_season_index = convert_date_to_index(season_two_start)
                #Assume storage happens at a constant rate over all of its season
                for i in range((season_one_end-season_one_start).days):
                    year_index = (i+first_season_index) % 365
                    total_demand[year_index] = total_demand[year_index] + storage_per_day
                for i in range((season_two_end-season_two_start).days):
                    year_index = (i+second_season_index) % 365
                    total_demand[year_index] = total_demand[year_index] + storage_per_day
        else:
            #All storage outside of policy season, just use storage start and end dates (ezpz)
            storage_start = datetime.strptime(row['storage_season_start'], "%d-%m-%Y").date()
            storage_end = datetime.strptime(row['storage_season_end'], "%d-%m-%Y").date()
            storage_delta = (storage_end-storage_start).days
            storage_per_day = row['max_storage_af']/storage_delta
            first_index = convert_date_to_index(storage_start)
            #Assume storage happens at a constant rate over its season
            for i in range(storage_delta):
                year_index = (i+first_index) % 365
                total_demand[year_index] = total_demand[year_index] + storage_per_day
    #Diversion
    if(row['days_of_diversion'] != None and int(row['days_of_diversion']) > 0):
        if(type(row['use_codes']) is list):
            use_codes = row['use_codes']
        else:
            use_codes = format_use_codes(row)
        if("Irrigation" in use_codes and len(use_codes) in [1,2] and all(use_code in ['Frost Protection', 'Irrigation'] for use_code in use_codes)):
            #Irrigation case - assume diversion happens OUTSIDE of policy season
            diversion_days_outside_of_policy = int(row['days_of_diversion']) - row['overlapping_days_of_direct_diversion_and_policy_season']
            diversion_start = datetime.strptime(row['direct_div_season_start'], "%d-%m-%Y").date()
            if(policy_season.end > diversion_start and diversion_start > policy_season.start):
                #Start the day after the policy season ends
                start_date = policy_season.end + timedelta(days=1)
            else:
                start_date = diversion_start
            first_index = convert_date_to_index(start_date)
            for i in range(diversion_days_outside_of_policy):
                year_index = (i+first_index) % 365
                total_demand[year_index] = total_demand[year_index] + row['diversion_per_day_af']
        else:
            #Constant over diversion season
            diversion_start = datetime.strptime(row['direct_div_season_start'], "%d-%m-%Y").date()
            first_index = convert_date_to_index(diversion_start)
            for i in range(int(row['days_of_diversion'])):
                year_index = (i+first_index) % 365
                total_demand[year_index] = total_demand[year_index] + row['diversion_per_day_af']
    #Frost
    if(row['frost_demand_af'] > 0):
        #Frost demand occurs every other day between March 15 and April 30th, with some exceptions
        if(row['max_storage_af'] == row['face_amount_af'] and row['overlapping_days_of_storage_and_policy_season'] > 0):
            # Max Storage and Face amount are equal -> assume this means that all diversions go to storage
            # This implies that the frost diversions would happen in the policy season if storage is in policy season
            frost_range = Range(
                start=dt_date(2019, 3, 15),
                end=dt_date(2019, 3, 31)
            )
            storage_range = Range(
                start=datetime.strptime(row["storage_season_start"], "%d-%m-%Y").date(),
                end=datetime.strptime(row["storage_season_end"], "%d-%m-%Y").date()
            )
            overlap = get_month_date_range_overlap(frost_range, storage_range)
            start_date = max(dt_date(2019, 3, 15), storage_range[0])
        else:
            frost_range = Range(
                start=dt_date(2019, 3, 15),
                end=dt_date(2019, 4, 30)
            )
            if(row["direct_div_season_start"] is None or row["direct_div_season_end"] is None):
                # If there aren't diversion dates (storage but storage outside of season), just use the frost range
                diversion_range = frost_range
            else:
                diversion_range = Range(
                    start=datetime.strptime(row["direct_div_season_start"], "%d-%m-%Y").date(),
                    end=datetime.strptime(row["direct_div_season_end"], "%d-%m-%Y").date()
                )
            overlap = get_month_date_range_overlap(frost_range, diversion_range)
            start_date = max(dt_date(2019, 3, 15), diversion_range[0])
        # Every other day
        overlap = overlap // 2
        first_index = convert_date_to_index(start_date)
        frost_per_day = row['frost_demand_af']/overlap
        for i in range(overlap):
            total_demand[i*2 + first_index] = total_demand[i*2 + first_index] + frost_per_day
    if(row['seasonal_demand_af'] != None and row['seasonal_demand_af'] > 0):
        #We want to downscale or upscale to user-entered demand if it exists
        overall_demand = row['max_storage_af'] + row['diversion_amount_af'] + row['frost_demand_af']
        ratio = row['seasonal_demand_af']/overall_demand
        for i in range(len(total_demand)):
            total_demand[i] = total_demand[i]* ratio
    elif(row['face_amount_af'] is not None):
        #scale down so total is not greater than face_amount_af
        total_sum_af = sum(total_demand)
        if(row['face_amount_af'] < total_sum_af):
            ratio = row['face_amount_af']/total_sum_af
            for i in range(len(total_demand)):
                total_demand[i] = total_demand[i] * ratio
    #Finally, convert to cfs
    for i in range(len(total_demand)):
        total_demand[i] = total_demand[i]* AFD_TO_CFS
    return total_demand

def get_first_water_year_after_date(date):
    if date is None:
        return None
    year = int(date.split('-')[0])
    month = int(date.split('-')[1])
    if(month > 10):
        return year + 1
    else:
        return year

def generate_senior_diverter_ts(senior_diverters_data, start_year):
    """
        Generates time series of the form:
            Imagine we have 4 days in the analysis (instead of a full year).
            There are the following water rights:
            wr1: started 1965, [1,1,1,1]
            wr2: started 1980, [2,2,2,2]
            wr3: started 2000, [3,3,3,3]
            wr4: started 2000, [0.5, 0.5, 0.5, 0.5]
            wr5: started 2015, [4,4,4,4]
            The gage began getting data in 1990, so the following would be the multi-year dataset
            {"1990" : [3,3,3,3], <-- two existing water rights summed together at start of gage
            "2000" : [6.5,6.5,6.5,6.5], <-- add wr3+wr4 as two started in 2000
            "2015" : [10.5,10.5,10.5,10.5]} <-- add wr5 as it started in 2015
        Args:
            senior_diverters_data - data from the senior diverters
            start_year - year the gage began
        Returns
            Data structure as above
    """
    df = pd.DataFrame.from_dict(senior_diverters_data)
    if(df.empty):
       return {f"{start_year}": [0]*365}
    df['first_year_of_diversion'] = df['priority_date'].apply(get_first_water_year_after_date)
    df['first_year_of_diversion'] = df['first_year_of_diversion'].fillna(start_year)
    df.sort_values(by='first_year_of_diversion', inplace=True)
    df['use_codes'] = df.apply(format_use_codes, axis=1)
    senior_diverter_ts = {}
    for index, row in df.iterrows():
        yearly_ts = generate_yearly_ts_from_row(row)
        diversion_start_year = int(row['first_year_of_diversion'])
        if(start_year != None and diversion_start_year <= start_year):
            #Sum all of the values less than the given start year into an initial series
            if(len(senior_diverter_ts)== 0):
                #First entry case
                senior_diverter_ts[str(start_year)] = yearly_ts
            else:
                #Other case, add old diverters
                existing_ts = senior_diverter_ts[max(senior_diverter_ts)].copy()
                for i in range(len(yearly_ts)):
                    existing_ts[i] = existing_ts[i] + yearly_ts[i]
                senior_diverter_ts[str(start_year)] = existing_ts
        elif(str(diversion_start_year) in senior_diverter_ts):
            #Multiple diverters in same year case
            existing_ts = senior_diverter_ts[str(diversion_start_year)].copy()
            for i in range(len(yearly_ts)):
                existing_ts[i] = existing_ts[i] + yearly_ts[i]
            senior_diverter_ts[str(diversion_start_year)] = existing_ts
        else:
            #New diverter this year case
            if(len(senior_diverter_ts)== 0):
                #First entry case
                senior_diverter_ts[str(diversion_start_year)] = yearly_ts
            else:
                #Other case, add old diverters
                existing_ts = senior_diverter_ts[max(senior_diverter_ts)].copy()
                for i in range(len(yearly_ts)):
                    existing_ts[i] = existing_ts[i] + yearly_ts[i]
                senior_diverter_ts[str(diversion_start_year)] = existing_ts
    #Add check if gage data begins earlier than diverter data
    years = list(map(lambda x: int(x), senior_diverter_ts.keys()))
    if(min(years) > start_year):
        #Add 0 row at start year
        senior_diverter_ts[f"{start_year}"] = [0]*365
    return senior_diverter_ts


def generate_non_onstream_dam_diversions(row, non_onstream_diverters):
    """
        Given a senior diverter data structure, sort into onstream dams and other diverters.
        For the diverters that are not onstream dams, generate their yearly diversion time series.
    """
    if(pd.isna(row['pod_type']) or not "Point of Onstream Storage" in row['pod_type']):
        # An onstream dam has different demands for diversions than a point of direct diversion or storage
        yearly_ts = generate_yearly_ts_from_row(row)
        non_onstream_diverters[row['order_upstream_to_downstream']] = yearly_ts


def generate_point_of_onstream_storage_output(year, diverter, yearly_diversion, upstream_diverters, gage_ratio_raw):
    """
        For a point of onstream storage, generate its diversions for the year based on the impacts of its upstream diverters and the flow available in the stream.
        Uses a "fill and spill" approach, where all water is taken from the stream until the face_value is reached.
        Water available in the stream is calculated from the gaged streamflow data scaled to the diverter watershed.
    """
    total_upstream_diversions = [0] * 365
    for order in upstream_diverters:
        total_upstream_diversions = np.add(total_upstream_diversions, yearly_diversion[order]).tolist()
    i = 0
    total_in_reservoir = 0
    diverter_ratio_data = {'drainage_area_sqmi': diverter['drainage_area_sqmi'], 'map_1991_2020_in': diverter['annual_precip_in']}
    scaling_ratio = calculate_cda_ratio(diverter_ratio_data, gage_ratio_raw)['ratio']
    water_year_start = datetime.strptime("01-10-2019", "%d-%m-%Y")
    index_of_storage_season_start = (datetime.strptime(f"{int(diverter['storage_season_start_day'])}-{int(diverter['storage_season_start_month'])}-2019", "%d-%m-%Y") - water_year_start).days
    if(index_of_storage_season_start < 0):
        index_of_storage_season_start = 365 + index_of_storage_season_start
    index_of_storage_season_end = (datetime.strptime(f"{int(diverter['storage_season_end_day'])}-{int(diverter['storage_season_end_month'])}-2019", "%d-%m-%Y") - water_year_start).days
    if(index_of_storage_season_end < 0):
        index_of_storage_season_end = 365 + index_of_storage_season_end
    # Check if time range spans water year
    spans_water_year = index_of_storage_season_end < index_of_storage_season_start
    diverter_face_value = diverter['face_amount_af'] * AFD_TO_CFS
    # If there is a minimum bypass flow specified in the license, use it as a minimum flow that the dam allows through
    diverter_minimum_bypass_flow = diverter['minimum_bypass_flow_cfs']
    if(pd.isna(diverter_minimum_bypass_flow)):
        diverter_minimum_bypass_flow = 0
    current_diverter_flow = [0] * 365
    for _, day in year.iterrows():
        # I know iterrows isn't the best, but we need to continually pass through these values to work
        # towards the total, it is a definitely iterative process for this unfortunately
        if(day['date'].startswith("29-02")):
            # Skip leap year
            continue
        raw_flow_data = day['daily_flow']
        if(pd.isna(raw_flow_data)):
            raw_flow_data = 0
        scaled_flow = raw_flow_data * scaling_ratio
        available_flow = max(scaled_flow - total_upstream_diversions[i] - diverter_minimum_bypass_flow, 0)
        # Check if in season
        if(not spans_water_year and (index_of_storage_season_end >= i and index_of_storage_season_start <= i)
           or spans_water_year and (index_of_storage_season_end >= i or index_of_storage_season_start <= i)):
            current_diverter_flow[i] = min(available_flow, diverter_face_value - total_in_reservoir)
            total_in_reservoir += min(available_flow, diverter_face_value - total_in_reservoir)
        i += 1
        if(total_in_reservoir == diverter_face_value):
            # Onstream Reservoir is full
            break
    return current_diverter_flow


def generate_diversions_for_water_year(year, senior_diverters_df, non_onstream_diverters, water_year_total_diversions, water_year_diversions, onstream_storage_upstream_diverters, gage_ratio_raw):
    """
        For a given water year, generate the senior diversions for the given senior diverters.
        This analysis takes into account onstream dam points of storage that have a "fill and spill" approach.
        Args:
            year -> water year of record
            senior_diverters_df -> dataframe of senior diverters, will work upstream to downstream for demands
            onstream_storage_diverters -> list of diverters making diversions to onstream storage
            non_onstream_diverters -> list of diverters and their pre-computed yearly diversion time series
    """
    order_upstream_to_downstream = []
    if(not senior_diverters_df.empty):
        order_upstream_to_downstream = sorted(senior_diverters_df['order_upstream_to_downstream'].tolist())
    yearly_diversions = {}
    total_yearly_diversions = [0] * 365
    for order in order_upstream_to_downstream:
        if order in onstream_storage_upstream_diverters.keys():
            storage_diverter_output = generate_point_of_onstream_storage_output(year,
                                                                                senior_diverters_df[senior_diverters_df['order_upstream_to_downstream'] == order].iloc[0],
                                                                                yearly_diversions,
                                                                                onstream_storage_upstream_diverters[order],
                                                                                gage_ratio_raw)
            yearly_diversions[order] = storage_diverter_output
            total_yearly_diversions = np.add(total_yearly_diversions, storage_diverter_output).tolist()
        else:
            yearly_diversions[order] = non_onstream_diverters[order]
            total_yearly_diversions = np.add(total_yearly_diversions, non_onstream_diverters[order]).tolist()
    water_year = year.iloc[0]['water_year']
    water_year_diversions[water_year.item()] = yearly_diversions
    water_year_total_diversions[water_year.item()] = total_yearly_diversions

def generate_senior_diverter_ts_poi(senior_diverters_data, unimpaired_gage_ts, onstream_storage_upstream_diverters, gage_ratio_raw, output_package = False):
    """
    Generate a time series of the form:
        {<year> : [365 days of yearly diversions]}
    Diversions are generated in two different ways:
        1. If a diverter is a point of offstream storage or a diversion of flow from a stream, it is said to average its diversions over its diversion or storage season.
            This is done by taking its face value and spreading it over its season, with each day getting an equal amount of water.
        2. If a diverter is an onstream dam (pod_type = "Point of Onstream Storage") it uses a "fill and spill approach"
            This is done by diverting all water (other than that necessary to maintain the minimum bypass flow in the license) to the diverter.
            Once its face value is reached, the dam is said to be full, and all additional flow "spills" over the end of the dam and there are no diversions
    If the output_package field is True, the output is broken down by diverter (necessary for output package formatting)
        {<year> : {<diverter_1> : [365 days], <diverter_2> : [365 days] ... etc.}}
    Args:
        senior_diverters_data -> raw data of the user's senior diverters
        unimpaired_gage_ts -> gage data unimpaired to use as a baseline for flow
        onstream_storage_upstream_diverters -> list of the upstream diverters for each onstream dam, used to estimate the flow at the dam
        gage_ratio_raw -> mean precipitation and drainage area information for the gage to scale the gage streamflow to a diverter watershed
        output_package -> boolean to indicate whether the data is being prepared for the output package or for API consumption
    """
    senior_diverters_df = pd.DataFrame.from_dict(senior_diverters_data)
    if(not senior_diverters_df.empty):
        senior_diverters_df['use_codes'] = senior_diverters_df.apply(format_use_codes, axis=1)
    non_onstream_diverters = {}
    senior_diverters_df.apply(lambda row, non_onstream_diverters :
                        generate_non_onstream_dam_diversions(row, non_onstream_diverters),
                        non_onstream_diverters = non_onstream_diverters,
                        axis = 1)
    unimpaired_gage_df = pd.DataFrame.from_dict(unimpaired_gage_ts)
    unimpaired_gage_df['water_year'] = unimpaired_gage_df['date'].apply(lambda date: generate_water_year(date))
    water_year_total_diversions = {}
    water_year_diversions = {}
    unimpaired_gage_df.groupby('water_year').apply(lambda year, senior_diverters_df, non_onstream_diverters, water_year_total_diversions, water_year_diversions, onstream_storage_upstream_diverters, gage_ratio_raw:
                                                    generate_diversions_for_water_year(
                                                            year,
                                                            senior_diverters_df,
                                                            non_onstream_diverters,
                                                            water_year_total_diversions,
                                                            water_year_diversions,
                                                            onstream_storage_upstream_diverters,
                                                            gage_ratio_raw),
                                                    senior_diverters_df = senior_diverters_df,
                                                    non_onstream_diverters = non_onstream_diverters,
                                                    water_year_total_diversions = water_year_total_diversions,
                                                    water_year_diversions = water_year_diversions,
                                                    onstream_storage_upstream_diverters = onstream_storage_upstream_diverters,
                                                gage_ratio_raw = gage_ratio_raw)
    if(output_package):
        return water_year_diversions
    return water_year_total_diversions

def impair_poi_time_series(unimpaired_poi_ts, upstream_diverters_ts):
    """
        Impair the unimpaired time series at the poi using the generated time series from generate_senior_diverter_ts_poi.
        Fill down to 0 if necessary.
        Args:
            unimpaired_poi_ts -> time series of flow scaled to the POI watershed
            upstream_diverters_ts -> ttime series of the diversions upstream of the POI
    """
    unimpaired_poi_df = pd.DataFrame.from_dict(unimpaired_poi_ts)
    upstream_diverters_full_ts = []
    for year in upstream_diverters_ts.keys():
        start_of_water_year = dt_date(year-1, 10, 1)
        index = 0
        for day in upstream_diverters_ts[year]:
            date = start_of_water_year + timedelta(days = index)
            index += 1
            output = {'date': date.strftime("%d-%m-%Y"), 'demand': day}
            upstream_diverters_full_ts.append(output)
    upstream_diverters_df = pd.DataFrame.from_dict(upstream_diverters_full_ts)
    merged_df = unimpaired_poi_df.merge(upstream_diverters_df, on='date')
    merged_df['impaired'] = merged_df['daily_flow'] - merged_df['demand']
    merged_df['impaired'] = merged_df['impaired'].clip(lower = 0)
    merged_df = merged_df[['date', 'impaired']]
    merged_df.rename(columns = {'impaired': 'daily_flow'}, inplace=True)
    return merged_df.to_dict(orient='records')

def calculate_yearly_peaks(unimpaired_gage_df, output_package=False):
    """
        Calculate peak for each year of the unimpaired gage time-series.
        Args:
            unimpaired_gage_df - time series data frame
        Returns:
            Array of highest flows for each year
    """
    df = unimpaired_gage_df.copy()

    yearly_peaks = []
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    unique_years = df['date'].dt.year.unique()
    if(output_package):
        yearly_data = {}
    for year in unique_years:
        year_data = df[df['date'].dt.year == year]
        if(output_package):
            yearly_data[year] = max(year_data['daily_flow'])
        yearly_peaks.append(max(year_data['daily_flow']))
    if(output_package):
        return (yearly_peaks, yearly_data)
    return yearly_peaks

def calculate_peak_flow_yearly_method(unimpaired_gage_data):
    """
        Uses the "yearly flow method" to calculate 1.5-year instantaneous peak flow.
        Simply put - finds the peak of each year and finds the one which occurs (on average), every year and a half.
        Args:
            unimpaired_gage_data - time series
        Returns:
            1.5 year instantaneous peak.
    """
    df = pd.DataFrame.from_dict(unimpaired_gage_data)
    df = df.fillna(method='ffill')

    yearly_peaks = calculate_yearly_peaks(df)
    yearly_peaks.sort(reverse=True)
    #We can use generate_recurrence intervals function here as well!
    recurrence_intervals = generate_recurrence_intervals(yearly_peaks, len(yearly_peaks))

    first_index_below = 0
    while(recurrence_intervals[first_index_below] > 1.5):
        first_index_below = first_index_below+1
    if(first_index_below == 0):
        instantaneous_peak_flow = yearly_peaks[first_index_below]
    else:
        #take average of values surrounding 1.5
        instantaneous_peak_flow = (yearly_peaks[first_index_below] + yearly_peaks[first_index_below-1])/2

    return instantaneous_peak_flow

def calculate_mcd(instantaneous_peak_flow, ratio):
    """
        Calculate maximum cumulative diversion (5% of peak flow)
        Args:
            instantaneous_peak_flow - instantaneous peak flow at the gage
            ratio - ratio between gage and poi flows
        Returns:
            maximum cumulative diversion value
    """
    return instantaneous_peak_flow*ratio*0.05

def calculate_natural_flow_variability(daily_time_seriess):
    """
        Calculate the "Evaluate whether the proposed project contributes to reductions in natural flow variability"(B5.3.5).
        This involves calling the peaks over threshold functionality and calculating ratios.
        See https://foundryspatial.atlassian.net/wiki/x/G4BldQ for more information!
        Args:
            daily_time_seriess - unimpaired and impaired daily time series for analysis
        Returns:
            Data structure containing the necessary data for the natural flow variability user display.
    """
    # Use peaks over threshold for calculation
    peaks_thresholds = peaks_and_threshold(daily_time_seriess['unimpaired'], output_package=False)
    recurrence_intervals = generate_recurrence_intervals(peaks_thresholds['peaks'], peaks_thresholds['record_years'])
    peak_unimpaired = plot_and_find_instantaneous_peak_flow(peaks_thresholds['peaks'], recurrence_intervals)['curve_fit']

    peaks_thresholds = peaks_and_threshold(daily_time_seriess['impaired_with_diverters'], output_package=False)
    recurrence_intervals = generate_recurrence_intervals(peaks_thresholds['peaks'], peaks_thresholds['record_years'])
    peak_diverters = plot_and_find_instantaneous_peak_flow(peaks_thresholds['peaks'], recurrence_intervals)['curve_fit']

    peaks_thresholds = peaks_and_threshold(daily_time_seriess['impaired_with_pod'], output_package=False)
    recurrence_intervals = generate_recurrence_intervals(peaks_thresholds['peaks'], peaks_thresholds['record_years'])
    peak_pod = plot_and_find_instantaneous_peak_flow(peaks_thresholds['peaks'], recurrence_intervals)['curve_fit']

    #Multiply by 100 to convert to percentages
    ratio_unimpaired_diverters = (1 - peak_diverters/peak_unimpaired) * 100
    ratio_unimpaired_pod = (1 - peak_pod/peak_unimpaired) * 100
    ratio_difference = ratio_unimpaired_pod - ratio_unimpaired_diverters

    under_reduction_limit = (ratio_unimpaired_diverters == ratio_unimpaired_pod or ratio_unimpaired_pod < 5)

    return {"instantaneousPeakUnimpaired": peak_unimpaired,
            "instantaneousPeakDiverters": peak_diverters,
            "instantaneousPeakPod": peak_pod,
            "ratioUnimpairedDiverters": ratio_unimpaired_diverters,
            "ratioUnimpairedPod": ratio_unimpaired_pod,
            "ratioDifference" : ratio_difference,
            "underReductionLimit" : bool(under_reduction_limit)}

def scale_gage_ts_to_poi(unimpaired_gage_data, ratio):
    """
        Given the unimpaired gage data time series, scale to work with the POI using the poi ratio.
        Args:
            unimpaired_gage_data - from the database, the gage data
            ratio - poi proration ratio (calculated by calculate_cda_ratio function)
        Returns:
            unimpaired poi data
    """
    df = pd.DataFrame.from_dict(unimpaired_gage_data)
    if(ratio < 0):
        raise Exception("Can't enter negative ratio to scale gage time-series")
    if('daily_flow' in df.columns):
        df['daily_flow'] = df['daily_flow'].apply(lambda x: x*ratio)
    return df

def generate_pod_diversion_row(senior_diverter_df, cda_session):
    """
        Generate a csv row for the pod of its diversion as if the pod were a senior diverter.
        Args:
            senior_diverter_csv - contains existing pod data
            cda_session - contains user-entered pod data
        Returns:
            row of pod, formatted as we would like
    """
    #Squeeze into pandas series so that we can operate as a series
    pod_row = senior_diverter_df.loc[senior_diverter_df.analysis_label == "Proposed POD"].squeeze()
    if('seasonOfDiversionStart' in cda_session):
        start_datetime = pd.to_datetime(cda_session["seasonOfDiversionStart"], dayfirst=True).tz_localize(None)
        pod_row['direct_div_season_start_month'] = start_datetime.month
        pod_row['direct_div_season_start_day'] = start_datetime.day
    else:
        raise Exception("No season of diversion start in given cda session!")
    if('seasonOfDiversionEnd' in cda_session):
        end_datetime = pd.to_datetime(cda_session["seasonOfDiversionEnd"], dayfirst=True).tz_localize(None)
        pod_row['direct_div_season_end_month'] = end_datetime.month
        pod_row['direct_div_season_end_day'] = end_datetime.day
    else:
        raise Exception("No season of diversion end in given cda session!")
    if('volumeOfDiversion' in cda_session):
        volume_of_diversion = cda_session['volumeOfDiversion']
        if(volume_of_diversion['unit'] == 'gallons'):
            total_diversion = float(volume_of_diversion['value']) / float(GALLONS_TO_AF)
        else:
            total_diversion = volume_of_diversion['value']
        pod_row['face_amount_af'] = total_diversion
    else:
        raise Exception("No volume of diversion in given cda session!")
    return pod_row


def find_closest_water_right(lat, long, senior_diverter_df):
    """
        Find the closest water right to a given lat and long from the user's senior_diverters
        Args:
            lat - latitude
            long - longitude
        Returns:
            water right id of closest water right
    """
    min_index = 0
    for index, row in senior_diverter_df.iterrows():
        distance = math.sqrt((row['latitude']-lat)**2+ (row['longitude']-long)**2)
        if(index == 0):
            min_distance = distance
        elif(min_distance > distance):
            min_distance = distance
            min_index = index
    #Found the POD - no water right id
    if(pd.isna(senior_diverter_df.iloc[min_index-1]['wr_water_right_id']) or senior_diverter_df.iloc[min_index-1]['wr_water_right_id'] ==None):
        return -1
    if(min_index == 0):
        return senior_diverter_df.iloc[min_index]['wr_water_right_id']
    else:

        return senior_diverter_df.iloc[min_index-1]['wr_water_right_id']

def get_senior_diverters_upstream_of_poi(senior_diverter_csv, poi, cda_session):
    """
        Gets the senior diverters upstream of a given poi from the user's given senior diverters.
        Creates datasets with and without the user's given POD to generate the correct time-series.
        Args:
            senior_diverter_csv - the user's senior diverters with intermediate values
            poi - poi data object
        Returns:
            senior diverters, limited to those above the poi.
    """
    df = pd.DataFrame.from_dict(senior_diverter_csv)
    index_of_pod = df.index[df['analysis_label'] == 'Proposed POD'].tolist()[0]
    #Create the pod diversion row (as if the user-entered POD was a senior diverter)
    pod_row = generate_pod_diversion_row(df, cda_session)
    df.drop(index_of_pod)
    df.loc[index_of_pod] = pod_row
    # If the user has entered an upstream senior diverter, default to that
    upstream_water_right = None
    if('upstreamWaterRight' in poi):
        if(poi['upstreamWaterRight'] == 'null'):
            upstream_water_right = -1
        elif(float(poi['upstreamWaterRight']) in df['wr_water_right_id'].tolist()):
            upstream_water_right = float(poi['upstreamWaterRight'])
    else:
        upstream_water_right = find_closest_water_right(poi['lat'], poi['long'], df)
    if(upstream_water_right == -1):
        #POD was closest - doesn't have a wr id
        index_of_wr = index_of_pod
    else:
        index_of_wr = df.index[df['wr_water_right_id'] == upstream_water_right].tolist()[0]
    if(index_of_wr < index_of_pod):
        raise Exception("Cannot have all diversions for POI upstream of the POD!")
    df = df.iloc[0: index_of_wr+1]
    calculated_df = calculate_cda_intermediate_values(df)
    only_senior_diverters = calculated_df.drop(axis=1, index = index_of_pod)
    # mock data coming from database using psycopg2 datatype so the pipeline remains the same
    converted_diverters = [RealDictRow(row) for row in only_senior_diverters.to_dict(orient='records')]
    converted_diverters_with_pod = [RealDictRow(row) for row in calculated_df.to_dict(orient='records')]
    return (converted_diverters, converted_diverters_with_pod)

def filter_df_to_date_range(df, start_date, end_date):
    """
    Filters a time-series dataframe to only those values within a specific date range.

    Args:
        df - DataFrame
        start_date - start date (inclusive) as a datetime.datetime object
        end_date - end date (inclusive) as a datetime.datetime object

    Returns:
        filtered_df - DataFrame filtered to the specified date range
    """
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
    filtered_df = df[(df['date'].dt.month >= start_date.month) & (df['date'].dt.day >= start_date.day) & (df['date'].dt.month <= end_date.month) & (df['date'].dt.day <= end_date.day)]
    return filtered_df

def get_months_between_dates(start_date, end_date):
    """
    Args:
        start_date - datetime.datetime (inclusive) first day of date range
        end_date - datetime.datetime (inclusive) last day of the date range

    Returns - a list of 3-tuples that each contain the start and end date for a month within a range aswell as a flag to indicate if the month is complete or not, a majority will be the first and last day of the month with true as the flag but the 'ends' of the range may be partial months.
    """
    current_date = start_date.replace(day=1)
    month_ranges = []

    while current_date <= end_date:
        partial_month = False
        month_start = current_date
        next_month = current_date + relativedelta(months=1)
        month_end = next_month - relativedelta(days=1)

        # handle the ends of the range
        if month_start < start_date:
            month_start = start_date
            partial_month = True
        if month_end > end_date:
            month_end = end_date
            partial_month = True

        month_ranges.append((month_start, month_end, partial_month))
        current_date = next_month

    return month_ranges

def calculate_instream_flows_reduction(daily_time_seriess, threshold, start_date, end_date):
    """
        Perform the instream flows reduction calculation (section B5.3.4 of the policy)
        Args:
            daily_time_seriess - data structure containing the daily time series's for the POI necessary for comparison
            threshold - calculated threshold value for the POI, is either MBF or feb median for daily flow study
            start_date - the start date of the diversion season to be analyzed, a string expected to be in the format %Y-%m-%dT%H:%M:%S.%fZ
            end_date - the end date of the diversion season to be analyzed, a string expected to be in the format %Y-%m-%dT%H:%M:%S.%fZ
        Returns:
            Data structure containing the daily time series analysis values
    """
    date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    start_date_obj = datetime.strptime(start_date, date_format)
    end_date_obj = datetime.strptime(end_date, date_format)
    # set years to be in a leap year so thatwe include feb 29th in our calculations
    if start_date_obj.year == end_date_obj.year:
        start_date_obj = start_date_obj.replace(year=2000)
        end_date_obj = end_date_obj.replace(year=2000)
    else:
        start_date_obj = start_date_obj.replace(year=1999)
        end_date_obj = end_date_obj.replace(year=2000)
    months_to_calculate = get_months_between_dates(start_date_obj,end_date_obj)
    unimpaired_poi_df = pd.DataFrame.from_dict(daily_time_seriess['unimpaired'])
    diverter_impaired_df = pd.DataFrame.from_dict(daily_time_seriess['impaired_with_diverters'])
    pod_impaired_df = pd.DataFrame.from_dict(daily_time_seriess['impaired_with_pod'])
    # if any values are nan (sometimes happens) forward-fill
    unimpaired_poi_df.fillna(method='ffill', inplace=True)
    diverter_impaired_df.fillna(method='ffill', inplace=True)
    pod_impaired_df.fillna(method='ffill', inplace=True)
    unimpaired_month_exceedances = [0] * len(months_to_calculate)
    diverter_month_exceedances = [0] * len(months_to_calculate)
    pod_month_exceedances = [0] * len(months_to_calculate)
    percentage_impaired = []
    percentage_pod = []
    months_calculated = []
    month_is_partial = []
    mbf = threshold
    for index, month in enumerate(months_to_calculate):
        # Filter to each month
        unimpaired_month = filter_df_to_date_range(unimpaired_poi_df, month[0], month[1])
        diverter_impaired_month = filter_df_to_date_range(diverter_impaired_df, month[0], month[1])
        pod_impaired_month = filter_df_to_date_range(pod_impaired_df, month[0], month[1])
        # Get the mbf exceedance count for each month
        unimpaired_month_exceedances[index] = len(unimpaired_month[unimpaired_month['daily_flow'] >= mbf].index)
        diverter_month_exceedances[index] = len(diverter_impaired_month[diverter_impaired_month['daily_flow'] >= mbf].index)
        pod_month_exceedances[index] = len(pod_impaired_month[pod_impaired_month['daily_flow'] >= mbf].index)
        # Get the percentage impaired and see whether the water is available
        percentage_impaired_month = {}
        percentage_pod_month = {}
        if(unimpaired_month_exceedances[index] == 0):
            if(diverter_month_exceedances[index] != 0):
                percentage_impaired_month['percentage'] = 100
            else:
                percentage_impaired_month['percentage'] = 0
            if(pod_month_exceedances[index] != 0):
                percentage_pod_month['percentage'] = 100
            else:
                percentage_pod_month['percentage'] = 0
        else:
            percentage_impaired_month['percentage'] = (unimpaired_month_exceedances[index]-diverter_month_exceedances[index])/float(unimpaired_month_exceedances[index]) * 100
            percentage_pod_month['percentage'] = (unimpaired_month_exceedances[index]-pod_month_exceedances[index])/float(unimpaired_month_exceedances[index]) * 100
        percentage_impaired_month['waterAvailable'] = percentage_impaired_month['percentage'] <= 10
        percentage_pod_month['waterAvailable'] = percentage_pod_month['percentage'] <= 10
        percentage_impaired.append(percentage_impaired_month)
        percentage_pod.append(percentage_pod_month)
        months_calculated.append(month[0].month - 1) # Front end dev requests that we represent months as 0 indexed integers
        month_is_partial.append(month[2])
    return {"unimpairedExceedances" : unimpaired_month_exceedances,
            "diverterImpairedExceedances" : diverter_month_exceedances,
            "podImpairedExceedances" : pod_month_exceedances,
            "percentageImpaired" : percentage_impaired,
            "percentagePod" : percentage_pod,
            "months": months_calculated,
            "is_partial_month": month_is_partial}
