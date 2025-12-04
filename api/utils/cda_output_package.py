from copy import deepcopy
import csv
import datetime
from io import BytesIO
import os
import time
import zipfile
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import requests

from utils.cda_utils import (
    AFD_TO_CFS,
    SLIDING_WINDOW_LOOKBACK,
    calculate_yearly_peaks,
    generate_recurrence_intervals,
    generate_senior_diverter_ts_poi,
    generate_water_year,
    generate_yearly_ts_from_row,
    format_use_codes,
    get_senior_diverters_upstream_of_poi,
    peaks_and_threshold,
    plot_and_find_instantaneous_peak_flow,
    get_first_water_year_after_date
)


def generate_yearly_data(calculated_data):
    """
    Generates a pandas dataframe with columns "water_year" and "total_diversion" for output package
    Args:
        calculated_data - the calcualted data, with the yearly_total_diversion_af
    Returns:
        Dataframe as above
    """
    df = calculated_data.copy()
    df['first_year_of_diversion'] = df['priority_date'].apply(get_first_water_year_after_date)
    df["first_year_of_diversion"] = df["first_year_of_diversion"].fillna(
        min(df["first_year_of_diversion"])
    )
    df.sort_values(by="first_year_of_diversion", inplace=True)
    output_list = []
    current_diversion = 0
    this_year = datetime.date.today().year
    for i in range(int(min(df["first_year_of_diversion"])), this_year):
        if not df[df["first_year_of_diversion"] == i].empty:
            for index, row in df[df["first_year_of_diversion"] == i].iterrows():
                demand = row["wr_seasonal_demand"]
                if pd.isna(demand):
                    demand = 0
                current_diversion = current_diversion + demand
        output_list.append({"water_year": i, "total_diversion_af": current_diversion})
    return pd.DataFrame.from_dict(output_list)


def format_output_gage_data(gage_csvs):
    """
    Formats the gage csv data as wanted by the output package
    Args:
        gage_csvs - Dictionary containing the raw, user-uploaded, and gage data with calculated values
    Outputs:
        Pandas dataframes of the outputs, to be put together as csvs by outer functions
    """
    if(gage_csvs is None):
        return {}
    raw_data = pd.DataFrame.from_dict(gage_csvs["raw_diverters"])
    if(raw_data.empty):
        #Gage does not have senior diverters case
        return {}
    raw_data = raw_data[
        [
            "application_number",
            "water_right_type",
            "water_right_status",
            "face_amount_af",
            "priority_date",
            "max_rate_of_diversion_cfs",
        ]
    ]
    raw_data.rename(
        columns={
            "face_amount_af": "face_value_amount",
            "max_rate_of_diversion_cfs": "diversion_rate_cfs",
        },
        inplace=True,
    )
    calculated_data = pd.DataFrame.from_dict(gage_csvs["intermediate"])
    if not calculated_data.empty:
        yearly_data = generate_yearly_data(calculated_data)
        calculated_data = calculated_data[
            [
                "application_number",
                "water_right_type",
                "water_right_status",
                "face_amount_af",
                "priority_date",
                "max_rate_of_diversion_cfs",
                "wr_seasonal_demand",
            ]
        ]
        calculated_data.rename(
            columns={
                "face_amount_af": "face_value_amount",
                "max_rate_of_diversion_cfs": "diversion_rate_cfs",
                "wr_seasonal_demand": "yearly_total_diversion_af",
            },
            inplace=True,
        )
    else:
        # If no calculated data just set to empty df
        yearly_data = pd.DataFrame()
    output_data = {}
    output_data["senior_diverters_upstream_of_gage_raw_B.5.2.1-A3"] = raw_data
    if not calculated_data.empty:
        output_data[
            "senior_diverters_upstream_of_gage_edited_B.5.2.1-A3"
        ] = calculated_data
    if not yearly_data.empty:
        output_data[
            "senior_diverters_upstream_of_gage_water_year_total_diversion_B.5.2.1-A3"
        ] = yearly_data
    return output_data


def peaks_over_threshold_output(
    peaks_thresholds,
    recurrence_intervals,
    instantaneous_flow,
    unimpaired_gage_data,
    poi_id_str=None,
):
    """
    Format and generate output for peaks over threshold method.
    This is calculated for the unimpaired gage time series and all the POI time series.
    Args:
        peaks_thresholds - dictionary of peaks and threshold data
        recurrence_intervals - list of recurrence intervals - same length as the peaks list
        daily_flow - data structure containing the curve and table found daily flow values
    Returns:
        dictionary of output dataframes, and png files of associated plots
    """
    full_timeseries_df = pd.DataFrame.from_dict(unimpaired_gage_data)
    full_timeseries_df = full_timeseries_df.fillna(method="ffill")
    full_timeseries_df["is_peak"] = [0] * len(full_timeseries_df.index)
    full_timeseries_df["peak_value"] = [0] * len(full_timeseries_df.index)
    for index, row in full_timeseries_df.iterrows():
        current_flow = float(row['daily_flow'])
        start = max(index-SLIDING_WINDOW_LOOKBACK, 0)
        end = min(index+SLIDING_WINDOW_LOOKBACK+1, len(full_timeseries_df.index))
        is_peak = True
        for i in range(start, end):
            if(index != i and float(full_timeseries_df.iloc[i]['daily_flow']) >= current_flow):
                is_peak = False
                break
        if(is_peak):
            full_timeseries_df.loc[index, "is_peak"] = 1
            full_timeseries_df.loc[index, "peak_value"] = current_flow
    peaks = peaks_thresholds["peaks"]
    thresholds_df = pd.DataFrame()
    thresholds_df["rank"] = list(range(1, len(peaks) + 1))
    thresholds_df["daily_peak_flow_cfs"] = peaks
    thresholds_df["recurrence_interval"] = recurrence_intervals

    calculated_df = pd.DataFrame.from_dict([instantaneous_flow])
    calculated_df["threshold"] = [peaks_thresholds["threshold"]]
    output_csvs = {}
    if poi_id_str is None:
        output_csvs[
            "peaks_over_threshold_time_series_B.5.2.3-A_gage"
        ] = full_timeseries_df
        output_csvs["peaks_over_threshold_B.5.2.3-A_gage"] = thresholds_df
        output_csvs["maximum_cumulative_diversion_gage"] = calculated_df
    else:
        output_csvs[
            f"peaks_over_threshold_time_series_B.5.2.3-A_poi_{poi_id_str}"
        ] = full_timeseries_df
        output_csvs[f"peaks_over_threshold_B.5.2.3-A_poi_{poi_id_str}"] = thresholds_df
        output_csvs[f"peaks_over_threshold_results_poi_{poi_id_str}"] = calculated_df
    output_plots = {}
    fig = Figure(figsize=(10, 7), dpi=80)
    ax = fig.subplots()
    a, b = np.polyfit(np.log10(recurrence_intervals), np.log10(peaks), 1)
    ax.loglog(recurrence_intervals, peaks, label="fit", color="green")
    ax.loglog(
        recurrence_intervals,
        10 ** (float(a) * np.log10(recurrence_intervals) + float(b)),
        color="blue",
    )
    ax.annotate(
        f"1.5 Year Peak Flow: {10**(float(a)*np.log10(1.5) + float(b)): .3f}",
        xy=(1.5, 10 ** (float(a) * np.log10(1.5) + float(b))),
        xytext=(1.5, 10 ** (float(a) * np.log10(1.5) + float(b)) / 1.6),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    fig.suptitle(
        f"Peaks vs Recurrence intervals \n Line of best fit: peak = {a: .3f}*recurrence_interval + {b: .3f}"
    )
    fig.supxlabel("Recurrence Interval (Years)")
    fig.supylabel("Peak Flow (CFS)")
    ax.grid(color="grey", linestyle="-", linewidth=0.25)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi="figure")
    # Save binary buffer to png location
    if poi_id_str is None:
        output_plots["distinct_flood_events_gage"] = buf
    else:
        output_plots[f"distinct_flood_events_poi_{poi_id_str}"] = buf
    return (output_csvs, output_plots)


def thresholds_table_output(cda_session):
    """
    Formats the thresholds table into a dataframe to be outputted as a csv for the user.
    Args:
        cda_session - user cda session data (contains necessary information)
    Returns:
        dict containing reference to thresholds output file
    """
    thresholds_data = deepcopy(cda_session["thresholdTableData"])
    for row in thresholds_data:
        if row["poiId"] == -2:
            row["location"] = "Gage"
        elif row["poiId"] == -1:
            row["location"] = "POD"
            row["stream_class"] = cda_session["podStreamClass"]
            if "minimumBypassFlowRegional" in row:
                if row["minimumBypassFlow"] == row["minimumBypassFlowRegional"]:
                    row["minimumBypassFlow"] = None
            if "maximumCumulativeDiversionRegional" in row:
                if (
                    "maximumCumulativeDiversion" in row
                    and row["maximumCumulativeDiversion"]
                    == row["maximumCumulativeDiversionRegional"]
                ):
                    row["maximumCumulativeDiversion"] = None
        else:
            row["location"] = f"POI {row['poiId']}"
            poi = next(
                (p for p in cda_session["pointsOfInterest"] if p["id"] == row["poiId"]),
                None,
            )
            row["stream_class"] = poi["class"]
            row["position_relative_to_ula"] = poi["anadromy"]
            if (
                "minimumBypassFlowRegional" in row
                and row["minimumBypassFlow"] == row["minimumBypassFlowRegional"]
            ):
                row["minimumBypassFlow"] = None
            if (
                "minimumBypassFlowRegional" in row
                and row["maximumCumulativeDiversion"]
                == row["maximumCumulativeDiversionRegional"]
            ):
                row["maximumCumulativeDiversion"] = None
    thresholds_df = pd.DataFrame.from_dict(thresholds_data)
    thresholds_df = thresholds_df.drop(columns="poiId")
    thresholds_df.rename(
        columns={
            "drainageArea": "area_sqmi",
            "averagePrecipitation": "average_annual_precipitation_in",
            "meanAnnualUnimpairedVolumeAf": "mean_annual_flow_volume_af",
            "meanAnnualUnimpairedVolumeCfs": "mean_annual_flow_volume_cfs",
            "minimumBypassFlow": "up_mbf_cfs",
            "minimumBypassFlowRegional": "rc_mbf_cfs",
            "maximumCumulativeDiversion": "up_mcd_cfs",
            "maximumCumulativeDiversionRegional": "rc_mcd_cfs",
            "februaryMedian": "fmf_cfs",
        },
        inplace=True,
    )
    correct_ordered_columns = [
        "location",
        "stream_class",
        "position_relative_to_ula",
        "area_sqmi",
        "average_annual_precipitation_in",
        "mean_annual_flow_volume_af",
        "mean_annual_flow_volume_cfs",
        "ratio",
        "rc_mbf_cfs",
        "up_mbf_cfs",
        "rc_mcd_cfs",
        "up_mcd_cfs",
        "fmf_cfs",
    ]
    for column in correct_ordered_columns:
        if(column not in thresholds_df.columns):
            thresholds_df[column] = ''
    thresholds_df = thresholds_df[correct_ordered_columns]
    return {
        "initial_calculations_and_regional_criteria_or_user_thresholds_mbf_mcd_fmf_B.5.2.1-A4": thresholds_df
    }


def generate_gage_impairment_output(raw_gage_data, unimpaired_gage_data, poi_ts):
    """
    Generates the required gage impairment data for the output package. Backfills yearly data from the raw/unimpaired gage daily data series.
    Args:
        raw_gage_data - raw gage time series
        unimpaired_gage_data - unimpaired gage time series
        poi_ts - dict of poi_id : unimpaired_poi_data
    Returns:
        dict containing reference to a dataframe with the required yearly breakdown output, dict containing byteArray of output plot
    """
    # Need to convert to acre-feet
    raw_gage_df = pd.DataFrame.from_dict(raw_gage_data)
    raw_gage_df["daily_flow"] = raw_gage_df["daily_flow"] / AFD_TO_CFS
    unimpaired_gage_df = pd.DataFrame.from_dict(unimpaired_gage_data)
    unimpaired_gage_df["daily_flow"] = unimpaired_gage_df["daily_flow"] / AFD_TO_CFS
    # Doing 'inner join' sort of operation to ensure that the raw and unimpaired are the same shape
    merged_df = pd.merge(
        raw_gage_df, unimpaired_gage_df, on=["date"], suffixes=["_raw", "_unimpaired"]
    )
    merged_df["date"] = pd.to_datetime(merged_df.date, dayfirst=True)
    merged_df = merged_df.groupby([merged_df.date.dt.year, merged_df.date.dt.month])[
        ["daily_flow_raw", "daily_flow_unimpaired"]
    ].sum()
    poi_dfs = []
    poi_years_data = []
    for id in poi_ts.keys():
        poi_df = pd.DataFrame.from_dict(poi_ts[id])
        poi_df["daily_flow"] = poi_df["daily_flow"] / AFD_TO_CFS
        poi_df["date"] = pd.to_datetime(poi_df.date, dayfirst=True)
        poi_df = poi_df.groupby([poi_df.date.dt.year, poi_df.date.dt.month])[
            "daily_flow"
        ].sum()
        poi_dfs.append(poi_df)
        poi_years_data.append({})
    raw_years_data = {}
    unimpaired_years_data = {}
    for index, row in merged_df.iterrows():
        (year, month) = index
        # Scale year to 'water year'
        if month < 10:
            year = year - 1
        if year in raw_years_data:
            raw_years_data[year] = raw_years_data[year] + row["daily_flow_raw"]
            unimpaired_years_data[year] = (
                unimpaired_years_data[year] + row["daily_flow_unimpaired"]
            )
            for i in range(len(poi_dfs)):
                poi_years_data[i][year] = poi_years_data[i][year] + poi_dfs[i][index]
        else:
            raw_years_data[year] = row["daily_flow_raw"]
            unimpaired_years_data[year] = row["daily_flow_unimpaired"]
            for i in range(len(poi_dfs)):
                poi_years_data[i][year] = poi_dfs[i][index]
    difference_years_data = []
    raw_years_mean_annual_flow_cfs = []

    for year in unimpaired_years_data.keys():
        #Occasionally leap day handling causes unimpaired to be lower than raw, minimize at 0
        difference_years_data.append(max(unimpaired_years_data[year] - raw_years_data[year], 0))
        if year % 4 == 0:
            # Writing leap year handling code on the leap day!
            raw_years_mean_annual_flow_cfs.append(
                unimpaired_years_data[year] * AFD_TO_CFS / 366
            )
        else:
            raw_years_mean_annual_flow_cfs.append(
                unimpaired_years_data[year] * AFD_TO_CFS / 365
            )
    overall_df = pd.DataFrame(index=raw_years_data.keys())
    overall_df["water_year_starting_in"] = raw_years_data.keys()
    overall_df["gage_mean_flow_cfs"] = raw_years_mean_annual_flow_cfs
    overall_df["gage_annual_impaired_volume_af"] = list(raw_years_data.values())
    overall_df["gage_annual_diversion_af"] = difference_years_data
    overall_df["gage_unimpaired_annual_volume_af"] = list(
        unimpaired_years_data.values()
    )
    for id in poi_ts.keys():
        overall_df[f"unimpaired_annual_volume_af_prorated_to_poi_{id}"] = list(
            poi_years_data[0].values()
        )
        poi_years_data = poi_years_data[1:]
    # Make bar chart of the yearly impaired vs unimpaired data
    fig = Figure(figsize=(10, 7), dpi=80)
    width = 0.3
    ax = fig.subplots()
    ax.bar(
        raw_years_data.keys(),
        list(raw_years_data.values()),
        width=width,
        label="impaired gage data",
        color="green",
    )
    ax.bar(
        np.array(list(raw_years_data.keys())) + width,
        list(unimpaired_years_data.values()),
        width=width,
        label="unimpaired gage data",
        color="blue",
    )
    ax.legend()

    fig.suptitle("Gage Impaired vs Unimpaired Yearly Annual Flow Volume")
    fig.supxlabel("Year")
    fig.supylabel("Annual Flow Volume (AF)")
    ax.grid(color="grey", linestyle="-", linewidth=0.25)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi="figure")
    return (
        {"gage_streamflow_mean_annual_flow_rate_B.5.2.1-A4": overall_df},
        {"unimpaired_annual_gage_comparison": buf},
    )


def turn_year_to_current(val):
    """
    Change year to current year given a datetime value of the form 01-11-2018
    """
    if pd.isna(val):
        return
    date = pd.to_datetime(val, dayfirst=True)
    if date.year == 2019:
        date = date.replace(year=datetime.date.today().year)
    if date.year == 2018:
        date = date.replace(year=datetime.date.today().year - 1)
    return date.strftime("%d-%m-%Y")

def generate_senior_diverter_row(row, yearly_diversion_sum, yearly_diversion_sum_no_pod):
    yearly_ts = generate_yearly_ts_from_row(row)
    if "Frost Protection" in row["use_codes"]:
        row.loc["frost"] = "Yes"
    else:
        row.loc["frost"] = "No"
    start_of_water_year = datetime.date(2018, 10, 1)
    for i in range(len(yearly_ts)):
        day_rep = start_of_water_year + datetime.timedelta(days=i)
        row.loc[day_rep.strftime("%m/%d")] = yearly_ts[i]
        if row["analysis_label"] != "Proposed POD":
            yearly_diversion_sum.append(yearly_ts[i])
        yearly_diversion_sum_no_pod.append(yearly_ts[i])
    return row


def generate_poi_senior_diverters_output(poi_id, upstream_senior_diverters_with_pod):
    """
    Generate the senior diverters upstream of the POI's output package data.
    Args:
        poi_id - used for output formatting
        upstream_senior_diverters - upstream diverters
        upstream_senior_diverters_with_pod - diverters with pod
    Returns:
        Dictionary containing dataFrame-formatted csv data
    """
    # Start with generating a yearly diversion for each of the rows
    senior_diverters_df = pd.DataFrame(upstream_senior_diverters_with_pod)
    senior_diverters_df["use_codes"] = senior_diverters_df.apply(
        format_use_codes, axis=1
    )
    senior_diverters_df["frost"] = [None] * len(senior_diverters_df.index)
    senior_diverters_df["notes"] = [""] * len(senior_diverters_df.index)
    # Got performance warning that df is fragmented, clean up with copy operation
    output_df = senior_diverters_df.copy()
    output_df = output_df.drop(
        columns=[
            "appl_pod",
            "comments",
            "latitude",
            "longitude",
            "pod_count",
            "use_codes",
            "source_name",
            "annual_precip_in",
            "water_right_type",
            "wr_water_right_id",
            "drainage_area_sqmi",
            "seasonal_demand_af",
            "water_right_status",
            "wr_seasonal_demand",
            "diversion_amount_af",
            "diversion_per_day_af",
            "frost_demand_af",
            "application_primary_owner",
            "overwrite_seasonal_demand_af_justification",
            "overlapping_days_of_proposed_and_frost_season",
            "overlapping_days_of_storage_and_policy_season",
            "overlapping_days_of_proposed_and_policy_season",
            "overlapping_days_of_storage_and_proposed_season",
            "overlapping_proposed_and_days_of_direct_diversion",
            "overlapping_days_of_direct_diversion_and_policy_season",
        ],
        errors="ignore",
    )
    output_df["direct_div_season_start"] = output_df["direct_div_season_start"].apply(
        lambda x: turn_year_to_current(x)
    )
    output_df["direct_div_season_end"] = output_df["direct_div_season_end"].apply(
        lambda x: turn_year_to_current(x)
    )
    output_df["storage_season_start"] = output_df["storage_season_start"].apply(
        lambda x: turn_year_to_current(x)
    )
    output_df["storage_season_end"] = output_df["storage_season_end"].apply(
        lambda x: turn_year_to_current(x)
    )
    output_df["onstream_storage"] = output_df["pod_type"].apply(
        lambda x: "Yes" if (x == "Point of Onstream Storage") else "No"
    )
    # Put onstream storage in the right spot
    cols = output_df.columns.tolist()
    right_order_output_columns = [
        "application_number",
        "order_upstream_to_downstream",
        "analysis_label",
        "frost",
        "notes",
        "face_amount_af",
        "max_rate_of_diversion_cfs",
        "minimum_bypass_flow_cfs",
        "pod_type",
        "onstream_storage",
        "max_storage_af",
        "direct_div_season_start",
        "direct_div_season_end",
        "days_of_diversion",
        "storage_season_start",
        "storage_season_end",
        "days_of_storage",
    ]
    cols = right_order_output_columns
    output_df = output_df[cols]

    output_df.fillna(0, inplace=True)
    #put proposed POD at the end
    pod_row = output_df[output_df["analysis_label"] == "Proposed POD"].iloc[0]
    output_df = output_df.shift(-1)
    output_df.iloc[-1] = pod_row.squeeze()
    # Hold these in variables so they don't stomp all over each other
    return {f"daily_flow_study_senior_diverters_poi_{poi_id + 1}_B.5.3.2": output_df}

def generate_daily_flow_timeseries(
    poi_id,
    daily_time_seriess,
    unimpaired_gage_data,
    threshold_data,
    proposed_start_date,
    proposed_end_date,
    upstream_senior_diverters,
    onstream_storage_upstream_diverters,
    gage_ratio_raw,
    february=False,
):
    """
    Generate daily flow timeseries for output file daily_flow_study_time_series_poi{}_B.5.3.1-B.5.3.3 and for file february_median_flow_time_series_poi_{}_B.5.3.6
    Args:
        poi_id - used for output formatting
        ts_dict - dict containing poi time-series data
        unimpaired_gage_data - gage unimpaired data
    Returns:
        Dict containing dataframe with output data
    """
    date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    start_date_obj = datetime.datetime.strptime(proposed_start_date, date_format)
    end_date_obj = datetime.datetime.strptime(proposed_end_date, date_format)
    start_date_obj = start_date_obj.replace(year=2000)
    end_date_obj = end_date_obj.replace(year=2000)

    unimpaired_poi_df = pd.DataFrame.from_dict(daily_time_seriess["unimpaired"])
    diverter_impaired_df = pd.DataFrame.from_dict(
        daily_time_seriess["impaired_with_diverters"]
    )
    pod_impaired_df = pd.DataFrame.from_dict(daily_time_seriess["impaired_with_pod"])
    gage_df = pd.DataFrame.from_dict(unimpaired_gage_data)
    if not february:
        mbf = next((p for p in threshold_data if p["poiId"] == poi_id), None)[
            "minimumBypassFlow"
        ]
        name_of_comparison_column = "minimum_bypass_flow_cfs"
        output_filename = f"daily_flow_study_time_series_poi_{poi_id + 1}_B.5.3.1-B.5.3.3"
        suffix = "mbf"
    else:
        mbf = next((p for p in threshold_data if p["poiId"] == poi_id), None)[
            "februaryMedian"
        ]
        output_filename = f"february_median_flow_time_series_poi_{poi_id + 1}_B.5.3.6"
        name_of_comparison_column = "february_median_flow_cfs"
        suffix = "fmf"
    yearly_diversions = generate_senior_diverter_ts_poi(upstream_senior_diverters, unimpaired_gage_data, onstream_storage_upstream_diverters, gage_ratio_raw, output_package=True)

    columns = [
        "date",
        "gage_flow"
    ]
    sum_yearly_diversions_diverters = {}
    for diverter in upstream_senior_diverters:
        # Generate the scaling ratio for each of the diversions so their estimated flow can be calculated
        columns.append(f"{diverter['application_number']}_face_value_af")
        columns.append(f"{diverter['application_number']}_daily_diversion_cfs")
        columns.append(f"{diverter['application_number']}_daily_diversion_af")
        columns.append(f"{diverter['application_number']}_total_yearly_diversion_af")
        sum_yearly_diversions_diverters[diverter['order_upstream_to_downstream']] = 0
    columns.extend([
        "senior_diverters_total_diversion_af",
        "senior_diverters_diversion_rate_cfs",
        "pod_diversion_af",
        "pod_diversion_cfs",
        "poi_flow_unimpaired",
        "poi_flow_unimpaired_cfs",
        "poi_flow_impaired_with_senior_diverters_cfs",
        "poi_flow_impaired_with_senior_diverters_plus_proposed_cfs",
        name_of_comparison_column,
        f"poi_unimpaired_flow_meets_or_exceeds_{suffix}",
        f"poi_impaired_flow_meets_or_exceeds_{suffix}",
        f"poi_impaired_flow_plus_proposed_meets_or_exceeds_{suffix}",
        "in_project_season"
        ])

    output_df = pd.DataFrame(
        columns=columns
    )

    year_index = 0
    index_offset = 0
    # Populate the output columns for each row - only last 20 years
    end_of_gage_year = max(pd.to_datetime(gage_df["date"], dayfirst=True).dt.year)
    start_year = min(pd.to_datetime(gage_df["date"], dayfirst=True).dt.year)
    this_year = datetime.date.today().year
    end_year = end_of_gage_year if end_of_gage_year < this_year else this_year
    for index, day_record in gage_df[
        pd.to_datetime(gage_df["date"], dayfirst=True).dt.year.between(
            start_year, end_year
        )
    ].iterrows():
        date_split = day_record['date'].split('-')
        day = date_split[0]
        month = date_split[1]
        # Skip leap year
        if(int(day) == 29 and int(month) == 2):
            index_offset = index_offset + 1
            continue
        non_february_index = index - index_offset
        year = date_split[2]
        water_year = generate_water_year(day_record['date'])
        diversions = yearly_diversions[int(water_year)]
        output_row = {}
        output_row['date'] = f"{month}-{day}-{year}"
        output_row['gage_flow'] = day_record['daily_flow']
        total_diversion = 0
        for diverter in upstream_senior_diverters:
            output_row[f"{diverter['application_number']}_face_value_af"] = diverter['face_amount_af']
            output_row[f"{diverter['application_number']}_daily_diversion_cfs"] = diversions[diverter['order_upstream_to_downstream']][year_index]
            output_row[f"{diverter['application_number']}_daily_diversion_af"] = diversions[diverter['order_upstream_to_downstream']][year_index] / AFD_TO_CFS
            sum_yearly_diversions_diverters[diverter['order_upstream_to_downstream']] += diversions[diverter['order_upstream_to_downstream']][year_index] / AFD_TO_CFS
            output_row[f"{diverter['application_number']}_total_yearly_diversion_af"] = sum_yearly_diversions_diverters[diverter['order_upstream_to_downstream']]
            total_diversion += diversions[diverter['order_upstream_to_downstream']][year_index] / AFD_TO_CFS
        output_row['senior_diverters_total_diversion_af'] = total_diversion
        output_row['senior_diverters_diversion_rate_cfs'] = total_diversion * AFD_TO_CFS
        output_row['pod_diversion_af'] = (diverter_impaired_df.loc[non_february_index]['daily_flow'] - pod_impaired_df.loc[non_february_index]['daily_flow']) / AFD_TO_CFS
        output_row['pod_diversion_cfs'] = output_row['pod_diversion_af'] * AFD_TO_CFS
        output_row['poi_flow_unimpaired'] = unimpaired_poi_df.loc[non_february_index]['daily_flow'] / AFD_TO_CFS
        output_row['poi_flow_unimpaired_cfs'] = unimpaired_poi_df.loc[non_february_index]['daily_flow']
        output_row['poi_flow_impaired_with_senior_diverters_cfs'] = diverter_impaired_df.loc[non_february_index]['daily_flow']
        output_row['poi_flow_impaired_with_senior_diverters_plus_proposed_cfs'] = pod_impaired_df.loc[non_february_index]['daily_flow']
        output_row[name_of_comparison_column] = mbf
        output_row[f"poi_unimpaired_flow_meets_or_exceeds_{suffix}"] = (
            1 if output_row["poi_flow_unimpaired_cfs"] > mbf else 0
        )
        output_row[f"poi_impaired_flow_meets_or_exceeds_{suffix}"] = (
            1 if output_row["poi_flow_impaired_with_senior_diverters_cfs"] > mbf else 0
        )
        output_row[f"poi_impaired_flow_plus_proposed_meets_or_exceeds_{suffix}"] = (
            1
            if output_row["poi_flow_impaired_with_senior_diverters_plus_proposed_cfs"]
            > mbf
            else 0
        )
        row_date = pd.to_datetime(day_record["date"], dayfirst=True).replace(year = 2000)

        if start_date_obj <= end_date_obj:
            # normal season in 1 year
            within_season =  start_date_obj <= row_date <= end_date_obj
        else:
            # season spans the 1st of jan ie the common dec 15 - mar 31
            within_season = row_date >= start_date_obj or row_date <= end_date_obj

        output_row["in_project_season"] = ( 1 if within_season else 0)
        output_df = pd.concat(
            [output_df, pd.DataFrame.from_dict([output_row])], ignore_index=True
        )
        year_index = (year_index + 1) % 365
        if(year_index == 0):
            # Reset the yearly diversions sums
            for diverter in sum_yearly_diversions_diverters.keys():
                sum_yearly_diversions_diverters[diverter] = 0

    return {output_filename: output_df}


def generate_yearly_flow_output_csv(poi_id, daily_time_seriess):
    """
    Generate data for the yearly flows, recurrence intervals for the output package.
    Args:
        poi_id - Id of POI
        daily_time_seriess - unimpaired and impaired daily poi time series
    Returns:
        dict containing reference to filename / data
    """
    unimpaired_poi_df = pd.DataFrame.from_dict(daily_time_seriess["unimpaired"]).fillna(
        method="ffill"
    )
    diverter_impaired_df = pd.DataFrame.from_dict(
        daily_time_seriess["impaired_with_diverters"]
    ).fillna(method="ffill")
    pod_impaired_df = pd.DataFrame.from_dict(
        daily_time_seriess["impaired_with_pod"]
    ).fillna(method="ffill")

    (yearly_peaks_unimpaired, years_data) = calculate_yearly_peaks(
        unimpaired_poi_df, output_package=True
    )
    yearly_peaks_unimpaired.sort(reverse=True)
    yearly_peaks_impaired = calculate_yearly_peaks(diverter_impaired_df)
    yearly_peaks_impaired.sort(reverse=True)
    yearly_peaks_pod = calculate_yearly_peaks(pod_impaired_df)
    yearly_peaks_pod.sort(reverse=True)
    # Same recurrence intervals for each
    recurrence_intervals = generate_recurrence_intervals(
        yearly_peaks_unimpaired, len(yearly_peaks_unimpaired)
    )
    first_under = False
    output_df = pd.DataFrame(
        columns=[
            "water_year",
            "daily_peak_flow_unimpaired",
            "daily_peak_flow_impaired_with_senior_diverters",
            "daily_peak_flow_impaired_with_project",
            "rank",
            "return_period",
        ]
    )
    for i in range(len(yearly_peaks_unimpaired)):
        output_row = {}
        output_row["water_year"] = next(
            (
                year
                for year in years_data.keys()
                if (
                    years_data[year] == yearly_peaks_unimpaired[i]
                    and year not in output_df["water_year"].tolist()
                )
            ),
            None,
        )
        output_row["daily_peak_flow_unimpaired"] = yearly_peaks_unimpaired[i]
        output_row[
            "daily_peak_flow_impaired_with_senior_diverters"
        ] = yearly_peaks_impaired[i]
        output_row["daily_peak_flow_impaired_with_project"] = yearly_peaks_pod[i]
        output_row["rank"] = i + 1
        if recurrence_intervals[i] <= 1.5 and first_under is False:
            output_row["return_period"] = f"{recurrence_intervals[i]}***"
            first_under = True
        else:
            output_row["return_period"] = recurrence_intervals[i]
        output_df = pd.concat(
            [output_df, pd.DataFrame.from_dict([output_row])], ignore_index=True
        )
    return {f"daily_flow_study_time_series_B.5.3.5_poi_{poi_id + 1}": output_df}


def generate_daily_flow_summary_csvs(cda_session):
    """
    Generate the daily flow summary output data from the data stored in the cda session.
    Args:
        cda_session
    Returns:
        dict of daily flow summary output files for POI's
    """
    if "dailyFlowData" not in cda_session:
        raise Exception(
            "Daily flow study data not found in cda session - required for output package!"
        )
    daily_flow_data = cda_session["dailyFlowData"]
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    output_csvs = {}
    for poi_dfs in daily_flow_data:
        poi_id = poi_dfs["poiId"]
        # Generate Output file for spawning, rearing passage daily flow study
        spawning_passage = poi_dfs["spawningPassage"]
        unimpaired_dates = {}
        unimpaired_dates[
            "metric"
        ] = "Number of Days Unimpaired Flow Meets or Exceeds MBF"
        impaired_dates = {}
        impaired_dates[
            "metric"
        ] = "Number of Days Impaired with Senior Diverters Meets or Exceeds MBF"
        pod_dates = {}
        pod_dates[
            "metric"
        ] = "Number of Days Impaired with Senior Diverters and Project Meets or Exceeds MBF"
        impaired_pct = {}
        impaired_pct[
            "metric"
        ] = "Percent change in the number of days flow is above MBF (unimpaired and impaired without project)"
        pod_pct = {}
        pod_pct[
            "metric"
        ] = "Percent change in the number of days flow is above MBF (unimpaired and impaired with project)"
        for index, offset_month in enumerate(spawning_passage['months']):
            unimpaired_dates[months[offset_month]] = spawning_passage["unimpairedExceedances"][index]
            impaired_dates[months[offset_month]] = spawning_passage["diverterImpairedExceedances"][
                index
            ]
            pod_dates[months[offset_month]] = spawning_passage["podImpairedExceedances"][index]
            impaired_pct[months[offset_month]] = spawning_passage["percentageImpaired"][index][
                "percentage"
            ]
            pod_pct[months[offset_month]] = spawning_passage["percentagePod"][index]["percentage"]
        spawning_passage_df = pd.DataFrame.from_dict(
            [unimpaired_dates, impaired_dates, pod_dates, impaired_pct, pod_pct]
        )
        output_csvs[
            f"daily_flow_study_summary_B.5.3.4_poi_{poi_id + 1}"
        ] = spawning_passage_df

        # Generate output file for natural flow variability
        natural_flow_variability = poi_dfs["naturalFlowVariability"]
        natural_flow_row = {}
        natural_flow_row["location"] = poi_id
        natural_flow_row[
            "1.5 Year Daily Peak Flow Unimpaired"
        ] = natural_flow_variability["instantaneousPeakUnimpaired"]
        natural_flow_row[
            "1.5 Year Daily Peak Flow Impaired with Senior Diverters"
        ] = natural_flow_variability["instantaneousPeakDiverters"]
        natural_flow_row[
            "1.5 Year Daily Peak Flow Impaired with Senior Diverters and Project"
        ] = natural_flow_variability["instantaneousPeakPod"]
        natural_flow_row["Calculation B.5.3.5-2a 1"] = f"""{natural_flow_variability[
            "ratioUnimpairedDiverters"
        ]} %"""
        natural_flow_row["Calculation B.5.3.5-2b 1"] = f"""{natural_flow_variability[
            "ratioUnimpairedPod"
        ]} %"""
        natural_flow_row["Change in instream flow"] = f"""{natural_flow_variability[
            "ratioDifference"
        ]} %"""
        natural_flow_df = pd.DataFrame.from_dict([natural_flow_row])
        output_csvs[f"daily_flow_study_summary_B.5.3.5_poi_{poi_id + 1}"] = natural_flow_df
        if "februaryMedian" in poi_dfs:
            # Generate output file for february median data
            feb_median = poi_dfs["februaryMedian"]
            unimpaired_dates = {}
            unimpaired_dates[
                "metric"
            ] = "Number of Days Unimpaired Flow Meets or Exceeds FMF"
            impaired_dates = {}
            impaired_dates[
                "metric"
            ] = "Number of Days Impaired with Senior Diverters Meets or Exceeds FMF"
            pod_dates = {}
            pod_dates[
                "metric"
            ] = "Number of Days Impaired with Senior Diverters and Project Meets or Exceeds FMF"
            impaired_pct = {}
            impaired_pct[
                "metric"
            ] = "Percent change in the number of days flow is above FMF (unimpaired and impaired without project)"
            pod_pct = {}
            pod_pct[
                "metric"
            ] = "Percent change in the number of days flow is above FMF (unimpaired and impaired with project)"
            for index, offset_month in enumerate(feb_median['months']):
                unimpaired_dates[months[offset_month]] = feb_median["unimpairedExceedances"][index]
                impaired_dates[months[offset_month]] = feb_median["diverterImpairedExceedances"][index]
                pod_dates[months[offset_month]] = feb_median["podImpairedExceedances"][index]
                impaired_pct[months[offset_month]] = feb_median["percentageImpaired"][index][
                    "percentage"
                ]
                pod_pct[months[offset_month]] = feb_median["percentagePod"][index]["percentage"]
            feb_median_df = pd.DataFrame.from_dict(
                [unimpaired_dates, impaired_dates, pod_dates, impaired_pct, pod_pct]
            )
            output_csvs[
                f"daily_flow_study_summary_B.5.3.6_poi_{poi_id + 1}"
            ] = feb_median_df
    return output_csvs

def peaks_threshold_output_execution(time_series, poi_id_str, output_package_csvs, output_package_plots):
    peaks_thresholds = peaks_and_threshold(time_series, output_package=True)
    recurrence_intervals = generate_recurrence_intervals(peaks_thresholds['peaks'], peaks_thresholds['record_years'])
    instantaneous_flow = plot_and_find_instantaneous_peak_flow(peaks_thresholds['peaks'], recurrence_intervals)
    (peaks_threshold_output_csvs, peaks_threshold_output_plots) = peaks_over_threshold_output(peaks_thresholds, recurrence_intervals, instantaneous_flow, time_series, poi_id_str=poi_id_str)
    for key in peaks_threshold_output_csvs.keys():
        output_package_csvs[key] = peaks_threshold_output_csvs[key]
    for key in peaks_threshold_output_plots.keys():
        output_package_plots[key] = peaks_threshold_output_plots[key]

def email_output_package_zip(email=None, attachment=None, file_name=None, project_id=None,
                             poi_id=None):
    # Ensure email isn't sent on unit test runs
    if(email == 'test'):
        return
    sendgrid_apikey = os.environ["SENDGRID_API_KEY"]
    if(poi_id is None):
        sbj = """Your Cumulative Diversion Analysis Output Package is ready!"""
    else:
        sbj = f"""Your Cumulative Diversion Analysis Output Package for {poi_id} is ready!"""
    if(poi_id is None):
        msg = f"""The cumulative diversion analysis output package for project {project_id} has been processed.\n
            Please review and visit northcoastwater.codefornature.org to change!"""
    else:
        msg = f"""The cumulative diversion analysis output package for project {project_id}, {poi_id} has been processed.\n
            Please review and visit northcoastwater.codefornature.org to change!"""
    payload = {
        "to": email,
        "toname": "Water Applicant",
        "cc[]": "",
        "text": msg,
        "subject": sbj,
        "html": msg.replace("\n", "<br>\n"),
        "from": "california_watertool@foundryspatial.com",
        "fromname": "California North Coast Water Availability Tool",
        "files[%s]" % file_name: attachment.read(),
    }
    r = requests.post(
        "https://api.sendgrid.com/api/mail.send.json",
        data=payload,
        headers={"Authorization": "Bearer {}".format(sendgrid_apikey)},
    )
    retry_limit = 3
    retry_wait = 5
    while(r.status_code != 200 and retry_limit > 0):
        retry_limit = retry_limit - 1
        time.sleep(retry_wait)
        r = requests.post(
            "https://api.sendgrid.com/api/mail.send.json",
            data=payload,
            headers={"Authorization": "Bearer {}".format(sendgrid_apikey)},
        )
    if(r.status_code != 200):
        print(f"ERROR - Unable to send email for project {project_id} after retries")

def email_error(email=None, id=None, error_message=None):
    """
    Inform the user if their output package generation has failed.
    """
    if(os.getenv("ENVIRONMENT")  == 'local'):
        print(error_message)
        return
    sendgrid_apikey = os.environ["SENDGRID_API_KEY"]
    sbj = f"""Failed Generating output package for project {id}"""
    msg = f"""Failed to generate the cumulative diversion analysis output package for project {id}.\n
            Please review and visit northcoastwater.codefornature.org to retry!\n
            Error message:\n{error_message}"""
    payload = {
        "to": email,
        "toname": "Water Applicant",
        "cc[]": "",
        "text": msg,
        "subject": sbj,
        "html": msg.replace("\n", "<br>\n"),
        "from": "california_watertool@foundryspatial.com",
        "fromname": "California North Coast Water Availability Tool"
    }
    r = requests.post(
        "https://api.sendgrid.com/api/mail.send.json",
        data=payload,
        headers={"Authorization": "Bearer {}".format(sendgrid_apikey)},
    )
    retry_limit = 3
    retry_wait = 5
    while(r.status_code != 200 and retry_limit > 0):
        retry_limit = retry_limit - 1
        time.sleep(retry_wait)
        r = requests.post(
            "https://api.sendgrid.com/api/mail.send.json",
            data=payload,
            headers={"Authorization": "Bearer {}".format(sendgrid_apikey)},
        )
    if(r.status_code != 200):
        raise Exception(f"ERROR - Unable to send email for project {id} after retries")


def generate_cda_output_package(
    gage_csvs,
    raw_gage_timeseries,
    unimpaired_gage_data,
    poi_unimpaired,
    poi_time_seriess,
    cda_session,
    wsr_senior_diverters,
    diverters_upstream_of_onstream_storage,
    gage_ratio_raw,
    email,
    id
):
    """
    Generate the CDA output package, called as an asynchronous function.
    Emails the resulting output package to the user and stores a link in their session.
    Args:
        gage_csvs - data from gage_senior_diverter_csv database table
        raw_gage_timeseries - time-series of gage readings
        unimpaired_gage_data - time-series of gage readings unimpaired with upstream diversions
        poi_unimpaired - time-series of poi data scaled from unimpaired_gage_data
        poi_time_seriess - list of poi time series including impairments
        cda_session - user cda session data
        wsr_senior_diverters - senior diverters used to impair POIs
        email - user email
    """
    output_package_csvs= {}
    output_package_plots = {}
    try:
        gage_output_files = format_output_gage_data(gage_csvs)
        #Merge dicts
        output_package_csvs = output_package_csvs | gage_output_files
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nUnable to create output package gage csvs.')
        return
    try:
        if(raw_gage_timeseries == None):
            raise Exception(f"Unable to find gage time series data for selected gage.")
        if(unimpaired_gage_data == None):
            raise Exception("Must have uploaded gage diverters for output package formatting.")
        (impairment_csvs, impairment_plot) = generate_gage_impairment_output(raw_gage_timeseries, unimpaired_gage_data, poi_unimpaired)
        output_package_csvs = output_package_csvs | impairment_csvs
        output_package_plots = output_package_plots | impairment_plot
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nUnable to generate gage impaired/unimpaired data')
        return
    try:
        threads = []
        peaks_threshold_output_execution(unimpaired_gage_data, None, output_package_csvs, output_package_plots)
        for poi_id in poi_time_seriess.keys():
            peaks_threshold_output_execution(poi_time_seriess[poi_id]['unimpaired'], f'{poi_id + 1}_unimpaired', output_package_csvs, output_package_plots)
            peaks_threshold_output_execution(poi_time_seriess[poi_id]['impaired_with_diverters'], f'{poi_id + 1}_impaired_with_diverters', output_package_csvs, output_package_plots)
            peaks_threshold_output_execution(poi_time_seriess[poi_id]['impaired_with_pod'], f'{poi_id + 1}_impaired_with_pod', output_package_csvs, output_package_plots)
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nUnable to generate peaks over threshold output data.')
        return
    try:
        if(not 'thresholdTableData' in cda_session or cda_session['thresholdTableData'] == None):
            raise Exception("Output package requires calculated thresholds table data!")
        threshold_output_csvs = thresholds_table_output(cda_session)
        output_package_csvs = output_package_csvs | threshold_output_csvs
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nUnable to make threshold table data.')
        return
    try:
        for index, poi_id in enumerate(poi_time_seriess):
            if(wsr_senior_diverters == {}):
                raise Exception("No wsr senior diverters found - is the wsr section complete?")
            poi = next((p for p in cda_session['pointsOfInterest'] if p["id"] == poi_id), None)
            (upstream_senior_diverters, upstream_senior_diverters_with_pod) = get_senior_diverters_upstream_of_poi(wsr_senior_diverters, poi, cda_session)
            poi_senior_diverter_output_csv = generate_poi_senior_diverters_output(poi_id, upstream_senior_diverters_with_pod)
            output_package_csvs = output_package_csvs | poi_senior_diverter_output_csv
            daily_flow_study_timeseries = generate_daily_flow_timeseries(poi_id, poi_time_seriess[poi_id], unimpaired_gage_data, cda_session['thresholdTableData'], cda_session['seasonOfDiversionStart'], cda_session['seasonOfDiversionEnd'], upstream_senior_diverters, diverters_upstream_of_onstream_storage[poi_id], gage_ratio_raw)
            output_package_csvs = output_package_csvs | daily_flow_study_timeseries
            if("februaryMedian" in cda_session['thresholdTableData']):
                february_median_timeseries = generate_daily_flow_timeseries(poi_id, poi_time_seriess[poi_id], unimpaired_gage_data, cda_session['thresholdTableData'], cda_session['seasonOfDiversionStart'], cda_session['seasonOfDiversionEnd'],upstream_senior_diverters, diverters_upstream_of_onstream_storage[poi_id], gage_ratio_raw, february=True)
                output_package_csvs = output_package_csvs | february_median_timeseries
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nUnable to generate time-series csvs.')
        raise
    try:
        daily_flow_summary_csvs = generate_daily_flow_summary_csvs(cda_session)
        output_package_csvs = output_package_csvs | daily_flow_summary_csvs
        file_like = BytesIO()
        with zipfile.ZipFile(file_like, mode='w') as zipFileByteObject:
            #Put CSVs in zip
            for file_name in output_package_csvs.keys():
                prepared_csv = output_package_csvs[file_name].to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
                zipFileByteObject.writestr(f'{file_name}.csv', prepared_csv)
            #Put PNGs in zip
            for file_name in output_package_plots.keys():
                zipFileByteObject.writestr(f'{file_name}.png', output_package_plots[file_name].getvalue())

            #Add output package documentation to zip
            with open('zipfiles/Cumulative-Diversion-Analysis-Output-Package.pdf', 'rb') as file:
                zipFileByteObject.writestr('Cumulative-Diversion-Analysis-Output-Package.pdf', file.read())

            #Add gage senior diverters help to zip
            with open('zipfiles/CDA-Gage-Senior-Diverters-Help.pdf', 'rb') as file:
                zipFileByteObject.writestr('CDA Gage Senior Diverters Help.pdf', file.read())

        #Risk of exceeding data sendgrid size of 30 MB, zip separately
        if(int(file_like.__sizeof__()) > 25000000):
            #Do per POI
            already_output = []
            output_package_plot_names = list(output_package_plots.keys())
            for poi_id in poi_time_seriess.keys():
                file_like = BytesIO()
                with zipfile.ZipFile(file_like, mode='w') as zipFileByteObject:
                    #Put CSVs in zip
                    for file_name in output_package_csvs.keys():
                        if(f'poi_{poi_id + 1}' in file_name):
                            prepared_csv = output_package_csvs[file_name].to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
                            zipFileByteObject.writestr(f'{file_name}.csv', prepared_csv)
                            already_output.append(file_name)
                    #Put PNGs in zip
                    for file_name in output_package_plots.keys():
                        if(f'poi_{poi_id + 1}' in file_name):
                            zipFileByteObject.writestr(f'{file_name}.png', output_package_plots[file_name].getvalue())
                            already_output.append(file_name)
                file_like.seek(0)
                email_output_package_zip(email, file_like, file_name=f'CDA_Output_Package_poi_{poi_id + 1}.zip', project_id=id, poi_id=f"POI {poi_id + 1}")
                del file_like
            #Clean up leftover files
            file_like = BytesIO()
            with zipfile.ZipFile(file_like, mode='w') as zipFileByteObject:
                #Put CSVs in zip
                for file_name in output_package_csvs.keys():
                    if(not file_name in already_output):
                        prepared_csv = output_package_csvs[file_name].to_csv(quotechar='"', quoting=csv.QUOTE_NONNUMERIC, index=False)
                        zipFileByteObject.writestr(f'{file_name}.csv', prepared_csv)
                #Put PNGs in zips
                for file_name in output_package_plot_names:
                    if(not file_name in already_output):
                        zipFileByteObject.writestr(f'{file_name}.png', output_package_plots[file_name].getvalue())
                #Add documentation to gage output
                with open('zipfiles/Cumulative-Diversion-Analysis-Output-Package.pdf', 'rb') as file:
                    zipFileByteObject.writestr('Cumulative-Diversion-Analysis-Output-Package.pdf', file.read())

                #Add gage senior diverters help to zip
                with open('zipfiles/CDA-Gage-Senior-Diverters-Help.pdf', 'rb') as file:
                    zipFileByteObject.writestr('CDA Gage Senior Diverters Help.pdf', file.read())
            file_like.seek(0)
            email_output_package_zip(email, file_like, file_name=f'CDA_Output_Package_gage.zip', project_id=id, poi_id="Gage")
            del file_like
        else:
            file_like.seek(0)
            email_output_package_zip(email, file_like, file_name='CDA_Output_Package.zip', project_id=id)
    except Exception as e:
        email_error(email=email, id=id, error_message=f'{str(e)}\nFailed to generate and send emails.')
        return
