import os
import pandas as pd
import numpy as np


def setpaths():
    exploratory_path = os.path.split(os.path.dirname(__file__))[0]
    hsptl_path = os.path.join(exploratory_path, os.path.join('./data', "michigan_missouri_hospitalization.csv"))
    cases_path = os.path.join(exploratory_path, os.path.join('./data', "cases.csv"))
    vac_path = os.path.join(exploratory_path, os.path.join('./data', "vaccinations.csv"))
    return hsptl_path, cases_path, vac_path


def readCSV(hsptl_path, cases_path, vac_path):
    hsptl_path = pd.read_csv(hsptl_path)
    cases_path = pd.read_csv(cases_path)
    vac_path = pd.read_csv(vac_path)
    return hsptl_path, cases_path, vac_path


def getRollingMean(hsptl_df):
    #aqi_df['roll_mean'] = aqi_df['Currently_hospitalized'].rolling(7).mean()
    hsptl_df = hsptl_df.fillna(0)
    hsptl_df.Date = pd.to_datetime(hsptl_df.Date)
    return hsptl_df


def processConfirmedCases(cases_df):
    MI = cases_df[cases_df["State"]=="MI"]
    MO = cases_df[cases_df["State"] == "MO"]
    frames = [MI,MO]
    cases_df = pd.concat(frames)
    return cases_df


def processDeaths(vac_df):
    MI = vac_df[vac_df["State"] == "MI"]
    MO = vac_df[vac_df["State"] == "MO"]
    frames = [MI, MO]
    vac_df = pd.concat(frames)
    return vac_df


def preprocess(hsptl_df, cases_df, vac_df):
    hsptl_df = getRollingMean(hsptl_df)
    cases_df = processConfirmedCases(cases_df)
    vac_df = processDeaths(vac_df)
    return hsptl_df, cases_df, vac_df


def pearson_corr_coeff(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)

    diff_sq_X = np.sum(np.square(X - x_bar))
    diff_sq_Y = np.sum(np.square(Y - y_bar))

    num = np.sum((X - x_bar) * (Y - y_bar))

    den = (np.sqrt((diff_sq_X) * (diff_sq_Y)))
    coeff = num / den

    return coeff


def chi_squared_test(var_a, var_b):
    n = len(var_a)

    a_thresh = np.median(var_a)
    b_thresh = np.median(var_b)

    var_a_less = 0
    var_b_less = 0

    for i in var_a:
        if i < a_thresh:
            var_a_less += 1

    for i in var_b:
        if i < b_thresh:
            var_b_less += 1

    var_a_more = n - var_a_less
    var_b_more = n - var_b_less

    ## calculating table
    a = b = c = d = 0
    for i in range(n):
        if var_b[i] < b_thresh:
            if var_a[i] < a_thresh:
                a += 1
            else:
                b += 1
        else:
            if var_a[i] < a_thresh:
                c += 1
            else:
                d += 1

    total_observations = n
    expected_a = var_a_less * var_b_less / total_observations
    expected_b = var_a_more * var_b_less / total_observations
    expected_c = var_a_less * var_b_more / total_observations
    expected_d = var_a_more * var_b_more / total_observations

    Q_obs = (((expected_a - a) ** 2) / expected_a) + (((expected_b - b) ** 2) / expected_b) + (
            ((expected_c - c) ** 2) / expected_c) + (((expected_d - d) ** 2) / expected_d)
    return Q_obs


def generateHeader(num, start, end):
    print("==============================================================================================")
    print("Inference {}:\n".format(num))
    print("During time interval: {start} - {end}\n".format(start=start, end=end))


def generateParams_inference1(hsptl_df_filter, cases_df_filter):
    x_inf_1 = hsptl_df_filter["Currently_hospitalized"].astype("float").to_numpy().flatten()
    y_inf_1 = cases_df_filter["tot_cases"].to_numpy().flatten()
    return x_inf_1, y_inf_1

def generateParams_inference3(hsptl_df_filter, vac_df_filter):
    x_inf_1 = hsptl_df_filter["currently_on_ventilator"].astype("float").to_numpy().flatten()
    y_inf_1 = vac_df_filter["Distributed"].to_numpy().flatten()
    return x_inf_1, y_inf_1


def PearsonTest(inf_num, x, y):
    pearson_coeff = pearson_corr_coeff(x, y)
    generatePearsonTestResults(inf_num,pearson_coeff)


def ChiSquaredTest(inf_num, x, y):
    Q_obs_inf_1 = chi_squared_test(x, y)
    generateChiSquaredResults(inf_num,Q_obs_inf_1)


def generateChiSquaredResults(inf_num, param):
    print("------------------------------------")
    print("CHI-Square Test")
    print("------------------------------------\n")

    if inf_num == 1:
        print("H0: Number of cases in Michigan and Missouri are INDEPENDENT of the Number of hospitalizations")
        print("H1: Number of cases in Michigan and Missouri are DEPENDENT of the Number of hospitalizations\n")

        print("Q-Observed using Chi-Squared test: {}\n".format(param))
        print("DoF: 1 \np-value (Pr(CHI_sq_dof > {}) from chi_square table: 0.00001\n".format(param))

        print("Since, p-value < 0.05 so H0 will be REJECTED!!")
        print("Thus, the number of cases in Michigan and Missouri are DEPENDENT on the Number of Hospitalizations during 2020-04-10 to 2020-06-02")

        print("------------------------------------\n")

        print(
            "Above two tests shows the number of cases in Michigan and Missouri and the Number of hospitalizations in the states are DEPENDENT and POSITIVELY CORRELATED complementing the observation that when COVID-19 initially broke out, there weren't any medicines available in the market. Thus, it required infected people with symptoms to be hospitalized during 2020-04-10 - 2020-06-02!!")
        print("==============================================================================================")

    elif inf_num == 2:
        print("H0: Number of vaccines distributed in Michigan and Missouri are INDEPENDENT with Number of hospitalizations")
        print("H1: Number of vaccines distributed in Michigan and Missouri are DEPENDENT with Number of hospitalizations\n")

        print("Q-Observed using Chi-Squared test: {}\n".format(param))
        print("DoF: 1 \np-value (Pr(CHI_sq_dof > {}) from chi_square table: 0.00001\n".format(param))

        print("Since, p-value < 0.05 so H0 will be ACCEPTED!!")
        print("Thus, Number of vaccines distributed in Michigan and Missouri are DEPENDENT with Number of hospitalizations during 2020-12-14 to 2021-03-7")

        print("------------------------------------\n")

        print(
            "Above two tests shows Number of vaccines distributed in Michigan and Missouri and Number of hospitalizations are DEPENDENT and NEGATIVELY CORRELATED. This re-inforces the observation that as vaccine distributions increased, the number of hospitalizations decreased. This gives us an overall indication of the vaccine's efficacy on COVID-19 and suggests that the vaccines had a positive impact in the fight against COVID-19.")

        print("==============================================================================================")

    elif inf_num == 3:
        print("H0: Number of vaccines distributed in Michigan and Missouri are INDEPENDENT of Number of Patients on Ventilator")
        print("H1: Number of vaccines distributed in Michigan and Missouri are DEPENDENT of Number of Patients on Ventilator\n")

        print("Q-Observed using Chi-Squared test: {}\n".format(param))
        print("DoF: 1 \np-value (Pr(CHI_sq_dof > {}) from chi_square table: 0.01\n".format(param))

        print("Since, p-value < 0.05 so H0 will be REJECTED!!")
        print("Thus, Number of vaccines distributed in Michigan and Missouri are DEPENDENT of Number of Patients on Ventilator during 2020-12-14 to 2021-03-7")

        print("------------------------------------\n")

        print(
            "Above two tests shows Number of vaccines distributed in Michigan and Missouri and Number of Patients on Ventilator are DEPENDENT and NEGATIVE LINEARLY CORRELATED during 2020-10-18 to 2020-11-16 complementing the efficacy results of vaccines. The results clearly shows the efficacy of the vaccines are good and thus, number of patients required to be put on ventilators has reduced after vaccine distribution began.")
        print("==============================================================================================")


def generatePearsonTestResults(inf_num, coeff):
    print("------------------------------------")
    print("Pearson's Correlation Test")
    print("------------------------------------\n")
    if inf_num == 1:

        print("H0: Number of cases in Michigan and Missouri NOT LINEARLY CORRELATED with number of hospitalizations")
        print("H1: Number of cases in Michigan and Missouri LINEARLY CORRELATED with number of hospitalizations\n")

        print("Pearson's Correlation Coefficient: {}\n".format(coeff))

        print("Since, |Pearson's Correlation Cofficient| > 0.5, H0 is REJECTED!!")
        print("Thus, Number of cases in Michigan and Missouri are POSITIVE LINEARLY CORRELATED with Number of hospitalizations\n")

    elif inf_num == 2:

        print("H0: Number of vaccines distributed in Michigan and Missouri are NOT LINEARLY CORRELATED with Number of hospitalizations")
        print("H1: Number of vaccines distributed in Michigan and Missouri are LINEARLY CORRELATED with Number of hospitalizations\n")

        print("Pearson's Correlation Coefficient: {}\n".format(coeff))

        print("Since, |Pearson's Correlation Cofficient| > 0.5, H0 is REJECTED!!")
        print("Thus, Number of vaccines distributed in Michigan and Missouri are NEGATIVE LINEARLY CORRELATED with Number of hospitalizations\n")

    elif inf_num == 3:
        print("H0: Number of vaccines distributed in Michigan and Missouri are NOT LINEARLY CORRELATED with Number of Patients on Ventilator")
        print("H1: Number of vaccines distributed in Michigan and Missouri are LINEARLY CORRELATED with Number of Patients on Ventilator\n")

        print("Pearson's Correlation Coefficient: {}\n".format(coeff))

        print("Since, |Pearson's Correlation Cofficient| > 0.5, H0 is REJECTED!!")
        print("Thus, Number of vaccines distributed in Michigan and Missouri are NEGATIVE LINEARLY CORRELATED with Number of Patients on Ventilator\n")


def inference1(start, end, hsptl_df, cases_df, vac_df):
    cases_df_filter, vac_df_filter, hsptl_df_filter = getFilters(start, end, hsptl_df, cases_df, vac_df)
    generateHeader("1", start, end)
    x_inf_1, y_inf_1 = generateParams_inference1(hsptl_df_filter, cases_df_filter)
    PearsonTest(1, x_inf_1, y_inf_1)
    ChiSquaredTest(1, x_inf_1, y_inf_1)


def generateParams_inference2(hsptl_df_filter, cases_df_filter):
    x_inf_2 = hsptl_df_filter["Currently_hospitalized"].astype("float").to_numpy().flatten()
    y_inf_2 = cases_df_filter["Distributed"].to_numpy().flatten()
    return x_inf_2, y_inf_2


def inference2(start, end, hsptl_df, cases_df, vac_df):
    cases_df_filter, vac_df_filter, hsptl_df_filter = getFilters(start, end, hsptl_df, cases_df, vac_df)
    generateHeader("2", start, end)
    x_inf_2, y_inf_2 = generateParams_inference2(hsptl_df_filter, vac_df_filter)
    PearsonTest(2, x_inf_2, y_inf_2)
    ChiSquaredTest(2, x_inf_2, y_inf_2)


def inference3(start, end, hsptl_df, cases_df, vac_df):
    cases_df_filter, vac_df_filter, hsptl_df_filter = getFilters(start, end, hsptl_df, cases_df, vac_df)
    generateHeader("3", start, end)
    x_inf_3, y_inf_3 = generateParams_inference3(hsptl_df_filter, vac_df_filter)
    PearsonTest(3, x_inf_3, y_inf_3)
    ChiSquaredTest(3, x_inf_3, y_inf_3)


def generateInferences(hsptl_df, cases_df, vac_df):
    inference1("2020-04-10", "2020-06-02", hsptl_df, cases_df, vac_df)
    inference2("2020-12-14", "2021-03-7", hsptl_df, cases_df, vac_df)
    inference3("2020-12-14", "2021-03-7", hsptl_df, cases_df, vac_df)


def getFilters(start, end, hsptl_df, cases_df, vac_df):
    start_date = pd.to_datetime(start)
    #import pdb ; pdb.set_trace()
    end_date = pd.to_datetime(end)

    cases_df_filter = cases_df.loc[
        (pd.to_datetime(cases_df["Date"]) >= start_date) & (pd.to_datetime(cases_df["Date"]) <= end_date)]
    vac_df_filter = vac_df.loc[
        (pd.to_datetime(vac_df["Date"]) >= start_date) & (pd.to_datetime(vac_df["Date"]) <= end_date)]
    hsptl_df_filter = hsptl_df.loc[
        (pd.to_datetime(hsptl_df["Date"]) >= start_date) & (pd.to_datetime(hsptl_df["Date"]) <= end_date)]
    return cases_df_filter, vac_df_filter, hsptl_df_filter


def main():
    hsptl_path, cases_path, vac_path = setpaths()
    hsptl_df, cases_df, vac_df = readCSV(hsptl_path, cases_path, vac_path)
    hsptl_df, cases_df, vac_df = preprocess(hsptl_df, cases_df, vac_df)
    generateInferences(hsptl_df, cases_df, vac_df)


if __name__ == "__main__":
    main()