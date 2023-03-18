import os
import sys

import File_path as fp
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from data_loader import TestDatasets
from morp_plots import MorpPlots as mp
from pred_and_resp_graphs import PlotGraph as pg
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

path = fp.GLOBAL_PATH
isExist = os.path.exists(path)

if not isExist:
    os.mkdir(fp.GLOBAL_PATH)

path_res = fp.GLOBAL_PATH_REG
isExist = os.path.exists(fp.GLOBAL_PATH_REG)

if not isExist:
    os.mkdir(fp.GLOBAL_PATH_REG)


def plot_graphs(df, pred_col, pred_type, resp_col, resp_type):
    if resp_type == "Boolean":
        if pred_type == "Categorical":
            return pg.cat_response_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            return pg.cat_resp_cont_predictor(df, pred_col, resp_col)
    elif resp_type == "Continuous":
        if pred_type == "Categorical":
            return pg.cont_resp_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            return pg.cont_response_cont_predictor(df, pred_col, resp_col)


def linear_reg_plots(y, x, fet_nm):
    feature_name = fet_nm
    predictor = sm.add_constant(x)
    linear_regression_model = sm.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
    p_value = np.float64(p_value)

    # Plot the figure
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    file_name = path_res + "/linear_reg_plot_" + fet_nm + ".html"

    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )

    return {
        "Column_name": fet_nm,
        "Column_type": "Continuous",
        "P_value": p_value,
        "T_value": t_value,
        "File_name": file_name,
    }


def log_reg_plots(y, X, fet_nm):
    feature_name = fet_nm
    log_reg = sm.Logit(y, X).fit()

    # Get the stats
    t_value = round(log_reg.tvalues[0], 6)
    p_value = "{:.6e}".format(log_reg.pvalues[0])
    p_value = np.float64(p_value)

    # Plot the figure
    fig = px.scatter(x=X, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    file_name = path_res + "/log_reg_plot_" + fet_nm + ".html"

    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )

    return {
        "Column_name": fet_nm,
        "Column_type": "Continuous",
        "P_value": p_value,
        "T_value": t_value,
        "File_name": file_name,
    }


def plot_morp_graphs(df, pred_col, pred_type, resp_col, resp_type):
    if resp_type == "Boolean":
        if pred_type == "Categorical":
            return mp.morp_cat_resp_cat_pred(df, pred_col, pred_type, resp_col)
        elif pred_type == "Continuous":
            return mp.morp_cat_resp_cont_pred(df, pred_col, pred_type, resp_col)
    elif resp_type == "Continuous":
        if pred_type == "Categorical":
            return mp.morp_cont_resp_cat_pred(df, pred_col, pred_type, resp_col)
        elif pred_type == "Continuous":
            return mp.morp_cont_resp_cont_pred(df, pred_col, pred_type, resp_col)


def rf_var_ranking_cont_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
    x_cont = df[cont_var_pred_list]
    x_cat = df[cat_var_pred_list]
    y = df[resp]

    for i in x_cat:
        le = LabelEncoder()
        x_cat[i] = le.fit_transform(x_cat[i])

    df = pd.concat([x_cont, x_cat], axis=1)

    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(df, y)
    rank_list = []
    for i, j in zip(df, rf.feature_importances_):
        if i in x_cont.columns:
            rank_list.append(
                {"Column_name": i, "Column_type": "Continuous", "fet_imp_coeff": j}
            )
        else:
            rank_list.append(
                {"Column_name": i, "Column_type": "Categorical", "fet_imp_coeff": j}
            )

    rank_list_df = pd.DataFrame(rank_list)
    return rank_list_df.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def rf_var_ranking_cat_resp(df, cont_var_pred_list, cat_var_pred_list, resp):
    x_cont = df[cont_var_pred_list]
    x_cat = df[cat_var_pred_list]
    y = df[resp]

    for i in x_cat:
        le = LabelEncoder()
        x_cat[i] = le.fit_transform(x_cat[i])

    df = pd.concat([x_cont, x_cat], axis=1)

    rf = RandomForestClassifier(n_estimators=150)
    rf.fit(df, y)
    rank_list = []
    for i, j in zip(df, rf.feature_importances_):
        if i in x_cont.columns:
            rank_list.append(
                {"Column_name": i, "Column_type": "Continuous", "fet_imp_coeff": j}
            )
        else:
            rank_list.append(
                {"Column_name": i, "Column_type": "Categorical", "fet_imp_coeff": j}
            )

    rank_list_df = pd.DataFrame(rank_list)
    return rank_list_df.sort_values("fet_imp_coeff", ascending=False).reset_index(
        drop=True
    )


def pred_typ(data_set, pred_list):
    pred_dict = {}
    for i in pred_list:
        if data_set[i].nunique() == 2:
            if data_set[i].dtype.kind in "iufc":
                pred_dict[i] = "Continuous"
            else:
                pred_dict[i] = "Categorical"
        elif type(data_set[i][0]) == str:
            pred_dict[i] = "Categorical"
        else:
            pred_dict[i] = "Continuous"
    return pred_dict


def url_click(url):
    if url:
        if "," in url:
            x = url.split(",")
            return f'{x[0]} <a target="_blank" href="{x[1]}">link to plot</a>'
        else:
            return f'<a target="_blank" href="{url}">link to plot</a>'


def main():
    df_dict = {}
    test_datasets = TestDatasets()
    for test in test_datasets.get_all_available_datasets():
        df, predictors, response = test_datasets.get_test_data_set(data_set_name=test)
        df_dict[test] = [df, predictors, response]
    flag = True
    while flag:
        print("Please select one of the five datasets given below:")
        for i in test_datasets.get_all_available_datasets():
            print(i)
        data_set_nm = input()
        if data_set_nm in ["mpg", "tips", "titanic", "diabetes", "breast_cancer"]:
            flag = False
        else:
            print("you have not selected one of the above datasets")

    print("you have selected,", data_set_nm.strip().lower())
    data_set = df_dict[data_set_nm.strip().lower()][0]
    predictors = df_dict[data_set_nm.strip().lower()][1]
    response = df_dict[data_set_nm.strip().lower()][2]

    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    else:
        print("what's up bro??! How many categories my response var got??")

    pred_dict = pred_typ(data_set, predictors)

    graph_list = []
    for i in pred_dict:
        graph_list.append(
            plot_graphs(
                data_set,
                pred_col=i,
                pred_type=pred_dict[i],
                resp_col=response,
                resp_type=resp_type,
            )
        )
    graph_df = pd.DataFrame(graph_list)

    regression_ranking_list = []
    for i in pred_dict:
        if pred_dict[i] == "Continuous":
            if resp_type == "Continuous":
                regression_ranking_list.append(
                    linear_reg_plots(
                        data_set[response].to_numpy(), data_set[i].to_numpy(), i
                    )
                )
            elif resp_type == "Boolean":
                regression_ranking_list.append(
                    log_reg_plots(
                        data_set[response].to_numpy(), data_set[i].to_numpy(), i
                    )
                )
    df_regression_ranking = pd.DataFrame(regression_ranking_list)
    df_regression_ranking = df_regression_ranking.sort_values(
        by=["P_value", "T_value"], ascending=[True, False]
    ).reset_index(drop=True)

    morp_rank_list = []
    for i in pred_dict:
        morp_rank_list.append(
            plot_morp_graphs(
                data_set,
                pred_col=i,
                pred_type=pred_dict[i],
                resp_col=response,
                resp_type=resp_type,
            )
        )

    morp_rank_df = (
        pd.DataFrame(morp_rank_list)
        .sort_values(by=["weighted_morp", "unweighted_morp"], ascending=[False, False])
        .reset_index(drop=True)
    )

    cat_pred_list = [i for i in pred_dict if pred_dict[i] == "Categorical"]
    cont_pred_list = [i for i in pred_dict if pred_dict[i] == "Continuous"]

    if resp_type == "Continuous":
        rf_rank_df = rf_var_ranking_cont_resp(
            data_set, cont_pred_list, cat_pred_list, response
        )

    else:

        rf_rank_df = rf_var_ranking_cat_resp(
            data_set, cont_pred_list, cat_pred_list, response
        )

    table_styles = [
        {"selector": "", "props": [("border", "1px solid black")]},
        {
            "selector": "th",
            "props": [
                ("background-color", "lightgray"),
                ("color", "black"),
                ("font-size", "12pt"),
                ("font-family", "Arial, Helvetica, sans-serif"),
                ("border", "1px solid black"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("font-size", "12pt"),
                ("font-family", "Arial, Helvetica, sans-serif"),
                ("border", "1px solid black"),
            ],
        },
    ]

    graph_df = graph_df.style.format(
        {"plot_link_1": url_click, "plot_link_2": url_click}
    ).set_table_styles(table_styles)

    df_regression_ranking = df_regression_ranking.style.format(
        {"File_name": url_click}
    ).set_table_styles(table_styles)

    df_regression_ranking.set_table_styles(table_styles)

    morp_rank_df = morp_rank_df.style.format({"Plot_link": url_click}).set_table_styles(
        table_styles
    )

    rf_rank_df = rf_rank_df.style.set_table_styles(table_styles)

    with open("dataset.html", "w") as out:
        out.write("<h3>Plots for all the Predictors</h3>")
        out.write(graph_df.to_html())
        out.write("<br><br>")
        out.write("<h3>Regression ranking with plots</h3>")
        out.write(df_regression_ranking.to_html())
        out.write("<br><br>")
        out.write("<h3>Mean of response plots</h3>")
        out.write(morp_rank_df.to_html())
        out.write("<br><br>")
        out.write("<h3>Random forrest variable ranking</h3>")
        out.write(rf_rank_df.to_html())


if __name__ == "__main__":
    sys.exit(main())
