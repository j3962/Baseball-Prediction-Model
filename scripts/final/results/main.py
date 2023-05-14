import os
import sys

import File_path as fp
import numpy as np
import pandas as pd

# import plotly.graph_objects as go
import sqlalchemy
import statsmodels.api as sm
from correlation_and_plots import PredsCorrelation as pc
from Morp_2d_plots import Morp2dPlots as mp2d
from morp_plots import MorpPlots as mp
from pred_and_resp_graphs import PlotGraph as pg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

path = fp.GLOBAL_PATH
isExist = os.path.exists(path)

if not isExist:
    os.mkdir(path)

path_2d_morp = fp.GLOBAL_PATH_2D_MORP
isExist = os.path.exists(path_2d_morp)

if not isExist:
    os.mkdir(path_2d_morp)


def pred_typ(data_set, pred_list):
    pred_dict = {}
    for i in pred_list:
        if i == "":
            pred_dict[i] = "Continuous"
        elif data_set[i].nunique() == 2:
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


def html_report(data_set, predictors, response, file_name):

    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    else:
        print("what's up bro??! How many categories my response var got??")

    pred_dict = pred_typ(data_set, predictors)

    cat_pred_list = [i for i in pred_dict if pred_dict[i] == "Categorical"]
    cont_pred_list = [i for i in pred_dict if pred_dict[i] == "Continuous"]
    #
    if resp_type == "Continuous":
        rf_rank_df = pg.rf_var_ranking_cont_resp(data_set, cont_pred_list, response)

    else:

        rf_rank_df = pg.rf_var_ranking_cat_resp(data_set, cont_pred_list, response)

    cont_fet_prop_list = []
    cont_fet_prop_dict = {}
    for i in cont_pred_list:
        if resp_type == "Continuous":
            dict1 = pg.cont_resp_cont_pred(data_set, i, response)
            dict2 = mp.morp_cont_resp_cont_pred(data_set, i, "Continuous", response)
            dict3 = pg.linear_reg_plots(
                data_set[response].to_numpy(), data_set[i].to_numpy(), i
            )
            cont_fet_prop_dict = {
                "Feature_nm": i,
                "Plot_link1": dict1["plot_link_1"],
                "Plot_link2": None,
                "Weighted_morp": dict2["weighted_morp"],
                "Unweighted_morp": dict2["unweighted_morp"],
                "Morp_plot_link": dict2["Plot_link"],
                "P_value": dict3["P_value"],
                "T_value": dict3["T_value"],
            }

        elif resp_type == "Boolean":
            dict1 = pg.cat_resp_cont_pred(data_set, i, response)
            dict2 = mp.morp_cat_resp_cont_pred(data_set, i, "Continuous", response)
            dict3 = pg.log_reg_plots(
                data_set[response].to_numpy(), data_set[i].to_numpy(), i
            )
            cont_fet_prop_dict = {
                "Feature_nm": i,
                "Plot_link1": dict1["plot_link_1"],
                "Plot_link2": dict1["plot_link_2"],
                "Weighted_morp": dict2["weighted_morp"],
                "Unweighted_morp": dict2["unweighted_morp"],
                "Morp_plot_link": dict2["Plot_link"],
                "P_value": dict3["P_value"],
                "T_value": dict3["T_value"],
            }

        cont_fet_prop_list.append(cont_fet_prop_dict)

    cont_fet_prop_df = pd.DataFrame(cont_fet_prop_list)
    if len(cont_fet_prop_df) >= 1:
        cont_fet_prop_df = cont_fet_prop_df.merge(
            rf_rank_df, left_on=["Feature_nm"], right_on=["Column_name"]
        )
        cont_fet_prop_df = cont_fet_prop_df.sort_values(
            by="Weighted_morp", ascending=False
        ).reset_index(drop=True)
        cont_fet_prop_df = cont_fet_prop_df.drop(labels=["Column_name"], axis=1)

    cat_fet_prop_list = []
    cat_fet_prop_dict = {}
    for i in cat_pred_list:
        if resp_type == "Continuous":
            dict1 = pg.cont_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cont_resp_cat_pred(data_set, i, "Continuous", response)
            cat_fet_prop_dict = {
                "Feature_nm": i,
                "Plot_link1": dict1["plot_link_1"],
                "Plot_link2": dict1["plot_link_2"],
                "Weighted_morp": dict2["weighted_morp"],
                "Unweighted_morp": dict2["unweighted_morp"],
                "Morp_plot_link": dict2["Plot_link"],
            }

        elif resp_type == "Boolean":
            dict1 = pg.cat_resp_cat_pred(data_set, i, response)
            dict2 = mp.morp_cat_resp_cat_pred(data_set, i, "Continuous", response)
            cat_fet_prop_dict = {
                "Feature_nm": i,
                "Plot_link1": dict1["plot_link_1"],
                "Plot_link2": None,
                "Weighted_morp": dict2["weighted_morp"],
                "Unweighted_morp": dict2["unweighted_morp"],
                "Morp_plot_link": dict2["Plot_link"],
            }
        cat_fet_prop_list.append(cat_fet_prop_dict)

    cat_fet_prop_df = pd.DataFrame(cat_fet_prop_list)

    if len(cat_fet_prop_df) >= 1:
        cat_fet_prop_df = cat_fet_prop_df.sort_values(
            by="Weighted_morp", ascending=False
        ).reset_index(drop=True)

    # cont_cont_correlation driver and logic
    cont_cont_list = []
    cont_cont_dict = {}
    for i in cont_pred_list:
        for j in cont_pred_list:
            if resp_type == "Continuous":
                cont_cont_dict = {
                    "Cont_1": i,
                    "Cont_2": j,
                    "Correlation": pc.cont_pred_corr(data_set[i], data_set[j]),
                    "Cont_1_morp_url": mp.morp_cont_resp_cont_pred(
                        data_set, i, "Continuous", response
                    )["Plot_link"],
                    "Cont_2_morp_url": mp.morp_cont_resp_cont_pred(
                        data_set, j, "Continuous", response
                    )["Plot_link"],
                }
            elif resp_type == "Boolean":
                cont_cont_dict = {
                    "Cont_1": i,
                    "Cont_2": j,
                    "Correlation": pc.cont_pred_corr(data_set[i], data_set[j]),
                    "Cont_1_morp_url": mp.morp_cat_resp_cont_pred(
                        data_set, i, "Continuous", response
                    )["Plot_link"],
                    "Cont_2_morp_url": mp.morp_cat_resp_cont_pred(
                        data_set, j, "Continuous", response
                    )["Plot_link"],
                }
            cont_cont_list.append(cont_cont_dict)
    if len(cont_cont_list) >= 1:
        cont_cont_corr_df = pd.DataFrame(cont_cont_list)
        cont_cont_corr_htmp_plt = pc.corr_heatmap_plots(
            cont_cont_corr_df, "Cont_1", "Cont_2", "Correlation"
        )
        cont_cont_corr_df = cont_cont_corr_df[
            cont_cont_corr_df["Cont_1"] != cont_cont_corr_df["Cont_2"]
        ]
        cont_cont_corr_df = cont_cont_corr_df.sort_values(
            by="Correlation", ascending=False
        ).reset_index(drop=True)

    else:
        cont_cont_corr_df = pd.DataFrame(cont_cont_list)
        cont_cont_corr_htmp_plt = ""

    # cat_cat_correlation driver and logic
    cat_cat_list_t = []
    cat_cat_dict_t = {}
    cat_cat_list_v = []
    cat_cat_dict_v = {}
    for i in cat_pred_list:
        for j in cat_pred_list:
            if resp_type == "Continuous":
                cat_cat_dict_v = {
                    "Cat_1": i,
                    "Cat_2": j,
                    "Correlation_V": pc.cat_correlation(data_set[i], data_set[j]),
                    "Cat_1_morp_url": mp.morp_cont_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cat_2_morp_url": mp.morp_cont_resp_cat_pred(
                        data_set, j, "Categorical", response
                    )["Plot_link"],
                }
                cat_cat_dict_t = {
                    "Cat_1": i,
                    "Cat_2": j,
                    "Correlation_T": pc.cat_correlation(
                        data_set[i], data_set[j], tschuprow=True
                    ),
                    "Cat_1_morp_url": mp.morp_cont_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cat_2_morp_url": mp.morp_cont_resp_cat_pred(
                        data_set, j, "Categorical", response
                    )["Plot_link"],
                }
            elif resp_type == "Boolean":
                cat_cat_dict_v = {
                    "Cat_1": i,
                    "Cat_2": j,
                    "Correlation_V": pc.cat_correlation(data_set[i], data_set[j]),
                    "Cat_1_morp_url": mp.morp_cat_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cat_2_morp_url": mp.morp_cat_resp_cat_pred(
                        data_set, j, "Categorical", response
                    )["Plot_link"],
                }
                cat_cat_dict_t = {
                    "Cat_1": i,
                    "Cat_2": j,
                    "Correlation_T": pc.cat_correlation(
                        data_set[i], data_set[j], tschuprow=True
                    ),
                    "Cat_1_morp_url": mp.morp_cat_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cat_2_morp_url": mp.morp_cat_resp_cat_pred(
                        data_set, j, "Categorical", response
                    )["Plot_link"],
                }
            cat_cat_list_t.append(cat_cat_dict_t)
            cat_cat_list_v.append(cat_cat_dict_v)

    if len(cat_cat_list_t) >= 1 and len(cat_cat_list_v) >= 1:
        cat_cat_corr_t_df = pd.DataFrame(cat_cat_list_t)
        cat_cat_corr_v_df = pd.DataFrame(cat_cat_list_v)
        cat_cat_corr_t_htmp_plt = pc.corr_heatmap_plots(
            cat_cat_corr_t_df, "Cat_1", "Cat_2", "Correlation_T"
        )
        cat_cat_corr_v_htmp_plt = pc.corr_heatmap_plots(
            cat_cat_corr_v_df, "Cat_1", "Cat_2", "Correlation_V"
        )
        cat_cat_corr_t_df = cat_cat_corr_t_df[
            cat_cat_corr_t_df["Cat_1"] != cat_cat_corr_t_df["Cat_2"]
        ]
        cat_cat_corr_v_df = cat_cat_corr_v_df[
            cat_cat_corr_v_df["Cat_1"] != cat_cat_corr_v_df["Cat_2"]
        ]
        cat_cat_corr_t_df = cat_cat_corr_t_df.sort_values(
            by="Correlation_T", ascending=False
        ).reset_index(drop=True)
        cat_cat_corr_v_df = cat_cat_corr_v_df.sort_values(
            by="Correlation_V", ascending=False
        ).reset_index(drop=True)

    else:
        cat_cat_corr_t_df = pd.DataFrame(cat_cat_list_t)
        cat_cat_corr_v_df = pd.DataFrame(cat_cat_list_v)
        cat_cat_corr_v_htmp_plt = ""
        cat_cat_corr_t_htmp_plt = ""

    # cat_cont_correlatino driver and logic
    cat_cont_list = []
    cat_cont_dict = {}
    for i in cat_pred_list:
        for j in cont_pred_list:
            if resp_type == "Continuous":
                cat_cont_dict = {
                    "Cat": i,
                    "Cont": j,
                    "Correlation": pc.cat_cont_correlation_ratio(
                        data_set[i], data_set[j]
                    ),
                    "Cat_morp_url": mp.morp_cont_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cont_morp_url": mp.morp_cont_resp_cont_pred(
                        data_set, j, "Continuous", response
                    )["Plot_link"],
                }
            elif resp_type == "Boolean":
                cat_cont_dict = {
                    "Cat": i,
                    "Cont": j,
                    "Correlation": pc.cat_cont_correlation_ratio(
                        data_set[i], data_set[j]
                    ),
                    "Cat_morp_url": mp.morp_cat_resp_cat_pred(
                        data_set, i, "Categorical", response
                    )["Plot_link"],
                    "Cont_morp_url": mp.morp_cat_resp_cont_pred(
                        data_set, j, "Continuous", response
                    )["Plot_link"],
                }
            cat_cont_list.append(cat_cont_dict)

    if len(cat_cont_list) >= 1:
        cat_cont_corr_df = pd.DataFrame(cat_cont_list)
        cat_cont_corr_htmp_plt = pc.corr_heatmap_plots(
            cat_cont_corr_df, "Cont", "Cat", "Correlation"
        )
        cat_cont_corr_df = cat_cont_corr_df[
            cat_cont_corr_df["Cont"] != cat_cont_corr_df["Cat"]
        ]
        cat_cont_corr_df = cat_cont_corr_df.sort_values(
            by="Correlation", ascending=False
        ).reset_index(drop=True)

    else:
        cat_cont_corr_df = pd.DataFrame(cat_cont_list)
        cat_cont_corr_htmp_plt = ""
    # brute force cat cat logic

    # if cat_pred_listif \

    if len(cat_pred_list) > 1:

        cat_cat_2d_morp_list = []
        cat_cat_2d_morp_dict = {}

        for i in cat_pred_list:
            for j in cat_pred_list:
                if i == j:
                    continue
                else:
                    dict1 = mp2d.cat_cat_2d_morp(data_set, i, j, response)
                cat_cat_2d_morp_dict = {
                    "Cat_1": i,
                    "Cat_2": j,
                    "Weighted_morp": dict1["Weighted_morp"],
                    "Unweighted_morp": dict1["Unweighted_morp"],
                    "Plot_link": dict1["Plot_link"],
                }
                cat_cat_2d_morp_list.append(cat_cat_2d_morp_dict)

        cat_cat_2d_morp_df = pd.DataFrame(cat_cat_2d_morp_list)
        cat_cat_2d_morp_df = cat_cat_2d_morp_df.merge(
            cat_cat_corr_t_df, left_on=["Cat_1", "Cat_2"], right_on=["Cat_1", "Cat_2"]
        )
        cat_cat_2d_morp_df = cat_cat_2d_morp_df.merge(
            cat_cat_corr_v_df,
            left_on=["Cat_1", "Cat_2", "Cat_1_morp_url", "Cat_2_morp_url"],
            right_on=["Cat_1", "Cat_2", "Cat_1_morp_url", "Cat_2_morp_url"],
        )
        cat_cat_2d_morp_df["Correlation_T_Abs"] = cat_cat_2d_morp_df[
            "Correlation_T"
        ].abs()
        cat_cat_2d_morp_df["Correlation_V_Abs"] = cat_cat_2d_morp_df[
            "Correlation_V"
        ].abs()
        cat_cat_2d_morp_df = (
            cat_cat_2d_morp_df.sort_index(axis=1, ascending=True)
            .sort_values("Weighted_morp", ascending=False)
            .reset_index(drop=True)
        )

    else:
        cat_cat_2d_morp_df = pd.DataFrame([])

    # brute force cat cont logic

    if len(cat_pred_list) > 0 and len(cont_pred_list) > 0:

        cat_cont_2d_morp_list = []
        cat_cont_2d_morp_dict = {}

        for i in cat_pred_list:
            for j in cont_pred_list:
                if i == j:
                    continue
                else:
                    dict1 = mp2d.cat_cont_2d_morp(data_set, i, j, response)
                    # print(dict1)
                cat_cont_2d_morp_dict = {
                    "Cat": i,
                    "Cont": j,
                    "Weighted_morp": dict1["Weighted_morp"],
                    "Unweighted_morp": dict1["Unweighted_morp"],
                    "Plot_link": dict1["Plot_link"],
                }
                cat_cont_2d_morp_list.append(cat_cont_2d_morp_dict)

        cat_cont_2d_morp_df = pd.DataFrame(cat_cont_2d_morp_list)
        cat_cont_2d_morp_df = cat_cont_2d_morp_df.merge(
            cat_cont_corr_df, left_on=["Cat", "Cont"], right_on=["Cat", "Cont"]
        )
        cat_cont_2d_morp_df["Correlation_Abs"] = cat_cont_2d_morp_df[
            "Correlation"
        ].abs()
        cat_cont_2d_morp_df = (
            cat_cont_2d_morp_df.sort_index(axis=1, ascending=True)
            .sort_values("Weighted_morp", ascending=False)
            .reset_index(drop=True)
        )

    else:
        cat_cont_2d_morp_df = pd.DataFrame([])

        # brute force cont_cont_logic

    if len(cont_pred_list) > 1:

        cont_cont_2d_morp_list = []
        cont_cont_2d_morp_dict = {}

        for i in cont_pred_list:
            for j in cont_pred_list:
                if i == j:
                    continue
                else:
                    dict1 = mp2d.cont_cont_2d_morp(data_set, i, j, response)
                    # print(dict1)
                cont_cont_2d_morp_dict = {
                    "Cont_1": i,
                    "Cont_2": j,
                    "Weighted_morp": dict1["Weighted_morp"],
                    "Unweighted_morp": dict1["Unweighted_morp"],
                    "Plot_link": dict1["Plot_link"],
                }
                cont_cont_2d_morp_list.append(cont_cont_2d_morp_dict)

        cont_cont_2d_morp_df = pd.DataFrame(cont_cont_2d_morp_list)
        cont_cont_2d_morp_df = cont_cont_2d_morp_df.merge(
            cont_cont_corr_df,
            left_on=["Cont_1", "Cont_2"],
            right_on=["Cont_1", "Cont_2"],
        )
        cont_cont_2d_morp_df["Correlation_Abs"] = cont_cont_2d_morp_df[
            "Correlation"
        ].abs()
        cont_cont_2d_morp_df = (
            cont_cont_2d_morp_df.sort_index(axis=1, ascending=True)
            .sort_values("Weighted_morp", ascending=False)
            .reset_index(drop=True)
        )

    else:
        cont_cont_2d_morp_df = pd.DataFrame([])

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

    # making prop dfs clickables
    cat_fet_prop_df = cat_fet_prop_df.style.format(
        {"Plot_link1": url_click, "Plot_link2": url_click, "Morp_plot_link": url_click}
    ).set_table_styles(table_styles)

    cont_fet_prop_df = cont_fet_prop_df.style.format(
        {"Plot_link1": url_click, "Plot_link2": url_click, "Morp_plot_link": url_click}
    ).set_table_styles(table_styles)

    # making corr dfs clickable
    cont_cont_corr_df = cont_cont_corr_df.style.format(
        {"Cont_1_morp_url": url_click, "Cont_2_morp_url": url_click}
    ).set_table_styles(table_styles)

    cat_cat_corr_t_df = cat_cat_corr_t_df.style.format(
        {"Cat_1_morp_url": url_click, "Cat_2_morp_url": url_click}
    ).set_table_styles(table_styles)

    cat_cat_corr_v_df = cat_cat_corr_v_df.style.format(
        {"Cat_1_morp_url": url_click, "Cat_2_morp_url": url_click}
    ).set_table_styles(table_styles)

    cat_cont_corr_df = cat_cont_corr_df.style.format(
        {"Cat_morp_url": url_click, "Cont_morp_url": url_click}
    ).set_table_styles(table_styles)

    # making brute force dfs clickable

    cat_cat_2d_morp_df = cat_cat_2d_morp_df.style.format(
        {
            "Cat_1_morp_url": url_click,
            "Cat_2_morp_url": url_click,
            "Plot_link": url_click,
        }
    ).set_table_styles(table_styles)

    cat_cont_2d_morp_df = cat_cont_2d_morp_df.style.format(
        {"Cat_morp_url": url_click, "Cont_morp_url": url_click, "Plot_link": url_click}
    ).set_table_styles(table_styles)

    cont_cont_2d_morp_df = cont_cont_2d_morp_df.style.format(
        {
            "Cont_1_morp_url": url_click,
            "Cont_2_morp_url": url_click,
            "Plot_link": url_click,
        }
    ).set_table_styles(table_styles)

    with open(file_name, "w") as out:
        out.write("<h5>Continuous Predictors Properties</h5>")
        out.write(cont_fet_prop_df.to_html())
        out.write("<br><br>")
        out.write("<h5>Categorical Predictors Properties</h5>")
        out.write(cat_fet_prop_df.to_html())
        out.write("<h5>Categorical/ Categorical Correlation</h5>")
        out.write("<br><br>")
        out.write("<h4>Correlation Tschuprow Matrix Heatmap</h4>")
        out.write(cat_cat_corr_t_htmp_plt)
        out.write("<br><br>")
        out.write("<h4>Correlation Cramer's Matrix Heatmap</h4>")
        out.write(cat_cat_corr_v_htmp_plt)
        out.write("<br><br>")
        out.write("<h4>Correlation Tschuprow Matrix</h4>")
        out.write(cat_cat_corr_t_df.to_html())
        out.write("<br><br>")
        out.write("<h4>Correlation Cramer's Matrix</h4>")
        out.write(cat_cat_corr_v_df.to_html())
        out.write("<br><br>")
        out.write("<h5>Categorical/ Continuous Correlation</h5>")
        out.write(cat_cont_corr_htmp_plt)
        out.write("<br><br>")
        out.write("<h4>Categorical/ Continuous Correlation Matrix</h4>")
        out.write(cat_cont_corr_df.to_html())
        out.write("<br><br>")
        out.write("<h5>Continuous/ Continuous Correlation</h5>")
        out.write(cont_cont_corr_htmp_plt)
        out.write("<br><br>")
        out.write("<h4>Continuous/ Continuous Correlation Matrix</h4>")
        out.write(cont_cont_corr_df.to_html())
        out.write("<br><br>")
        out.write("<h4>Categorical Categorical Brute force combination</h4>")
        out.write(cat_cat_2d_morp_df.to_html())
        out.write("<br><br>")
        out.write("<h4>Cate0gorical Continuous Brute force combination</h4>")
        out.write(cat_cont_2d_morp_df.to_html())
        out.write("<br><br>")
        out.write("<h4>Continuous Continuous Brute force combination</h4>")
        out.write(cont_cont_2d_morp_df.to_html())

    return True


def ml_models(data_set, predictors_cont, response, predictors_cat):

    if len(predictors_cat) > 0:

        x_cont = data_set[predictors_cont]
        x_cat = data_set[predictors_cat]

        for i in x_cat:
            le = LabelEncoder()
            x_cat[i] = le.fit_transform(x_cat[i])

        predictors = list(predictors_cont) + list(predictors_cat)

        data_set = pd.concat([x_cont, x_cat, data_set[response]], axis=1)
    else:

        predictors = predictors_cont

        data_set = pd.concat([data_set[predictors], data_set[response]], axis=1)

    X_train = data_set[data_set["local_date"].dt.year < 2012][predictors[1:]]

    X_test = data_set[data_set["local_date"].dt.year == 2012][predictors[1:]]

    y_train = data_set[data_set["local_date"].dt.year < 2012][response]

    y_test = data_set[data_set["local_date"].dt.year == 2012][response]

    # Initialize a random forest classifier with 100 trees
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the random forest model on the training data
    rf.fit(X_train, y_train)

    # Evaluate the accuracy of the model
    accuracy = rf.score(X_test, y_test)
    print(f"Random Forrest Accuracy: {accuracy:.2f}")

    y_pred = rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Random Forrest Confusion matrix:\n", cm)

    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit()
    #
    # # Print summary of results
    print(result.summary())
    y_pred = result.predict(X_test) > 0.5
    #
    threshold = 0.5  # classification threshold
    y_pred_class = (y_pred >= threshold).astype(int)  # convert probabilities to classes
    accuracy = np.mean(y_pred_class == y_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # I tried SVM model but it takes too freakinnngggg long so not doing it

    # # Train the SVM model on the training set
    # svm.fit(X_train, y_train)
    #
    # # Make predictions on the testing set
    # y_pred = svm.predict(X_test)
    #
    # # Compute the accuracy of the SVM model
    # accuracy = accuracy_score(y_test, y_pred)
    # print('Accuracy:', accuracy)


def main():

    db_database = "baseball"
    db_user = "root"
    db_pass = "1997"
    db_host = "jay_mariadb"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
            SELECT * from final_feat_stats;
        """

    df = pd.DataFrame(sql_engine.connect().execute(text(query)))
    df_new = df.iloc[:, 5:].fillna(0)

    for i in df_new.columns:
        if df_new[i].dtype == "O":
            df_new[i] = df_new[i].astype(float)

    query_diff = """
            SELECT * from final_feat_stats_diff;
        """

    df_diff = pd.DataFrame(sql_engine.connect().execute(text(query_diff)))

    df_diff_new = df_diff.iloc[:, 5:].fillna(0)

    for i in df_diff_new.columns:
        if type(df_diff_new[i][0]) == str:
            continue
        elif df_diff_new[i].dtype == "O":
            df_diff_new[i] = df_diff_new[i].astype(float)

    data_set = df_new

    data_set_diff = df_diff_new

    print("*" * 80)
    print("*" * 80)

    print(
        "Initially, I had tried to take ration between home and away features. I was skeptical about it not doing "
        "well! If at any time I had"
        "a new starting pitcher from either of the team then my all the features were becoming zero."
        " This was not desirable"
    )
    html_report(
        data_set,
        df_new.columns[1:12],
        str(df_new.columns[13]),
        "dataset_ratio.html",
    )

    ml_models(
        data_set,
        df_new.columns[0:12],
        str(df_new.columns[13]),
        [],
    )

    print("*" * 80)
    print("*" * 80)

    print(
        "Trying to take difference between home and away predictors. And suddenly my features do a lot better"
    )

    html_report(
        data_set_diff,
        df_diff_new.columns[1:13],
        str(df_diff_new.columns[24]),
        "dataset_diff.html",
    )

    ml_models(
        data_set_diff, df_diff_new.columns[0:13], str(df_diff_new.columns[24]), []
    )

    print("*" * 80)
    print("*" * 80)

    print(
        "Now, I am adding some categorical features. Here are"
        "difference stats with stad_win_per,and few, extreme_temp_event, fav_overcast, hour"
        "I finding home team win rate at each stadium did better than just stadium"
    )

    html_report(
        data_set_diff,
        df_diff_new.columns[1:16],
        str(df_diff_new.columns[24]),
        "dataset_diff_cat.html",
    )

    ml_models(
        data_set_diff,
        df_diff_new.columns[0:14],
        str(df_diff_new.columns[24]),
        df_diff_new.columns[14:17],
    )

    print("*" * 80)
    print("*" * 80)

    print(
        "I took team level average of my bad features and suddenly my accuracy goes upp!!"
    )
    excluded_model_list = [
        "batting_average_against_diff",
        "strikeout_to_walk_diff",
        "whip_diff",
        "fip_diff",
        "bb_9_diff",
        "ip_gs_diff",
        "extreme_temp_event",
        "fav_overcast",
        "match_start_hour",
        "pitch_count_diff",
    ]

    excluded_list = [
        "batting_average_against_diff",
        "strikeout_to_walk_diff",
        "whip_diff",
        "fip_diff",
        "bb_9_diff",
        "ip_gs_diff",
        "pitch_count_diff",
    ]

    html_report(
        data_set_diff,
        [i for i in df_diff_new.columns[1:24] if i not in excluded_list],
        str(df_diff_new.columns[24]),
        "dataset_diff_avg.html",
    )

    ml_models(
        data_set_diff,
        [i for i in df_diff_new.columns[0:24] if i not in excluded_model_list],
        str(df_diff_new.columns[24]),
        df_diff_new.columns[14:17],
    )

    print("Taking out bad features: ")
    excluded_model_list = [
        "batting_average_against_diff",
        "strikeout_to_walk_diff",
        "whip_diff",
        "fip_diff",
        "bb_9_diff",
        "ip_gs_diff",
        "extreme_temp_event",
        "fav_overcast",
        "match_start_hour",
        "pitch_count_diff",
        "series_streak_diff",
        "starting_pitch_home_w_diff",
        "home_away_win_diff",
    ]

    excluded_list = [
        "batting_average_against_diff",
        "strikeout_to_walk_diff",
        "whip_diff",
        "fip_diff",
        "bb_9_diff",
        "ip_gs_diff",
        "match_start_hour",
        "pitch_count_diff",
        "series_streak_diff",
        "starting_pitch_home_w_diff",
        "home_away_win_diff",
    ]

    html_report(
        data_set_diff,
        [i for i in df_diff_new.columns[1:24] if i not in excluded_list],
        str(df_diff_new.columns[24]),
        "dataset_diff_avg_refined.html",
    )

    ml_models(
        data_set_diff,
        [i for i in df_diff_new.columns[0:24] if i not in excluded_model_list],
        str(df_diff_new.columns[24]),
        df_diff_new.columns[14:16],
    )


if __name__ == "__main__":
    sys.exit(main())
