import os
import sys

import pandas
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels
from sklearn import datasets

path = "Jay_hw_04_plots"
isExist = os.path.exists(path)

if not isExist:
    os.mkdir(path)

path_reg = "Jay_hw_04_regression_plots"
isExist = os.path.exists(path_reg)

if not isExist:
    os.mkdir(path_reg)


def cat_response_cat_predictor(data_set, pred_col, resp_col):
    pivoted_data = data_set.pivot_table(
        index=resp_col, columns=pred_col, aggfunc="size"
    )

    heatmap = go.Heatmap(
        x=pivoted_data.columns, y=pivoted_data.index, z=pivoted_data, colorscale="Blues"
    )

    # Define the layout of the plot
    layout = go.Layout(
        title="Heatmap", xaxis=dict(title=pred_col), yaxis=dict(title=resp_col)
    )

    # Create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the plot
    fig.write_html(
        file=path + "/" + "cat_" + resp_col + "_cat_" + pred_col + "_heatmap" + ".html",
        include_plotlyjs="cdn",
    )
    return


def cat_resp_cont_predictor(data_set, pred_col, resp_col):
    # Group data together

    hist_data = [
        data_set[data_set[resp_col] == i][pred_col] for i in data_set[resp_col].unique()
    ]

    group_labels = (
        data_set[resp_col]
        .value_counts()
        .to_frame()
        .reset_index()["index"]
        .astype("string")
    )

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels)
    fig_1.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig_1.write_html(
        file=path
        + "/"
        + "cat_"
        + resp_col
        + "_cont_"
        + pred_col
        + "_dist_plot"
        + ".html",
        include_plotlyjs="cdn",
    )
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=[curr_group] * len(data_set),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Categorical " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=resp_col,
        yaxis_title=pred_col,
    )
    fig_2.write_html(
        file=path
        + "/"
        + "cat_"
        + resp_col
        + "_cont_"
        + pred_col
        + "_violin_plot"
        + ".html",
        include_plotlyjs="cdn",
    )


def cont_resp_cat_predictor(data_set, pred_col, resp_col):
    # Group data together
    hist_data = [
        data_set[data_set[pred_col] == i][resp_col] for i in data_set[pred_col].unique()
    ]

    group_labels = data_set[pred_col].unique()

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous " + resp_col + " vs " + " Categorical " + pred_col,
        xaxis_title=resp_col,
        yaxis_title="Distribution",
    )
    fig_1.write_html(
        file=path
        + "/"
        + "cont_"
        + resp_col
        + "_cat_"
        + pred_col
        + "_dist_plot"
        + ".html",
        include_plotlyjs="cdn",
    )
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=[curr_group] * len(data_set),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous " + resp_col + " vs " + " Categorical " + pred_col,
        xaxis_title="Groupings",
        yaxis_title=resp_col,
    )
    fig_2.write_html(
        file=path + "cont_" + resp_col + "_cat_" + pred_col + "_violin_plot" + ".html",
        include_plotlyjs="cdn",
    )


def cont_response_cont_predictor(data_set, pred_col, resp_col):
    fig = px.scatter(x=data_set[pred_col], y=data_set[resp_col], trendline="ols")
    fig.update_layout(
        title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
        xaxis_title=pred_col,
        yaxis_title=resp_col,
    )
    fig.write_html(
        file=path
        + "/"
        + "cont_"
        + resp_col
        + "_cont_"
        + pred_col
        + "_scatter_plot"
        + ".html",
        include_plotlyjs="cdn",
    )


def plot_graphs(df, pred_col, pred_type, resp_col, resp_type):
    if resp_type == "Boolean":
        if pred_type == "Categorical":
            cat_response_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            cat_resp_cont_predictor(df, pred_col, resp_col)
    elif resp_type == "Continuous":
        if pred_type == "Categorical":
            cont_resp_cat_predictor(df, pred_col, resp_col)
        elif pred_type == "Continuous":
            cont_response_cont_predictor(df, pred_col, resp_col)


def linear_reg_plots(y, x, fet_nm):
    feature_name = fet_nm
    predictor = statsmodels.api.add_constant(x)
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {feature_name}")
    print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.write_html(
        file=path_reg + "/" + fet_nm + "_linear_reg_plot" + ".html",
        include_plotlyjs="cdn",
    )


def log_reg_plot(y, x, fet_nm):
    feature_name = fet_nm
    log_reg = statsmodels.api.Logit(y, x).fit()
    print(f"Variable: {feature_name}")
    print(log_reg.summary())
    # Get the stats
    t_value = round(log_reg.tvalues[0], 6)
    p_value = "{:.6e}".format(log_reg.pvalues[0])

    # Plot the figure
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {feature_name}",
        yaxis_title="y",
    )
    fig.write_html(
        file=path_reg + "/" + fet_nm + "_logistic_reg_plot" + ".html",
        include_plotlyjs="cdn",
    )


def main():
    data = datasets.load_diabetes()
    data_set = pandas.DataFrame(data.data, columns=data.feature_names)

    data_set["target"] = data.target
    predictors = data.feature_names
    response = "target"

    if len(data_set[response].value_counts()) > 2:
        resp_type = "Continuous"

    elif len(data_set[response].value_counts()) == 2:
        resp_type = "Boolean"

    else:
        print("what's up bro??! How many categories my response var got??")

    for col in predictors:
        if type(data_set[col][0]) == str:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        plot_graphs(
            data_set,
            pred_col=col,
            pred_type=pred_type,
            resp_col=response,
            resp_type=resp_type,
        )

    print("*" * 160)
    print("\n\n\n")
    print("*" * 160)
    print("Regression plots\n\n")
    for col in predictors:
        if type(data_set[col][0]) == str or data_set[col][0] in [True, False]:
            pred_type = "Categorical"
        else:
            pred_type = "Continuous"
        if pred_type == "Continuous":
            if resp_type == "Continuous":
                linear_reg_plots(
                    data_set[response].to_numpy(), data_set[col].to_numpy(), col
                )
            elif resp_type == "Boolean":
                log_reg_plot(
                    data_set[response].to_numpy(), data_set[col].to_numpy(), col
                )


if __name__ == "__main__":
    sys.exit(main())
