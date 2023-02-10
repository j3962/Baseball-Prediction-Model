import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

path = "Jay_hw_01_plots"
isExist = os.path.exists(path)

if not isExist:
    os.mkdir(path)


def load_data():
    col = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    iris_df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        names=col,
        sep=",",
    )
    print("the shape of the iris table is:", iris_df.shape)
    print("the various columns in the dataset are:", iris_df.columns)
    print(
        "The no of flower categories and there count is:\n",
        iris_df["class"].value_counts(),
    )
    iris_np = iris_df.to_numpy()
    print(
        "The mean value of Sepal_len, Sepal_wid, Petal_len and Petal_wid is: ",
        np.mean(iris_np[:, :4], axis=0),
    )
    print(
        "The max value of Sepal_len, Sepal_wid, Petal_len and Petal_wid is: ",
        np.max(iris_np[:, :4], axis=0),
    )
    print(
        "The min value of Sepal_len, Sepal_wid, Petal_len and Petal_wid is: ",
        np.min(iris_np[:, :4], axis=0),
    )

    return iris_df


def plots(iris_df):
    # Creating plot to check for distribution of numerical columns and to check for outliers
    trace0 = go.Box(
        y=iris_df["sepal_wid"][iris_df["class"] == "Iris-setosa"],
        boxmean=True,
        name="setosa",
        marker_color="#3D9970",
        showlegend=False,
    )

    trace1 = go.Box(
        y=iris_df["sepal_wid"][iris_df["class"] == "Iris-versicolor"],
        boxmean=True,
        name="versicolor",
        marker_color="#FF4136",
        showlegend=False,
    )

    trace2 = go.Box(
        y=iris_df["sepal_wid"][iris_df["class"] == "Iris-virginica"],
        boxmean=True,
        name="virginica",
        marker_color="#FF851B",
        showlegend=False,
    )
    trace3 = go.Box(
        y=iris_df["sepal_len"][iris_df["class"] == "Iris-setosa"],
        boxmean=True,
        name="setosa",
        marker_color="#3D9970",
    )

    trace4 = go.Box(
        y=iris_df["sepal_len"][iris_df["class"] == "Iris-versicolor"],
        boxmean=True,
        name="versicolor",
        marker_color="#FF4136",
    )

    trace5 = go.Box(
        y=iris_df["sepal_len"][iris_df["class"] == "Iris-virginica"],
        boxmean=True,
        name="virginica",
        marker_color="#FF851B",
    )

    data = [trace0, trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        title="Sepal Width and Sepal Length distribution for different Classes",
        boxmode="group",
        xaxis=dict(title="Classes"),
        yaxis=dict(title="Length in cm"),
    )

    fig_box_1 = go.Figure(data=data, layout=layout)
    fig_box_1.write_html(
        file="Jay_hw_01_plots/"
             + "box_plot_distribution_1"
               ".html",
        include_plotlyjs="cdn",
    )

    trace0 = go.Box(
        y=iris_df["petal_wid"][iris_df["class"] == "Iris-setosa"],
        boxmean=True,
        name="setosa",
        showlegend=False,
    )

    trace1 = go.Box(
        y=iris_df["petal_wid"][iris_df["class"] == "Iris-versicolor"],
        boxmean=True,
        name="versicolor",
        showlegend=False,
    )

    trace2 = go.Box(
        y=iris_df["petal_wid"][iris_df["class"] == "Iris-virginica"],
        boxmean=True,
        name="virginica",
        showlegend=False,
    )
    trace3 = go.Box(
        y=iris_df["petal_len"][iris_df["class"] == "Iris-setosa"],
        boxmean=True,
        name="setosa",
    )

    trace4 = go.Box(
        y=iris_df["petal_len"][iris_df["class"] == "Iris-versicolor"],
        boxmean=True,
        name="versicolor",
    )

    trace5 = go.Box(
        y=iris_df["petal_len"][iris_df["class"] == "Iris-virginica"],
        boxmean=True,
        name="virginica",
    )

    data = [trace0, trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        title="Petal Width and Petal Length distribution for different Classes",
        boxmode="group",
        xaxis=dict(title="Classes"),
        yaxis=dict(title="Length in cm"),
    )

    fig_box_2 = go.Figure(data=data, layout=layout)
    fig_box_2.write_html(
        file="Jay_hw_01_plots/"
             + "box_plot_distribution_2"
               ".html",
        include_plotlyjs="cdn",
    )

    # creating the plot of correlation matrix
    data = [
        go.Heatmap(
            z=np.array(iris_df.corr().values),
            x=np.array(iris_df.corr().columns),
            y=np.array(iris_df.corr().columns),
            colorscale="YlGnBu",
        )
    ]
    layout = go.Layout(
        {
            "title": "Heatmap of Correlation ",
        }
    )

    fig_heatmap = go.Figure(data=data, layout=layout)
    fig_heatmap.write_html(
        file="Jay_hw_01_plots/"
             + "fig_heatmap_of_correlation_mtrx"
               ".html",
        include_plotlyjs="cdn",
    )

    # Scatter plots for reinforcing the correlationship between columns
    fig_scatt_ptl = px.scatter(
        iris_df,
        x=iris_df["petal_wid"],
        y=["petal_len"],
        color=iris_df["class"],
        symbol=iris_df["class"],
    )

    fig_scatt_ptl.update_layout(
        xaxis_title="Petal_wid_cm",
        yaxis_title="Petal_len_cm",
        title="PetalLenVSPetalWid",
    )
    fig_scatt_ptl.write_html(
        file="Jay_hw_01_plots/"
             + "fig_scatt_plt_1"
               ".html",
        include_plotlyjs="cdn",
    )

    fig_scatt_sep = px.scatter(
        iris_df,
        x=iris_df["sepal_wid"],
        y=["sepal_len"],
        color=iris_df["class"],
        symbol=iris_df["class"],
    )
    fig_scatt_sep.update_layout(
        xaxis_title="Sepal_wid_cm",
        yaxis_title="Sepal_len_cm",
        title="SepalLenVsSepalWid",
    )
    fig_scatt_ptl.write_html(
        file="Jay_hw_01_plots/"
             + "fig_scatt_plt_2"
               ".html",
        include_plotlyjs="cdn",
    )

    # the weak correlation between sepal_len and petal_len
    fig_scatt_sep_pep = px.scatter(
        iris_df,
        x=iris_df["sepal_len"],
        y=["petal_len"],
        color=iris_df["class"],
        symbol=iris_df["class"],
    )
    fig_scatt_sep_pep.update_layout(
        xaxis_title="sepal_len_cm",
        yaxis_title="petal_len_cm",
        title="SepalLenVsPetalLen",
    )
    fig_scatt_sep_pep.write_html(
        file="Jay_hw_01_plots/"
             + "fig_scatt_plt_2"
               ".html",
        include_plotlyjs="cdn",
    )

    return


def ml_model_random_forrest(iris_df):
    encoding_func = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    iris_df["class_encoded"] = iris_df["class"].map(encoding_func)

    # DataFrame to numpy values

    X_orig = iris_df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]].values

    y = iris_df["class_encoded"].values

    print("Model via Pipeline Predictions")

    pipeline = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    pipeline.fit(X_orig, y)

    probability = pipeline.predict_proba(X_orig)

    prediction = pipeline.predict(X_orig)

    print(f"Probability for RandomForrest: {probability}")

    print(f"Predictions for RandomForrest: {prediction}")

    return


def ml_model_logistic_regression(iris_df):
    encoding_func = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    iris_df["class_encoded"] = iris_df["class"].map(encoding_func)

    # DataFrame to numpy values

    X_orig = iris_df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]].values

    y = iris_df["class_encoded"].values

    print("Model via Pipeline Predictions")

    pipeline = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("logisticRegr", LogisticRegression()),
        ]
    )

    pipeline.fit(X_orig, y)

    probability = pipeline.predict_proba(X_orig)

    prediction = pipeline.predict(X_orig)

    print(f"Probability for LogisticRegression: {probability}")

    print(f"Predictions for LogisticRegression: {prediction}")

    return


def ml_model_GaussianNB(iris_df):
    encoding_func = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    iris_df["class_encoded"] = iris_df["class"].map(encoding_func)

    # DataFrame to numpy values

    X_orig = iris_df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]].values

    y = iris_df["class_encoded"].values

    print("Model via Pipeline Predictions")

    pipeline = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("Gnb", GaussianNB()),
        ]
    )

    pipeline.fit(X_orig, y)

    probability = pipeline.predict_proba(X_orig)

    prediction = pipeline.predict(X_orig)

    print(f"Probability for GaussianNB: {probability}")

    print(f"Predictions for GaussianNB: {prediction}")

    return


def mean_of_response_plot(iris_df, column_nm, class_nm):
    # input colmn_nm and class_nm
    x = column_nm + "_bin"
    y = "is_" + (class_nm.lower()).replace("-", "_")
    iris_df[y] = iris_df["class"].apply(
        lambda x: 1 if x.lower() == class_nm.lower() else 0
    )
    # creating bins for each colmn_nm and takin their avg
    iris_df[x] = (pd.cut(iris_df[column_nm], bins=10, right=True)).apply(
        lambda x: x.mid
    )
    # findin bin mean for given class
    hist_df = iris_df[x].value_counts().to_frame().reset_index()
    bin_mean_df = iris_df[[x, y]].groupby(x).mean().reset_index()

    fig_bar = go.Figure()
    fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
    # first subgraph of bins and their counts
    fig_bar.add_trace(
        go.Bar(
            x=hist_df["index"],
            y=hist_df[x],
            name=column_nm,
        ),
        secondary_y=False,
    )
    # Second subgraph of bin mean for given class
    fig_bar.add_trace(
        go.Scatter(
            x=bin_mean_df[x],
            y=bin_mean_df[y],
            name="bin_avg_for_" + class_nm,
            mode="lines+markers",
            # marker=True,
            marker=dict(color="Red"),
        ),
        secondary_y=True,
    )
    # overall avg graph for given class
    fig_bar.add_trace(
        go.Scatter(
            x=bin_mean_df[x],
            y=[iris_df[y].mean()] * len(bin_mean_df[x]),
            name=class_nm + "_avg",
            mode="lines",
        ),
        secondary_y=True,
    )
    # updating layout
    fig_bar.update_layout(
        title_text="Mean_response_plot_for_"
                   + column_nm
                   + "_and_"
                   + (class_nm.lower()).replace("-", "_"),
        yaxis=dict(
            title=dict(text="Total Population"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
        xaxis=dict(
            title=dict(text=column_nm + "_bins"),
        ),
    )
    fig_bar.write_html(
        file="Jay_hw_01_plots/"
             + column_nm
             + "_and_"
             + "morp_"
             + class_nm.replace("-", "_")
             + ".html",
        include_plotlyjs="cdn",
    )
    return


def call_morp(iris_df):
    j = 0
    i = 0
    while j < 4:
        if i == 3:
            i = 0
            j = j + 1
        if j == 4:
            break
        mean_of_response_plot(
            iris_df, list(iris_df.columns)[j], list(iris_df["class"].unique())[i]
        )
        i += 1


def main():
    df = load_data()
    plots(df)
    # ml_model_random_forrest(df)
    # ml_model_logistic_regression(df)
    # ml_model_GaussianNB(df)
    call_morp(df)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    sys.exit(main())
