import File_path as fp
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class MorpPlots:
    @staticmethod
    def weighted_and_unweighted_morp(bin_mean_df, hist_df, df, x, y, resp_col):
        """

        :param bin_mean_df:
        :param hist_df:
        :param df:
        :param x:
        :param y:
        :param resp_col:
        :return:
        """
        bin_mean_df["unweighted_morp"] = bin_mean_df[resp_col].apply(
            lambda x: (df[resp_col].mean() - x) ** 2
        )
        hist_df["pop_prop"] = hist_df[x].apply(lambda y: y / (hist_df[x].sum()))
        morp_df = pd.merge(
            bin_mean_df, hist_df, how="inner", left_on=x, right_on="index"
        )
        morp_df["weighted_morp"] = morp_df.unweighted_morp * morp_df.pop_prop
        unweighted_morp = morp_df["unweighted_morp"].sum() / len(morp_df)
        weighted_morp = morp_df["weighted_morp"].sum()
        return unweighted_morp, weighted_morp

    @staticmethod
    def morp_cat_resp_cont_pred(df, pred_colmn, pred_type, resp_col):
        """

        :param df:
        :param pred_colmn:
        :param pred_type:
        :param resp_col:
        :return:
        """
        class_nm = "bool_true"
        # input colmn_nm and class_nm
        x = pred_colmn + "_bin"
        y = resp_col

        # creatin the is_class_nm column
        # creating bins for each colmn_nm and takin their avg
        df[x] = (pd.cut(df[pred_colmn], bins=10, right=True)).apply(lambda x: x.mid)
        # findin bin mean for given class
        hist_df = df[x].value_counts().to_frame().reset_index()
        bin_mean_df = df[[x, y]].groupby(x).mean().reset_index()

        unweighted_morp, weighted_morp = MorpPlots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        fig_bar = go.Figure()
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

        # first subgraph of bins and their counts
        fig_bar.add_trace(
            go.Bar(
                x=hist_df["index"],
                y=hist_df[x],
                name=pred_colmn,
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
                y=[df[y].mean()] * len(bin_mean_df[x]),
                name=class_nm + "_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig_bar.update_layout(
            title_text="Mean_response_plot_for_"
            + pred_colmn
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
                title=dict(text=pred_colmn + "_bins"),
            ),
        )
        file_name = (
            fp.GLOBAL_PATH + "/" + "cat_" + resp_col + "_cont_" + pred_colmn + ".html"
        )
        fig_bar.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {
            "col_name": pred_colmn,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    # input colmn_nm and class_nm

    @staticmethod
    def morp_cat_resp_cat_pred(df, pred_colmn, pred_type, resp_col):
        """

        :param df:
        :param pred_colmn:
        :param pred_type:
        :param resp_col:
        :return:
        """
        x = pred_colmn
        y = resp_col

        # findin bin mean for given class
        hist_df = df[x].value_counts().to_frame().reset_index().sort_values("index")
        bin_mean_df = df[[x, y]].groupby(x).mean().reset_index().sort_values(pred_colmn)
        fig_bar = go.Figure()
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

        unweighted_morp, weighted_morp = MorpPlots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        # first subgraph of bins and their counts
        fig_bar.add_trace(
            go.Bar(
                x=hist_df["index"],
                y=hist_df[x],
                name=pred_colmn,
            ),
            secondary_y=False,
        )
        # Second subgraph of bin mean for given class
        fig_bar.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=bin_mean_df[y],
                name="bin_avg_for_" + resp_col,
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
                y=[df[y].mean()] * len(bin_mean_df[x]),
                name=pred_colmn + "_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig_bar.update_layout(
            title_text="Mean_response_plot_for_"
            + pred_colmn
            + "_and_"
            + (resp_col.lower()),
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
                title=dict(text=pred_colmn + "_bins"),
            ),
        )
        file_name = (
            fp.GLOBAL_PATH + "/" + "cat_" + resp_col + "_cat_" + pred_colmn + ".html"
        )
        fig_bar.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {
            "col_name": pred_colmn,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

        # input colmn_nm and class_nm

    @staticmethod
    def morp_cont_resp_cont_pred(df, pred_colmn, pred_type, resp_col):
        """

        :param df:
        :param pred_colmn:
        :param pred_type:
        :param resp_col:
        :return:
        """
        x = pred_colmn + "_bin"
        y = resp_col
        df[x] = (pd.cut(df[pred_colmn], bins=10, right=True)).apply(lambda x: x.mid)
        # findin bin mean for given class
        hist_df = df[x].value_counts().to_frame().reset_index()
        bin_mean_df = df[[x, y]].groupby(x).mean().reset_index()

        unweighted_morp, weighted_morp = MorpPlots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        fig_bar = go.Figure()
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

        # first subgraph of bins and their counts
        fig_bar.add_trace(
            go.Bar(
                x=hist_df["index"],
                y=hist_df[x],
                name=pred_colmn,
            ),
            secondary_y=False,
        )
        # Second subgraph of bin mean for given class
        fig_bar.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=bin_mean_df[y],
                name="bin_avg_for_" + resp_col,
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
                y=[df[y].mean()] * len(bin_mean_df[x]),
                name=pred_colmn + "_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig_bar.update_layout(
            title_text="Mean_response_plot_for_"
            + pred_colmn
            + "_and_"
            + (resp_col.lower()),
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
                title=dict(text=pred_colmn + "_bins"),
            ),
        )
        file_name = (
            fp.GLOBAL_PATH + "/" + "cont_" + resp_col + "_cont_" + pred_colmn + ".html"
        )
        fig_bar.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {
            "col_name": pred_colmn,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }

    @staticmethod
    def morp_cont_resp_cat_pred(df, pred_colmn, pred_type, resp_col):
        """

        :param df:
        :param pred_colmn:
        :param pred_type:
        :param resp_col:
        :return:
        """
        x = pred_colmn
        y = resp_col

        # findin bin mean for given class
        hist_df = df[x].value_counts().to_frame().reset_index().sort_values("index")
        bin_mean_df = df[[x, y]].groupby(x).mean().reset_index().sort_values(pred_colmn)

        fig_bar = go.Figure()
        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])

        unweighted_morp, weighted_morp = MorpPlots.weighted_and_unweighted_morp(
            bin_mean_df, hist_df, df, x, y, resp_col
        )

        # first subgraph of bins and their counts
        fig_bar.add_trace(
            go.Bar(
                x=hist_df["index"],
                y=hist_df[x],
                name=pred_colmn,
            ),
            secondary_y=False,
        )
        # Second subgraph of bin mean for given class
        fig_bar.add_trace(
            go.Scatter(
                x=bin_mean_df[x],
                y=bin_mean_df[y],
                name="bin_avg_for_" + resp_col,
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
                y=[df[y].mean()] * len(bin_mean_df[x]),
                name=pred_colmn + "_avg",
                mode="lines",
            ),
            secondary_y=True,
        )
        # updating layout
        fig_bar.update_layout(
            title_text="Mean_response_plot_for_"
            + pred_colmn
            + "_and_"
            + (resp_col.lower()),
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
                title=dict(text=pred_colmn + "_bins"),
            ),
        )
        file_name = (
            fp.GLOBAL_PATH + "/" + "cont_" + resp_col + "_cat_" + pred_colmn + ".html"
        )
        fig_bar.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {
            "col_name": pred_colmn,
            "col_type": pred_type,
            "weighted_morp": weighted_morp,
            "unweighted_morp": unweighted_morp,
            "Plot_link": file_name,
        }
