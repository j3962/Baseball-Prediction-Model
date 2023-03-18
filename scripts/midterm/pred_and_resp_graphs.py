import File_path as fp
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


class PlotGraph:
    @staticmethod
    def cat_resp_cat_pred(data_set, pred_col, resp_col):
        """

        :param pred_col:
        :param resp_col:
        :return:
        """
        pivoted_data = data_set.pivot_table(
            index=resp_col, columns=pred_col, aggfunc="size"
        )

        heatmap = go.Heatmap(
            x=pivoted_data.columns,
            y=pivoted_data.index,
            z=pivoted_data,
            colorscale="Blues",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title="Heatmap", xaxis=dict(title=pred_col), yaxis=dict(title=resp_col)
        )

        # Create the figure
        fig = go.Figure(data=[heatmap], layout=layout)
        file_name = (
            fp.GLOBAL_PATH
            + "/"
            + "cat_"
            + resp_col
            + "_cat_"
            + pred_col
            + "_heatmap"
            + ".html"
        )
        # Show the plot
        fig.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}

    @staticmethod
    def cat_resp_cont_pred(data_set, pred_col, resp_col):
        """

        :param data_set:
        :param pred_col:
        :param resp_col:
        :return:
        """
        # Group data together

        hist_data = [
            data_set[data_set[resp_col] == i][pred_col]
            for i in data_set[resp_col].unique()
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
        file_name_1 = (
            fp.GLOBAL_PATH
            + "/"
            + "cat_"
            + resp_col
            + "_cont_"
            + pred_col
            + "_distplot"
            + ".html"
        )
        fig_1.write_html(
            file=file_name_1,
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
        file_name_2 = (
            fp.GLOBAL_PATH
            + "/"
            + "cat_"
            + resp_col
            + "_cont_"
            + pred_col
            + "_violin_plot"
            + ".html"
        )
        fig_2.write_html(
            file=file_name_2,
            include_plotlyjs="cdn",
        )

        return {
            "column_nm": pred_col,
            "plot_link_1": file_name_1,
            "plot_link_2": file_name_2,
        }

    @staticmethod
    def cont_resp_cat_pred(data_set, pred_col, resp_col):
        """

        :param data_set:
        :param pred_col:
        :param resp_col:
        :return:
        """
        # Group data together
        hist_data = [
            data_set[data_set[pred_col] == i][resp_col]
            for i in data_set[pred_col].unique()
        ]

        group_labels = data_set[pred_col].unique()

        # Create distribution plot with custom bin_size
        fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_1.update_layout(
            title="Continuous " + resp_col + " vs " + " Categorical " + pred_col,
            xaxis_title=resp_col,
            yaxis_title="Distribution",
        )
        file_name_1 = (
            fp.GLOBAL_PATH
            + "/"
            + "cont_"
            + resp_col
            + "_cat_"
            + pred_col
            + "_distplot"
            + ".html"
        )
        fig_1.write_html(
            file=file_name_1,
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
        file_name_2 = (
            fp.GLOBAL_PATH
            + "/"
            + "cont_"
            + resp_col
            + "_cat_"
            + pred_col
            + "_violin_plot"
            + ".html"
        )
        fig_2.write_html(
            file=file_name_2,
            include_plotlyjs="cdn",
        )
        return {
            "column_nm": pred_col,
            "plot_link_1": file_name_1,
            "plot_link_2": file_name_2,
        }

    @staticmethod
    def cont_resp_cont_pred(data_set, pred_col, resp_col):
        """

        :param data_set:
        :param pred_col:
        :param resp_col:
        :return:
        """
        fig = px.scatter(x=data_set[pred_col], y=data_set[resp_col], trendline="ols")
        fig.update_layout(
            title="Continuous " + resp_col + " vs " + " Continuous " + pred_col,
            xaxis_title=pred_col,
            yaxis_title=resp_col,
        )
        file_name = (
            fp.GLOBAL_PATH
            + "/"
            + "cont_"
            + resp_col
            + "_cont_"
            + pred_col
            + "_scatter_plot"
            + ".html"
        )
        fig.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        return {"column_nm": pred_col, "plot_link_1": file_name, "plot_link_2": None}
