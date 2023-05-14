import File_path as fp
import numpy as np
import plotly.graph_objects as go


class Morp2dPlots:
    @staticmethod
    def cat_cat_2d_morp(df_ip, x1, x2, y):
        df = df_ip

        df = df[[x1, x2, y]].groupby([x1, x2]).agg(["mean", "size"]).reset_index()
        df.columns = df.columns.to_flat_index().map("".join)

        df["unweighted_morp"] = (
            df[y + "mean"].to_frame().apply(lambda x: (df_ip[y].mean() - x) ** 2)
        )
        df["weighted_morp"] = df.apply(
            lambda a: (a[y + "size"] / df[y + "size"].sum()) * a["unweighted_morp"],
            axis=1,
        )

        df["mean_size"] = df.apply(
            lambda a: "{:.6f} pop:{}".format(a[y + "mean"], a[y + "size"]), axis=1
        )

        data = [
            go.Heatmap(
                z=np.array(df[y + "mean"]),
                x=np.array(df[x2]),
                y=np.array(df[x1]),
                colorscale="YlGnBu",
                text=np.array(df["mean_size"]),
                texttemplate="%{text}",
                colorbar=dict(title="Correlation"),
            )
        ]
        layout = go.Layout(
            {
                "title": x2.replace("_bin", "") + " vs " + x1.replace("_bin", ""),
                "xaxis": dict(title=x2.replace("_bin", "")),
                "yaxis": dict(title=x1.replace("_bin", "")),
            }
        )

        fig_heatmap = go.Figure(data=data, layout=layout)
        file_name = (
            fp.GLOBAL_PATH_2D_MORP + "/" + "cat_" + x1 + "_cat_" + x2 + "_2D_morp.html"
        )
        fig_heatmap.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        # fig_heatmap.show()
        return {
            "Weighted_morp": df["weighted_morp"].sum(),
            "Unweighted_morp": df["unweighted_morp"].sum()
            / (df_ip[x1].nunique() * df_ip[x2].nunique()),
            "Plot_link": file_name,
        }

    # @title cat_cont_2d_morp at B stg

    @staticmethod
    def cat_cont_2d_morp(df_ip, x1, x2, y):
        df = df_ip
        x2 = x2 + "_bin"

        df = df[[x1, x2, y]].groupby([x1, x2]).agg(["mean", "size"]).reset_index()
        df.columns = df.columns.to_flat_index().map("".join)

        df["unweighted_morp"] = (
            df[y + "mean"].to_frame().apply(lambda x: (df_ip[y].mean() - x) ** 2)
        )

        df["weighted_morp"] = df.apply(
            lambda a: (a[y + "size"] / df[y + "size"].sum()) * a["unweighted_morp"],
            axis=1,
        )

        df["mean_size"] = df.apply(
            lambda a: "{:.6f} pop:{}".format(a[y + "mean"], a[y + "size"]), axis=1
        )

        data = [
            go.Heatmap(
                z=np.array(df[y + "mean"]),
                x=np.array(df[x1]),
                y=np.array(df[x2]),
                colorscale="YlGnBu",
                text=np.array(df["mean_size"]),
                texttemplate="%{text}",
                colorbar=dict(title="Correlation"),
            )
        ]
        layout = go.Layout(
            {
                "title": x2.replace("_bin", "") + " vs " + x1.replace("_bin", ""),
                "xaxis": dict(title=x1.replace("_bin", "")),
                "yaxis": dict(title=x2.replace("_bin", ""), tickvals=np.array(df[x2])),
            }
        )

        fig_heatmap = go.Figure(data=data, layout=layout)
        file_name = (
            fp.GLOBAL_PATH_2D_MORP + "/" + "cat_" + x1 + "_cont_" + x2 + "_2D_morp.html"
        )
        fig_heatmap.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        # fig_heatmap.show()
        return {
            "Weighted_morp": df["weighted_morp"].sum(),
            "Unweighted_morp": df["unweighted_morp"].sum() / len(df),
            "Plot_link": file_name,
        }

    @staticmethod
    def cont_cont_2d_morp(df_ip, x1, x2, y):
        df = df_ip
        x1 = x1 + "_bin"
        x2 = x2 + "_bin"

        df = df[[x1, x2, y]].groupby([x1, x2]).agg(["mean", "size"]).reset_index()
        df.columns = df.columns.to_flat_index().map("".join)

        df["unweighted_morp"] = (
            df[y + "mean"].to_frame().apply(lambda x: (df_ip[y].mean() - x) ** 2)
        )

        df["weighted_morp"] = df.apply(
            lambda a: (a[y + "size"] / df[y + "size"].sum()) * a["unweighted_morp"],
            axis=1,
        )

        df["mean_size"] = df.apply(
            lambda a: "{:.6f} pop:{}".format(a[y + "mean"], a[y + "size"]), axis=1
        )

        data = [
            go.Heatmap(
                z=np.array(df[y + "mean"]),
                x=np.array(df[x1]),
                y=np.array(df[x2]),
                colorscale="YlGnBu",
                text=np.array(df["mean_size"]),
                texttemplate="%{text}",
                colorbar=dict(title="Correlation"),
            )
        ]
        layout = go.Layout(
            {
                "title": x2.replace("_bin", "") + " vs " + x1.replace("_bin", ""),
                "xaxis": dict(title=x1.replace("_bin", ""), tickvals=np.array(df[x1])),
                "yaxis": dict(title=x2.replace("_bin", ""), tickvals=np.array(df[x2])),
            }
        )

        fig_heatmap = go.Figure(data=data, layout=layout)
        file_name = (
            fp.GLOBAL_PATH_2D_MORP
            + "/"
            + "cont_"
            + x1
            + "_cont_"
            + x2
            + "_2D_morp.html"
        )
        fig_heatmap.write_html(
            file=file_name,
            include_plotlyjs="cdn",
        )
        # fig_heatmap.show()
        return {
            "Weighted_morp": df["weighted_morp"].sum(),
            "Unweighted_morp": df["unweighted_morp"].sum() / len(df),
            "Plot_link": file_name,
        }
