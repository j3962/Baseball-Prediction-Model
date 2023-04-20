import warnings

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from scipy import stats
from scipy.stats import pearsonr


class PredsCorrelation:
    @staticmethod
    def cont_pred_corr(df1, df2):
        corr, _ = pearsonr(df1, df2)
        return corr

    @staticmethod
    def fill_na(data):
        if isinstance(data, pd.Series):
            return data.fillna(0)
        else:
            return np.array([value if value is not None else 0 for value in data])

    @staticmethod
    def cat_correlation(x, y, bias_correction=True, tschuprow=False):
        """
        Calculates correlation statistic for categorical-categorical association.
        The two measures supported are:
        1. Cramer'V ( default )
        2. Tschuprow'T

        SOURCES:
        1.) CODE: https://github.com/MavericksDS/pycorr
        2.) Used logic from:
            https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
            to ignore yates correction factor on 2x2
        3.) Haven't validated Tschuprow

        Bias correction and formula's taken from :
        https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

        Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
        Parameters:
        -----------
        x : list / ndarray / Pandas Series
            A sequence of categorical measurements
        y : list / NumPy ndarray / Pandas Series
            A sequence of categorical measurements
        bias_correction : Boolean, default = True
        tschuprow : Boolean, default = False
                   For choosing Tschuprow as measure
        Returns:
        --------
        float in the range of [0,1]
        """
        corr_coeff = np.nan
        try:
            x, y = PredsCorrelation.fill_na(x), PredsCorrelation.fill_na(y)
            crosstab_matrix = pd.crosstab(x, y)
            n_observations = crosstab_matrix.sum().sum()

            yates_correct = True
            if bias_correction:
                if crosstab_matrix.shape == (2, 2):
                    yates_correct = False

            chi2, _, _, _ = stats.chi2_contingency(
                crosstab_matrix, correction=yates_correct
            )
            phi2 = chi2 / n_observations

            # r and c are number of categories of x and y
            r, c = crosstab_matrix.shape
            if bias_correction:
                phi2_corrected = max(
                    0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
                )
                r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
                c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
                if tschuprow:
                    corr_coeff = np.sqrt(
                        phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                    )
                    return corr_coeff
                corr_coeff = np.sqrt(
                    phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
                )
                return corr_coeff
            if tschuprow:
                corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
                return corr_coeff
            corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
            return corr_coeff
        except Exception as ex:
            print(ex)
            if tschuprow:
                warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
            else:
                warnings.warn("Error calculating Cramer's V", RuntimeWarning)
            return corr_coeff

    @staticmethod
    def cat_cont_correlation_ratio(categories, values):
        """
        Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
        SOURCE:
        1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        :param categories: Numpy array of categories
        :param values: Numpy array of values
        :return: correlation
        """
        f_cat, _ = pd.factorize(categories)
        cat_num = np.max(f_cat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = values[np.argwhere(f_cat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(
            np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
        )
        denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        return eta

    @staticmethod
    def corr_heatmap_plots(df, pred_col_1, pred_col_2, value_col):

        data = [
            go.Heatmap(
                z=np.array(df[value_col]),
                x=np.array(df[pred_col_1]),
                y=np.array(df[pred_col_2]),
                colorscale="YlGnBu",
                text=np.around(df[value_col].values, 6),
                texttemplate="%{text}",
            )
        ]
        layout = go.Layout(
            {
                "title": "Heatmap of Correlation ",
            }
        )

        fig_heatmap = go.Figure(data=data, layout=layout)
        return pio.to_html(fig_heatmap, include_plotlyjs="cdn")
