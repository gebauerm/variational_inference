from typing import List
import plotly.express as px
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import seaborn as sns


def plot_elbo(values: List, log_vals=False, seaborn=False):

    if log_vals:
        values = [-np.log(np.abs(value)) for value in values]
    values = pd.DataFrame({
        "Iteration": list(range(len(values))),
        "ELBO":values})
    if seaborn:
        fig = sns.lineplot(data=values, x="Iteration", y="ELBO")
    else:
        fig = px.line(values, x="Iteration", y="ELBO", title='Variational Inference Convergence')
    return fig


def plot_distributions(x, targets, mu_values=None):
    plot_dict = {"x" : x,
                 "distribution": targets
                 }
    plot_df = pd.DataFrame.from_dict(plot_dict)
    val_range = np.arange(int(round(plot_df["x"].min())), int(round(plot_df["x"].max()+1)), 0.2)
    fig = px.histogram(plot_df, x="x", color="distribution", nbins=100, opacity=0.7, histnorm='probability density')
    traces = [go.Scatter(x = val_range,
                         y = stats.norm(loc=mu_values[idx], scale=1).pdf(val_range),
                         mode="lines",
                         name=f"estimated_dist_{idx}") for idx, _ in enumerate(plot_df["distribution"].unique().tolist())]
    for trace in traces:
        fig.add_trace(trace)
    return fig
