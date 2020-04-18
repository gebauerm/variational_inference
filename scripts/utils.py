from typing import List
import plotly.express as px
import pandas as pd

def plot_elbo(values: List):
    values = pd.DataFrame({
        "Iteration": list(range(len(values))),
        "ELBO":values})
    fig = px.line(values, x="Iteration", y="ELBO", title='Variational Inference Convergence')
    return fig