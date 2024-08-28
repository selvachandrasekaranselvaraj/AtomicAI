import os
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_vasp_md():
    if not os.path.isfile('./OSZICAR'):
        print('OSZICAR file is not available in this directory')
        exit()

    g1 = ''.join([f"${i}\"  \"" for i in range(1, 20, 2)])

    os.system("grep \"T=\" OSZICAR|awk \'{print " + g1 + "}\'>T_E.txt")
    df_ = pd.read_csv('./T_E.txt',
                      sep="  ",
                      header=None,
                      index_col=None,
                      engine='python').tail(-10)
    n_columns = df_.shape[1]
    if n_columns == 8:
        df_.columns = ['MD_steps', 'T(k)', 'E (eV)', 'F', 'E0 (eV)', 'EK', 'SP', 'SK']
        y_labels = ['T(k)', 'E (eV)', 'F', 'E0 (eV)', 'EK', 'SP', 'SK']
    elif n_columns == 9:
        df_.columns = ['MD_steps', 'T(k)', 'E (eV)', 'F', 'E0 (eV)', 'EK', 'SP', 'SK', 'Spin moment']
        y_labels = ['T(k)', 'E (eV)', 'F', 'E0 (eV)', 'EK', 'SP', 'SK', 'Spin moment']

    fig = make_subplots(rows=len(y_labels), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for i, y_label in enumerate(y_labels):
        fig.append_trace(
            go.Scatter(
                x=df_['MD_steps'],
                y=df_[y_label],
            ),
            row=i+1,
            col=1,
        )
        fig.update_yaxes(title_text=y_label, row=i+1, col=1)

    fig.update_xaxes(title_text="MD steps", row=len(y_labels), col=1)

    fig.update_layout(height=1000, width=800, title_text="MD plots", showlegend=False)
    fig.write_html('md_plots.html')
    fig.write_image("md_plots.png")
    return
