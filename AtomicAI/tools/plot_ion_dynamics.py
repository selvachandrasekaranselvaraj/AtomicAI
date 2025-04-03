import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

def process_and_plot_two_xyz_files(filepath1, filepath2):
    """
    Reads two XYZ files, processes position data, and creates subplots with:
    - (a), (b), etc. labels in top-left corners
    - Two consistent colors across all subplots
    - Shared x-axis and proper units
    """

    def process_file(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None

    df1 = process_file(filepath1)
    df2 = process_file(filepath2)

    if df1 is None or df2 is None:
        return

    atom_types = df1['atom'].unique()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)'][:len(atom_types)]  # Dynamic labels
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange

    fig = make_subplots(
        rows=len(atom_types), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03  # Adjust spacing between subplots
    )

    font_size = 20

    # Define axis details
    x_axis_details = {
        'gridcolor': 'lightgray',
        'griddash': 'dash',
        'showline': True,
        'linecolor': 'black',
        'linewidth': 2,
        'mirror': True,
        'titlefont': {'size': font_size + 3, 'color': 'black'},
        'ticks': 'inside',
        'tickwidth': 2,
        'ticklen': 10,
        'minor': {'ticks': 'inside', 'ticklen': 5, 'tickwidth': 1, 'tickcolor': 'black', 'showgrid': False},
        'nticks': 5,  # Maximum number of ticks
        #'dtick': 0.5,  # Step between ticks
    }

    y_axis_details = {
        'gridcolor': 'lightgray',
        'griddash': 'dash',
        'showline': True,
        'linecolor': 'black',
        'linewidth': 2,
        'titlefont': {'size': font_size + 3, 'color': 'black'},
        'mirror': True,
        'ticks': 'inside',
        'tickwidth': 2,
        'ticklen': 10,
        'minor': {'ticks': 'inside', 'ticklen': 5, 'tickwidth': 1, 'tickcolor': 'black', 'showgrid': False, 'nticks': 2},
        'nticks': 3,  # Maximum number of ticks
        #'dtick': 0.5,  # Step between ticks
    }


    for i, atom_type in enumerate(atom_types):
        atom_df1 = df1[df1['atom'] == atom_type].copy()
        atom_df2 = df2[df2['atom'] == atom_type].copy()

        # Trim data and calculate time axis
        def process_df(df, trim=10000):
            if len(df) > trim:
                df = df.iloc[:trim]
            df = df.reset_index(drop=True)
            df['time_ns'] = df.index * 250 / 1000000  # Convert to nanoseconds
            return df

        atom_df1 = process_df(atom_df1)
        atom_df2 = process_df(atom_df2)

        # Calculate position norms
        def calculate_norms(df):
            initial = df[['x', 'y', 'z']].iloc[0]
            df[['dx', 'dy', 'dz']] = df[['x', 'y', 'z']] - initial
            df['d_angstrom'] = np.linalg.norm(df[['dx', 'dy', 'dz']], axis=1)
            return df

        atom_df1 = calculate_norms(atom_df1)
        atom_df2 = calculate_norms(atom_df2)

        # Adjust y-values if max displacement is less than 2.0 Å
        y1_max = atom_df1['d_angstrom'].max()
        y2_max = atom_df2['d_angstrom'].max()
        if 2 < max([y1_max, y2_max]):
           y_max = max([y1_max, y2_max]) + max([y1_max, y2_max])*0.1
        else:
           y_max = 1.9

        #if y1_max < 2.0:
        #    adjustment_1 = y1_max * 0.5
        #    atom_df1['d'] = atom_df1['d_angstrom'] - adjustment_1
        #else:
        #    atom_df1['d'] = atom_df1['d_angstrom']
        #
        #if y2_max < 2.0:
        #    adjustment_2 = y2_max * 0.5
        #    atom_df2['d'] = atom_df2['d_angstrom'] - adjustment_2
        #else:
        #    atom_df2['d'] = atom_df2['d_angstrom']

        # Add traces for both datasets
        fig.add_trace(go.Scatter(
            x=atom_df1['time_ns'],
            y=atom_df1['d_angstrom'],
            mode='lines',
            name='U=0.0eV',
            line=dict(color=colors[0], width=1),
            showlegend=(i == 0)  # Only show legend once
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=atom_df2['time_ns'],
            y=atom_df2['d_angstrom'],
            mode='lines',
            name='U=3.0eV',
            line=dict(color=colors[1], width=1),
            showlegend=(i == 0)  # Only show legend once
        ), row=i+1, col=1)


        # Add subplot label (a), (b), etc.
        fig.add_annotation(
            x=0.2,
            y=y_max*0.8,
            xref='paper',  # Use 'paper' to reference the figure's coordinate system
            yref='paper',
            text=f"{subplot_labels[i]}  {atom_type}",
            showarrow=False,
            font=dict(size=font_size + 5, color='black', family='Courier',),
            align="left",
            row=i+1,
            col=1
        )
        fig.update_layout({f"xaxis{i + 1}": x_axis_details, f"yaxis{i + 1}": y_axis_details})

        # Set y-axis for each subplot
        fig.update_yaxes(title_text='d(Å)', row=i+1, col=1, range=[0, y_max])

    # Update x-axis for the last subplot only
    fig.update_xaxes(showticklabels=True, row=len(atom_types), col=1, title_text='Simulation Time (ns)', title_standoff=10)

    # Update global layout
    fig.update_layout(
        height=120 * len(atom_types),
        width=1000,
        #margin=dict(t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            font_size=font_size + 3,
            font_color='black',
            font_family='Courier',
            bgcolor='rgba(0,0,0,0)'
        ),

        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin={'l': 0.5, 'r': 0.9, 'b': 0.5, 't': 0.9},
        font_size=font_size,
        font_color='black',
        font_family='Courier',
    )

    fig.write_image('ion_dynamics.png',
                    scale=3)
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one or two XYZ files and plot atomic displacements.")
    parser.add_argument("filepaths", nargs='+', help="Path(s) to XYZ file(s). Provide 1 or 2 files.")
    args = parser.parse_args()

    # Validate number of files
    if len(args.filepaths) not in [1, 2]:
        print("Error: Please provide either 1 or 2 input files.")
        exit(1)
        
    # If only 1 file provided, duplicate it for comparison
    if len(args.filepaths) == 1:
        args.filepaths.append(args.filepaths[0])
    
    process_and_plot_two_xyz_files(args.filepaths[0], args.filepaths[1])

