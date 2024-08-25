
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import sys, os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import r2_score

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import glob
import subprocess

#from plotly.offline import init_notebook_mode, iplot

#init_notebook_mode(connected=True)

import os

def check_files():
    """
    Checks if necessary files exist in the specified directories.

    Returns:
        file_force (str): Path to the force data file.
        file_energy (str): Path to the energy data file.
        atomic_indices_file (str): Path to the atomic indices file.
        atoms_file (str): Path to the atoms file.
    """
    # Specify the directory where you want to check for the file
    directory_path = './'

    # Check if the file exists in the specified directory
    file_force = os.path.join(directory_path, 'results.f.out')
    file_energy = os.path.join(directory_path, 'results.e.out')
    for file_ in [file_force, file_energy]:
        if not os.path.exists(file_):
            print(f"The file '{os.path.basename(file_)}' does not exist in the directory: {directory_path}")
            exit()

    # Specify the directory where you want to check for the file
    directory_path = './'

    # Check if the file exists in the specified directory
    atoms = os.path.join(directory_path, 'type.raw')
    atomic_indices = os.path.join(directory_path, 'type_map.raw')
    for file_ in [atoms, atomic_indices]:
        if not os.path.exists(file_):
            directory_path = "../00.data/training_data/"
            atoms = os.path.join(directory_path, 'type.raw')
            atomic_indices = os.path.join(directory_path, 'type_map.raw')
            for file_ in [atoms, atomic_indices]:
                if not os.path.exists(file_):
                    print(f"The file '{os.path.basename(file_)}' does not exist in the directory: {directory_path}")
                    exit()

    # File paths
    atomic_indices_file = os.path.join(directory_path, 'type.raw')
    atoms_file = os.path.join(directory_path, 'type_map.raw')

    return file_force, file_energy, atomic_indices_file, atoms_file


def n_col_row(n_atoms_type):
    """
    Determines the number of columns and rows for subplots based on the number of atom types.

    Args:
        n_atoms_type (int): Number of atom types.

    Returns:
        n_columns (int): Number of columns for subplots.
        n_rows (int): Number of rows for subplots.
    """
    # Set default values for n_columns and n_rows
    n_columns = 2
    n_rows = 1

    # Determine the number of columns and rows based on the number of atom types
    
    if n_atoms_type == 1:
        n_columns = 2
        n_rows = 1
    elif n_atoms_type == 2:
        n_columns = 3
        n_rows = 1
    elif n_atoms_type == 3:
        n_columns = 2
        n_rows = 2
    elif n_atoms_type in [4, 5]:
        n_columns = 3
        n_rows = 2
    elif n_atoms_type in [6, 7]:
        n_columns = 4
        n_rows = 2
    elif n_atoms_type == 8:
        n_columns = 3
        n_rows = 3
    elif n_atoms_type in [9, 10, 11]:
        n_columns = 4
        n_rows = 3
    elif n_atoms_type == [12, 13]:
        n_columns = 3
        n_rows = 5
    elif n_atoms_type == [14, 15, 16]:
        n_columns = 3
        n_rows = 6

    return n_columns, n_rows

def plot_deepPot_accuracy():
    # Check if necessary files exist and retrieve their paths
    file_force, file_energy, atomic_indices_file, atoms_file = check_files()

    # Initialize a dictionary to store the mapping of atom indices to names
    atom_index_map = {}

    # Read atoms and indices from files and create the mapping
    with open(atoms_file, 'r') as atoms_f, open(atomic_indices_file,
                                                'r') as indices_f:
        atoms_ = atoms_f.read().splitlines()
        indices = indices_f.read().splitlines()

        # Create the mapping of indices to atoms
        atom_index_map = {index: atom for index, atom in zip(indices, atoms_)}

    # Create a dictionary to map indices to atoms
    index_atom_map = {str(i): atom for i, atom in enumerate(atoms_)}

    # Replace indices with corresponding atoms
    atoms = [index_atom_map[index] for index in indices]

    # Read forces and energies from files
    data_f = pd.read_csv(file_force, header=None, sep=' ', skiprows=1)
    data_e = pd.read_csv(file_energy, header=None, sep=' ', skiprows=1)

    # Define column names for forces and energies
    x_name = [
        'DFT F<sub>x</sub> (eV/Å)', 'DFT F<sub>y</sub> (eV/Å)',
        'DFT F<sub>z</sub> (eV/Å)'
    ]
    y_name = [
        'DLP F<sub>x</sub> (eV/Å)', 'DLP F<sub>y</sub> (eV/Å)',
        'DLP F<sub>z</sub> (eV/Å)'
    ]
    legends = ['F<sub>x</sub>', 'F<sub>y</sub>', 'F<sub>z</sub>']

    # Rename columns in force and energy data frames
    data_f.columns = x_name + y_name
    data_e.columns = ['DFT Energy (eV)', 'DLP Energy (eV)']

    # Repeat atom labels to match the length of the force data
    data_f['atoms'] = atoms * int(len(data_f) / len(atoms))

    # Determine the minimum and maximum values for force data
    data_force_ = data_f.drop(columns=['atoms'])
    fmin = min(np.array(data_force_.min())) * 0.9
    fmax = max(np.array(data_force_.max())) * 0.9
    frange = [-max(-fmin, fmax), max(-fmin, fmax)]

    # Determine the number of rows and columns for subplots
    n_atoms_type = len(atoms_)
    n_columns, n_rows = n_col_row(n_atoms_type)

    # Create subplots
    fig = make_subplots(rows=n_rows,
                        cols=n_columns,
                        shared_xaxes=False,
                        vertical_spacing=0.45 / n_rows,
                        horizontal_spacing=0.45 / n_columns)

    # Define colors for plots
    colors = ['green', 'red', 'blue']
    n_atoms_ = 0

    # Open a file to write R2 scores
    f = open('r2.txt', 'w')

    # Iterate over each subplot
    for n_row in range(1, n_rows + 1):
        for n_col in range(1, n_columns + 1):
            if n_row == 1 and n_col == 1:
                # Plot energy data
                r2_e = str(
                    round(
                        r2_score(data_e['DFT Energy (eV)'],
                                 data_e['DLP Energy (eV)']), 2))
                f.write(f"Energy R2 {r2_e} \n")
                e0 = min(data_e.min())
                x_e = data_e['DFT Energy (eV)'] - e0
                y_e = data_e['DLP Energy (eV)'] - e0
                fig.append_trace(
                    go.Scatter(
                        x=x_e,
                        y=y_e,
                        mode="markers",
                        marker=dict(color=colors[0], size=7),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                e_d = min(x_e) - max(y_e)
                e_x_range = [min(y_e) + (e_d * 0.1), max(y_e) - (e_d * 0.1)]
                e_min_x = min(e_x_range)
                e_max_x = max(e_x_range)
                fig.append_trace(
                    go.Scatter(
                        x=e_x_range,
                        y=e_x_range,
                        mode="lines",
                        line=dict(dash='dot', width=1),
                        marker=dict(color='black', size=0),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            elif n_atoms_type > n_atoms_:
                if n_row == 1 and n_col == 2:
                    showlegend = True
                else:
                    showlegend = False

                # Plot force data
                data_force = data_f[data_f['atoms'] == atoms_[n_atoms_]].drop(
                    columns=['atoms'])

                for column_i, (column_x,
                               column_y) in enumerate(zip(x_name, y_name)):
                    y_pred = data_force[column_y].astype(float)
                    y_true = data_force[column_x].astype(float)
                    r2 = str(round(r2_score(y_true, y_pred), 2))
                    f.write(f"Force R2 {column_y} {atoms_[n_atoms_]} {r2} \n")

                    fig.append_trace(
                        go.Scatter(
                            x=y_true,
                            y=y_pred,
                            mode="markers",
                            name=legends[column_i],
                            marker=dict(color=colors[column_i], size=5),
                            showlegend=showlegend,
                        ),
                        row=n_row,
                        col=n_col,
                    )

                    fig.append_trace(
                        go.Scatter(
                            x=frange,
                            y=frange,
                            mode="lines",
                            line=dict(dash='dot', width=1),
                            marker=dict(color='black', size=0),
                            showlegend=False,
                        ),
                        row=n_row,
                        col=n_col,
                    )

                n_atoms_ += 1

    f.close()

    # Update layout for the plots
    font_size = 18
    x_axis_details = {
        'gridcolor': 'lightgray',
        'griddash': 'dash',
        'showline': True,
        'linecolor': 'black',
        'linewidth': 2,
        'mirror': True,
        'titlefont': {
            'size': font_size + 2,
            'color': 'black'
        },
        'ticks': 'inside',
        'tickwidth': 2,
        'ticklen': 10,
        'minor': {
            'ticks': 'inside',
            'ticklen': 5,
            'tickwidth': 1,
            'tickcolor': 'black',
            'showgrid': False
        },
    }

    y_axis_details = {
        'gridcolor': 'lightgray',
        'griddash': 'dash',
        'showline': True,
        'linecolor': 'black',
        'linewidth': 2,
        'titlefont': {
            'size': font_size + 2,
            'color': 'black'
        },
        'mirror': True,
        'ticks': 'inside',
        'tickwidth': 2,
        'ticklen': 10,
        'minor': {
            'ticks': 'inside',
            'ticklen': 5,
            'tickwidth': 1,
            'tickcolor': 'black',
            'showgrid': False
        },
    }

    if n_rows == 1:
        legend_fix_y = 1
    else:
        legend_fix_y = (1 / n_rows) * 0.3

    fig.update_layout(
        xaxis=x_axis_details,
        xaxis2=x_axis_details,
        xaxis3=x_axis_details,
        xaxis4=x_axis_details,
        xaxis5=y_axis_details,
        xaxis6=y_axis_details,
        xaxis7=y_axis_details,
        xaxis8=y_axis_details,
        xaxis9=y_axis_details,
        xaxis10=y_axis_details,
        xaxis11=y_axis_details,
        xaxis12=y_axis_details,
        yaxis=y_axis_details,
        yaxis2=y_axis_details,
        yaxis3=y_axis_details,
        yaxis4=y_axis_details,
        yaxis5=y_axis_details,
        yaxis6=y_axis_details,
        yaxis7=y_axis_details,
        yaxis8=y_axis_details,
        yaxis9=y_axis_details,
        yaxis10=y_axis_details,
        yaxis11=y_axis_details,
        yaxis12=y_axis_details,
        showlegend=True,
        legend={
            'orientation': 'v',
            'yref': 'paper',
            'xref': 'paper',
            'yanchor': 'bottom',
            'xanchor': 'right',
            #'x': (1 / n_columns) * (n_columns - 1) + (1 / n_columns)*0.25,
            'x': 1 - (1 / n_columns)*0.01,
            'y': (1 / n_rows)*0.01,
            'font_size': font_size + 2,
            'bgcolor':'rgba(0,0,0,0)'
        },
        plot_bgcolor='rgba(0,0,0,0)', #'white',
        paper_bgcolor='rgba(0,0,0,0)', #'white',
        margin={
            'l': 0.5,
            'r': 0.9,
            'b': 0.5,
            't': 0.9
        },
        font={
            'size': font_size,
            'color': 'black'
        },
    )

    fig.update_layout(legend={'itemsizing': 'constant'})

    height = 250 * n_rows
    width = 250 * n_columns

    flabel = [
        '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
        '(k)', '(l)', '(m)', '(n)'
    ]
    n_flabel = 0
    n_atoms_ = 0
    n_plot = 0
    # Update x-axis and y-axis labels
    for n_row in range(1, n_rows + 1):
        for n_col in range(1, n_columns + 1):
            if n_col == 1:
                title_standoff = 12
            else:
                title_standoff = 2

            n_plot += 1
            #print(f"Subplot{n_plot} is writting for {atoms_[n_atoms_]}...")
            if n_row == 1 and n_col == 1:
                fig.update_xaxes(title_text="E<sub>DFT</sub> (eV)",
                                 range=[e_min_x, e_max_x],
                                 title_standoff=10,
                                 row=1,
                                 col=1)
                fig.update_yaxes(title_text="E<sub>DLP</sub> (eV)",
                                 range=[e_min_x, e_max_x],
                                 title_standoff=10,
                                 row=1,
                                 col=1)
                fig.add_annotation(text=flabel[n_flabel],
                                   yanchor="top",
                                   xanchor="left",
                                   xref="paper",
                                   yref="paper",
                                   x=e_min_x,
                                   y=e_max_x,
                                   showarrow=False,
                                   font_size=font_size + 2,
                                   row=1,
                                   col=1)
            elif n_atoms_type > n_atoms_:

                data_force = data_f[data_f['atoms'] == atoms_[n_atoms_]].drop(
                    columns=['atoms'])

                y_pred = data_force[column_y].astype(float)
                y_true = data_force[column_x].astype(float)
               
                _min = -min(np.array([min(y_true), min(y_pred)]))
                _max = max(np.array([max(y_true), max(y_pred)]))
                __max = max(np.array([_min, _max]))*1.4
                #print(__max)
                frange = np.array([-__max, __max]) 

                fig.update_xaxes(title_text="F<sub>DFT</sub> (eV/Å)",
                                 range= frange, #[f_min_x, f_max_x],
                                 #scaleanchor = "y2",
                                 scaleratio = 1,
                                 title_standoff=10,
                                 row=n_row,
                                 col=n_col)
                fig.update_yaxes(title_text="F<sub>DLP</sub> (eV/Å)",
                                 range= frange, #[f_min_x, f_max_x],
                                 #scaleanchor = "x2",
                                 scaleratio = 1,
                                 title_standoff=0,
                                 row=n_row,
                                 col=n_col)
                fig.add_annotation(text=f"{flabel[n_flabel]} {atoms_[n_atoms_]}" ,
                                   yanchor="top",
                                   xanchor="left",
                                   xref="paper",
                                   yref="paper",
                                   font_size=font_size + 0,
                                   x=frange[0] * 0.95,
                                   y=frange[1] * 0.95,
                                   showarrow=False,
                                   row=n_row,
                                   col=n_col)
                # fig.add_annotation(text=atoms_[n_atoms_],
                #                  xref="paper",
                #                  yref="paper",
                #                  font_size=font_size + 1,
                #                  x=f_max_x * 0.8,
                #                  y=f_min_x * 0.8,
                #                  showarrow=False,
                #                  row=n_row,
                #                  col=n_col)
                n_atoms_ += 1

            n_flabel += 1

    # Write the plots to HTML, SVG, and PNG files
    #print("Writting HTML image")
    #fig.write_html('force_accuracy.html')
    #print("Writing SVG image")
    #fig.write_image('force_accuracy.svg',
    #                scale=0.9,
    #                width=width,
    #                height=height)
    print("Writing PNG image")
    #fig.write_image('force_accuracy.png')
    fig.write_image('force_accuracy.png',
                    scale=0.9,
                    width=width,
                    height=height)

    print("All DONE.")
    return
