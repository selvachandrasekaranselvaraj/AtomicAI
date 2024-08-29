import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to determine number of columns and rows for subplots
def n_col_row(n_plots):
    """
    Determines the number of columns and rows for subplots based on the number of plots.

    Args:
        n_plots (int): Number of subplots.

    Returns:
        n_rows (int): Number of rows for subplots.
        n_columns (int): Number of columns for subplots.
    """
    # Default values for n_columns and n_rows
    n_columns = 2
    n_rows = 1

    if n_plots == 1:
        n_columns = 2
        n_rows = 1
    elif n_plots == 2:
        n_columns = 2
        n_rows = 1
    elif n_plots == 3:
        n_columns = 2
        n_rows = 2
    elif n_plots == 4:
        n_columns = 2
        n_rows = 2
    elif n_plots in [5, 6]:
        n_columns = 2
        n_rows = 3
    elif n_plots in [7, 8]:
        n_columns = 2
        n_rows = 4
    elif n_plots == 9:
        n_columns = 2
        n_rows = 5
    elif n_plots in [10, 11, 12]:
        n_columns = 2
        n_rows = 5
    elif n_plots in [11, 12]:
        n_columns = 2
        n_rows = 6
    elif n_plots in [13, 14]:
        n_columns = 2
        n_rows = 7
    elif n_plots in [15, 16, 17]:
        n_columns = 3
        n_rows = 6

    return n_rows, n_columns

# Function to decorate plot borders and layout
def decorate_borders(fig, font_size, n_rows, n_columns, y_ranges, y_labels, show_flabels, time_unit):
    """
    Updates the layout and appearance of the plotly figure.

    Args:
        fig (go.Figure): The plotly figure object.
        font_size (int): Font size for labels and legends.
        n_rows (int): Number of rows in subplots.
        n_columns (int): Number of columns in subplots.
        y_ranges (list): Y-axis range values.
        y_labels (list): Y-axis labels.
        show_flabels (bool): Whether to show figure labels.
        time_unit (str): The unit of time for the x-axis labels.

    Returns:
        fig (go.Figure): The updated plotly figure object.
    """
    if n_columns == 3:
        width = 500 * n_columns
        height = 120 * n_rows
        font_size = font_size + 2
    elif n_columns == 2:
        width = 600 * n_columns
        height = 175 * n_rows
    else:
        width = 600 * n_columns
        height = 200

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

    y_axis_details_exp = y_axis_details.copy()
    y_axis_details_exp['tickformat'] = ".0e"

    # Adjust legend position
    legend_fix_y = 1 if n_rows == 1 else (1 / n_rows) * 0.3

    

    # Define a dictionary to map old y-axis labels to new ones
    label_mapping = {
        'Temp': 'T(K)',
        'Press': 'P(bar)',
        'PotEng': 'PE(eV)',
        'KinEng': 'KE(eV)',
        'Volume': 'V(Å<sup>3</sup>)',
        'Cella': 'a(Å)',
        'Cellb': 'b(Å)',
        'Cellc': 'c(Å)',
        'CellAlpha': 'α(<sup>o</sup>)',
        'CellBeta': 'β(<sup>o</sup>)',
        'CellGamma': 'γ(<sup>o</sup>)'
    }
    # Modify y-axis labels
    modified_y_labels = []
    for i_axis, y_label in enumerate(y_labels):
        # Use the dictionary to get the new label
        modified_y_label = label_mapping.get(y_label, y_label)
        
        # Update y_ranges if necessary
        if y_label == 'Volume':
            y_ranges[i_axis] += y_ranges[i_axis] * 0.02
        
        # Append the modified label
        modified_y_labels.append(modified_y_label)
        yaxis_detail = y_axis_details        
        fig.update_layout({f"xaxis{i_axis + 1}": x_axis_details, f"yaxis{i_axis + 1}": yaxis_detail})

    # Update subplot axes
    i_plot = 0
    for n_row in range(1, n_rows + 1):
        for n_col in range(1, n_columns + 1):
            if i_plot >= len(modified_y_labels):
                break
            fig.update_yaxes(title_text=modified_y_labels[i_plot], title_standoff=10, row=n_row, col=n_col)
            if n_row == n_rows:
                fig.update_xaxes(title_text=f"Time ({time_unit})", title_standoff=10, row=n_row, col=n_col)
            i_plot += 1

    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor='bottom',
            y=1.01,
            xanchor="center",
            x=0.5,
            orientation="h",
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
        height=height,
        width=width
    )

    # Add figure labels if required
    if show_flabels:
        flabel = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
        i_label = 0
        for n_row in range(1, n_rows + 1):
            for n_col in range(1, n_columns + 1):
                if i_label >= len(y_ranges):
                    break
                fig.add_annotation(
                    text=f"{flabel[i_label]}",
                    yanchor="top",
                    xanchor="left",
                    xref="paper",
                    yref="paper",
                    font_size=font_size + 5,
                    x=0.0,
                    y=y_ranges[i_label],
                    showarrow=False,
                    row=n_row,
                    col=n_col
                )
                i_label += 1

    return fig

# Function to plot data using Plotly
def plotly_plot(n_files, df, y_labels, fig, file_n, file_name, y_ranges, legend):
    """
    Creates a plotly figure with subplots based on the data provided.

    Args:
        n_files (int): Total number of files.
        df (pd.DataFrame): Dataframe with the data.
        y_labels (list): List of y-axis labels.
        fig (go.Figure): Plotly figure object.
        file_n (int): Current file index.
        file_name (str): Name of the current file.
        font_size (int): Font size for labels and legends.
        y_ranges (list): Y-axis range values.

    Returns:
        fig (go.Figure): The updated plotly figure object.
    """
    show_flabels = (n_files == file_n + 1)
    n_rows, n_columns = n_col_row(len(y_labels))
    if n_columns >= 3:
        font_size = 17
        horizontal_spacing = 0.10
        vertical_spacing = 0.06
    else:
        font_size = 18
        horizontal_spacing = 0.15
        vertical_spacing = 0.04
    
    if file_n == 0:
        fig = make_subplots(
            rows=n_rows,
            cols=n_columns,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing
        )

    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']

    try:
        x = np.array(df['Time']).astype(float) * 1e-12 / 1e-9  # time in ns
    except KeyError:
        x = np.array(df['Step']).astype(float) * 1e-15 / 1e-9  # time in ns

    if max(x) < 0.1:
        time_unit = 'ps'
        x = x * 1e3
    else:
        time_unit = 'ns'

    # Number of elements to select
    n_time = 10000
    if len(x) > n_time:
        interval = len(x) // n_time
        x = x[::interval]
    else:
        interval = 1
        x = x[::interval]

    i_plot = 0
    skip_initial = 0
    for n_row in range(1, n_rows + 1):
        for n_col in range(1, n_columns + 1):
            if i_plot >= len(y_labels):
                break
            y_label = y_labels[i_plot]
            y = np.array(df[y_label].astype(float))[::interval]
            showlegend = i_plot == 0 and legend
            fig.add_trace(
                go.Scatter(
                    x=x[skip_initial:],
                    #y=np.floor(y[skip_initial:]).astype(float),
                    y=np.array(y[skip_initial:]).astype(float),
                    name=file_name[:-4],
                    line=dict(color=colors[file_n]),
                    showlegend=showlegend,
                ),
                row=n_row,
                col=n_col
            )
            i_plot += 1

    fig = decorate_borders(fig, font_size, n_rows, n_columns, y_ranges, y_labels, show_flabels, time_unit)
    print(f"{file_name}...")
    return fig

# Function to find lines containing a specific word in a file
def find_word_in_file(file_path, word_to_find):
    """
    Searches for a specific word in a file and returns the line numbers where the word is found.

    Args:
        file_path (str): Path to the file.
        word_to_find (str): Word to search for in the file.

    Returns:
        line_numbers (list): List of line numbers where the word is found.
        tot_lines (int): Total number of lines in the file.
    """
    line_numbers = []
    with open(file_path, 'r') as file:
        tot_lines = 0
        for line_number, line in enumerate(file, start=1):
            tot_lines += line_number
            if word_to_find in line:
                line_numbers.append(line_number)
    if not line_numbers:
        line_numbers = [-3]
    return line_numbers, tot_lines

import re

def determine_ensemble(input_file):
    npt_pattern = r'fix\s+\S+\s+\S+\s+npt'
    nvt_pattern = r'fix\s+\S+\s+\S+\s+(nvt|temp/berendsen|temp/rescale|langevin)'
    nve_pattern = r'fix\s+\S+\s+\S+\s+nve'
    
    ensembles = []
    
    with open(input_file, 'r') as f:
        content = f.read()
        
        if re.search(npt_pattern, content, re.IGNORECASE):
            ensembles.append('NPT')
        if re.search(nvt_pattern, content, re.IGNORECASE):
            ensembles.append('NVT')
        if re.search(nve_pattern, content, re.IGNORECASE):
            ensembles.append('NVE')
    
    return ensembles

# Function to read and process the LAMMPS MD data file
def read_input_file(file_path, starting_word, ending_word):
    """
    Reads and processes the LAMMPS MD data file to prepare it for plotting.

    Args:
        file_path (str): Path to the data file.
        starting_word (str): The starting word to locate the data section.
        ending_word (str): The ending word to locate the data section.

    Returns:
        df (pd.DataFrame): Processed dataframe with the MD data.
        y_labels (list): List of y-axis labels for the data.
        y_ranges (list): List of maximum y-axis values for each label.
    """
    begin_line_numbers, tot_lines = find_word_in_file(file_path, starting_word)
    end_line_numbers, tot_lines = find_word_in_file(file_path, ending_word)

    if begin_line_numbers:
        if len(begin_line_numbers) == len(end_line_numbers) + 1:
            end_line_numbers.append(tot_lines - 2)

    with open(file_path, 'r') as file_data:
        data_ = file_data.readlines()

    df = pd.DataFrame()
    for i_l, f_l in zip(begin_line_numbers, end_line_numbers):
        data = [line.split() for line in data_[i_l:f_l - 1]]
        df_ = pd.DataFrame(data[1:-2], columns=data[0])
        df = pd.concat([df, df_.astype(float)])
    
    y_labels = data[0][1:len(data[0])-1]
    y_ranges = [max(np.array(df[key].astype(float))) for key in y_labels]

    ensembles = determine_ensemble(file_path)
    if 'NVT' in ensembles:
        y_labels = ['Temp', 'PotEng', 'KinEng', 'Press']
    if 'NPT' in ensembles:
        y_labels = ['Temp', 'PotEng', 'Press', 'Volume',]
    
    return df, y_labels, y_ranges

# Main function to generate plots from LAMMPS MD data files
def plot_lammps_md():
    """
    Main function to generate plots from LAMMPS MD data files.
    """
    filenames = []
    for i in range(1, 10):
        try:
            filenames.append(sys.argv[i])
        except IndexError:
            if i == 1:
                print("No log.lammps file is available HERE!!!")
                print("Usage: python plot_lammps_md.py file_name1 file_name2 file_name3 ...")
                exit()

    #filenames = ['340.log']#, '360.log', '380.log', '400.log']
    #filenames = ['log.lammps']
    fig = None
    y_ranges = []

    legend = 1 > len(filenames)

    for file_n, file_ in enumerate(filenames):
        starting_word = 'Per MPI rank memory'
        ending_word = 'Loop time of'

        df, y_labels, file_y_ranges = read_input_file(file_, starting_word, ending_word)

        if file_n == 0:
            y_ranges = file_y_ranges
        else:
            for i_key, key in enumerate(y_labels):
                if max(np.array(df[key].astype(float))) > y_ranges[i_key]:
                    y_ranges[i_key] = max(np.array(df[key].astype(float)))

        n_files = len(filenames)
        fig = plotly_plot(n_files, df, y_labels, fig, file_n, file_, y_ranges, legend)

    print(f"md_plots.png is writing...")
    fig.write_html('md_plots.html')
    fig.write_image('md_plots.png')
    print(f"md_plots.png is DONE")
    return fig.show()

