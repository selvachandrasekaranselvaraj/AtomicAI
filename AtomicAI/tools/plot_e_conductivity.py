import numpy as np
import plotly.graph_objects as go

#line_intervals = 704 # 504, or 2004
#ReCONDUCTIVITY_AVERFORM
file_name_ac = 'ReCONDUCTIVITYFORM'
with open(file_name_ac, 'r') as file:
    lines = file.readlines()

# Find the line index where 'deltae' no longer appears consecutively
i = 3  # Starting line index (adjust if needed)
while 'deltae' not in lines[i]:
    i += 1

line_intervals = i - 2  # Calculate interval length based on header lines
print(f"Detected line intervals: {line_intervals}")


deltae, omegas, conducs = [], [], []
for i in range(2, len(lines), line_intervals):  # 504, 5004
    deltae.append(float(lines[i][:-1].split()[2]))
    #print(lines[i])

print(deltae)

for i, delta in enumerate(deltae):
    omega, conduc = [], []
    for j in range(5, line_intervals):  # 504, 2004
        k = i*line_intervals+j  # 504, 2004
        omega.append(float(lines[k][:-1].split()[0]))
        conduc.append(float(lines[k][:-1].split()[1]))
    #omegas.append(omega)
    conducs.append([omega, conduc])
        #print(lines[i*504+j])


deltae_ = [0.74, 0.78, 0.82]
del_, omegas_, conducs_ = [], [], []

for d, c in zip(deltae, conducs):
    if d in deltae_:
        del_.append(d)
        omegas_.append(c[0])
        conducs_.append(c[1])

del_ = [0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001] #[0.22, 0.24, 0.26]
print(del_, len(omegas_), len(conducs_[0]))

#print(deltae, conduc)

# Create the step plot for Na-P
fig = go.Figure()

font_size = 24
conductivities = np.array([conducs_[i] for i in range(0, len(conducs_))]) 
labels = [del_[i] for i in range(0, len(conducs_))]
colors = ['red', 'green', 'blue']
for i, conductivity in enumerate(conductivities):
    y_tick_window = max(conductivity)/5
    fig.add_trace(go.Scatter(
        x=omegas_[i], 
        y=conductivity,
        mode='lines',
        #line_shape='hv',
        #text=df_plot['fu'],
        #textposition='top right',
        #textfont=dict(
        #    color=colors[i],  # Change text color here
        #    size=font_size  # Adjust font size as needed
        #),
        name=labels[i],
        #line=dict(color=colors[i]),
    ))

# Define axis details
x_axis_details = dict(
    gridcolor='lightgray',  # Set the gridline color
    griddash='dash',
    showline=True,  # Show the border line
    linecolor='black',  # Set the border line color
    linewidth=1,
    mirror=True,
    titlefont=dict(size=font_size+2, color="black"),
    ticks="inside",
    tickwidth=1,
    ticklen=10,
    #minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True, dtick=250), 
    #minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True), 
)

y_axis_details = dict(
    gridcolor='lightgray',  # Set the gridline color
    griddash='dash',
    showline=True,  # Show the border line
    linecolor='black',  # Set the border line color
    linewidth=1,
    titlefont=dict(size=font_size+2, color="black"),
    mirror=True,
    ticks="inside",
    tickwidth=1,
    ticklen=10,
    #dtick=y_tick_window,
    #minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True, dtick=y_tick_window*0.5), 
    #minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True), 
)

# Update layout
fig.update_layout(
    xaxis=x_axis_details,
    yaxis=y_axis_details,
    showlegend=True,
    legend=dict(
        x=0.7,
        y=1.0,
        traceorder='normal',
        font=dict(size=font_size),
        bgcolor='rgba(0,0,0,0)',
        itemsizing = 'constant'
    ),
    font=dict(size=font_size, color="black"),
    height=450,
    width=650,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
    paper_bgcolor='rgba(0,0,0,0)'  # Set paper background to transparent
)

#fig.update_xaxes(title_text='ω (eV)', range=[0.0, 0.5], title_standoff=30) # range=[0.0, 0.4],
#fig.update_yaxes(title_text='σ (S/cm)', range=[0.0, 1e-3], title_standoff=40) #range=[0.0, 1.5]

# Save the plot to an image file
#fig.write_image("omega_sigma1.png")

fig.update_xaxes(title_text='ω (eV)', range=[0.0, 6.5], title_standoff=30) # range=[0.0, 0.4],
fig.update_yaxes(title_text='σ (S/cm)', title_standoff=40) #range=[0.0, 1.5]

# Save the plot to an image file
fig.write_image("omega_sigma2.png")

