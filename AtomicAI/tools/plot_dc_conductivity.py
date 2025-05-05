import numpy as np
import plotly.graph_objects as go

file_name = './ReCONDUCTIVITY_DCFORM'
with open(file_name, 'r') as file:
    lines = file.readlines()
    
deltae, conduc = [], []
for i in range(2, len(lines), 7):
    deltae.append(float(lines[i][:-1].split(' ')[2]))

for i in range(5, len(lines), 7):
    conduc.append(float(lines[i][:-1].split()[1]))  
    #print(lines[i])

#print(deltae, conduc)
bst = [deltae, np.array(conduc)* 1e-2]
# Create the step plot for Na-P
fig = go.Figure()

font_size = 24
conductivities = [bst]
labels = ["Ti10"] #['Bi<sub>1.67</sub>Sb<sub>0.33</sub>Te<sub>3</sub>']
colors = ['red', 'green', 'blue']
for i, conductivity in enumerate(conductivities):
    y_tick_window = int(max(conductivity[1])/5)
    fig.add_trace(go.Scatter(
        x=conductivity[0], 
        y=conductivity[1],
        mode='lines',
        #line_shape='hv',
        #text=df_plot['fu'],
        #textposition='top right',
        textfont=dict(
            color=colors[i],  # Change text color here
            size=font_size  # Adjust font size as needed
        ),
        name=labels[i],
        line=dict(color=colors[i]),
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
    minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True, dtick=250), 
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
    dtick=y_tick_window,
    minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True, dtick=int(y_tick_window*0.5)), 
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

fig.update_xaxes(title_text='∆E (eV)', range=[0.0, 0.9], title_standoff=30)
fig.update_yaxes(title_text='σ (S/cm)', title_standoff=40) #range=[0.0, 1.5]


# Save the plot to an image file
fig.write_image("deltae_sigma.png")
fig.show()
