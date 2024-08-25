import plotly.express as px
def plot_line(vts, dimensions, r2, element):
    fig = px.line(x=dimensions, y=r2, markers=True, text= ["{:.1E}".format(vt_) for vt_ in vts])
    fig.update_layout(
                      width = 1000,
                      height = 400,

                      xaxis_title='VT',
                      yaxis_title=r'$R^2$',
    )
    fig.write_html(f"{element}_VT_optimization.html")
