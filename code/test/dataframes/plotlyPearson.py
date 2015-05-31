# Get this figure: fig = py.get_figure("https://plot.ly/~kalfasyan/177/")
# Get this figure's data: data = py.get_figure("https://plot.ly/~kalfasyan/177/").get_data()
# Add data to this figure: py.plot(Data([Scatter(x=[1, 2], y=[2, 3])]), filename ="Laminar Pearson Correlations", fileopt="extend"))
# Get y data of first trace: y1 = py.get_figure("https://plot.ly/~kalfasyan/177/").get_data()[0]["y"]

# Get figure documentation: https://plot.ly/python/get-requests/
# Add data documentation: https://plot.ly/python/file-options/

# You can reproduce this figure in Python with the following code!

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('username', 'api_key')
trace1 = Bar(
    x=['0_from_L23_to_L4', '0_from_L23_to_L5', '0_from_L4_to_L5', '1_from_L23_to_L4', '1_from_L23_to_L5', '1_from_L4_to_L5', '2_from_L23_to_L4', '2_from_L23_to_L5', '2_from_L4_to_L5', '3_from_L23_to_L4', '3_from_L23_to_L5', '3_from_L4_to_L5'],
    y=['0.7954282278', '0.4443265646', '0.3537094588', '0.7810189571', '0.410946672', '0.3660968837', '0.804674061', '0.4271859717', '0.3421852132', '0.7997426559', '0.4138625111', '0.3472419111'],
    name='0',
    error_y=ErrorY(
        color='rgb(0, 67, 88)',
        thickness=1,
        width=1
    ),
    error_x=ErrorX(
        copy_ystyle=True
    ),
    marker=Marker(
        color='rgb(4, 158, 215)',
        line=Line(
            color='white',
            width=0
        )
    ),
    opacity=1,
    visible=True
)
trace2 = Bar(
    x=['0_from_L23_to_L4', '0_from_L23_to_L5', '0_from_L4_to_L5', '1_from_L23_to_L4', '1_from_L23_to_L5', '1_from_L4_to_L5', '2_from_L23_to_L4', '2_from_L23_to_L5', '2_from_L4_to_L5', '3_from_L23_to_L4', '3_from_L23_to_L5', '3_from_L4_to_L5'],
    y=['0.7615026711', '0.434912242', '0.3148133618', '0.8121751694', '0.4431134256', '0.3452789586', '0.788944957', '0.450232405', '0.391214824', '0.8000257096', '0.4120200357', '0.3760079038'],
    name='1',
    error_y=ErrorY(
        color='rgb(31, 138, 112)',
        thickness=1,
        width=1
    ),
    error_x=ErrorX(
        copy_ystyle=True
    ),
    marker=Marker(
        color='rgb(114, 206, 243)',
        line=Line(
            color='white',
            width=0
        )
    ),
    opacity=1
)
trace3 = Bar(
    x=['0_from_L23_to_L4', '0_from_L23_to_L5', '0_from_L4_to_L5', '1_from_L23_to_L4', '1_from_L23_to_L5', '1_from_L4_to_L5', '2_from_L23_to_L4', '2_from_L23_to_L5', '2_from_L4_to_L5', '3_from_L23_to_L4', '3_from_L23_to_L5', '3_from_L4_to_L5'],
    y=['0.7869129906', '0.4040461398', '0.34178587', '0.7760122613', '0.4276066811', '0.3730659918', '0.7917867211', '0.395684563', '0.3331718857', '0.7724692628', '0.421830575', '0.3420531436'],
    name='2',
    error_y=ErrorY(
        color='rgb(190, 219, 57)',
        thickness=1,
        width=1
    ),
    error_x=ErrorX(
        copy_ystyle=True
    ),
    marker=Marker(
        color='rgb(2, 74, 97)',
        line=Line(
            color='white',
            width=0
        )
    ),
    opacity=1
)
trace4 = Bar(
    x=['from_to'],
    y=['correlation'],
    name='',
    error_y=ErrorY(
        color='rgb(255, 225, 26)',
        thickness=1,
        width=1
    ),
    error_x=ErrorX(
        copy_ystyle=True
    ),
    marker=Marker(
        color='rgb(171, 171, 173)',
        line=Line(
            color='white',
            width=0
        )
    ),
    opacity=1,
    visible='legendonly'
)
data = Data([trace1, trace2, trace3, trace4])
layout = Layout(
    title='Laminar Pearson Correlations',
    titlefont=Font(
        family='"Open sans", verdana, arial, sans-serif',
        size=17,
        color='#444'
    ),
    font=Font(
        family='"Droid Sans", sans-serif',
        size=12,
        color='#444'
    ),
    showlegend=True,
    autosize=True,
    width=869,
    height=476,
    xaxis=XAxis(
        title='minicolumn_Layer1_with_Layer2',
        titlefont=Font(
            family='"Verdana", monospace',
            size=12,
            color='black'
        ),
        range=[-0.5, 11.5],
        type='category',
        rangemode='normal',
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        nticks=0,
        ticks='',
        showticklabels=True,
        ticklen=6,
        tickcolor='rgba(0, 0, 0, 0)',
        tickangle='auto',
        tickfont=Font(
            family='"Verdana", monospace',
            size=10,
            color='black'
        ),
        exponentformat='B',
        showexponent='all',
        mirror=False,
        gridcolor='white',
        gridwidth=1,
        zerolinewidth=1,
        linecolor='rgba(152, 0, 0, 0.5)',
        linewidth=1.5
    ),
    yaxis=YAxis(
        title='Pearson corr. coefficient',
        titlefont=Font(
            family='"Verdana", monospace',
            size=12,
            color='black'
        ),
        range=[0, 0.8549212309473684],
        type='linear',
        rangemode='normal',
        autorange=True,
        showgrid=True,
        zeroline=False,
        showline=False,
        nticks=0,
        ticks='',
        showticklabels=True,
        ticklen=6,
        tickcolor='rgba(0, 0, 0, 0)',
        tickangle='auto',
        tickfont=Font(
            family='"Verdana", monospace',
            size=10,
            color='black'
        ),
        exponentformat='B',
        showexponent='all',
        mirror=False,
        gridcolor='white',
        gridwidth=1,
        zerolinecolor='#444',
        zerolinewidth=1,
        linecolor='rgba(152, 0, 0, 0.5)',
        linewidth=1.5
    ),
    legend=Legend(
        x=1.02,
        y=1,
        traceorder='normal',
        font=Font(
            family='"Open sans", verdana, arial, sans-serif',
            size=12,
            color='#444'
        ),
        bgcolor='#fff',
        bordercolor='#444',
        borderwidth=0,
        xref='paper',
        yref='paper'
    ),
    paper_bgcolor='rgb(213, 226, 233)',
    plot_bgcolor='rgb(213, 226, 233)',
    hovermode='x',
    dragmode='zoom',
    separators='.,',
    barmode='group',
    bargap=0.2,
    bargroupgap=0,
    hidesources=False
)
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)