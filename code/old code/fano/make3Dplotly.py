import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('kalfasyan', 'jl8d8seyyl')

# Give the name of the file as fname
fname = '100'
# Here the fano factors, of each layer component
# from the exc/inh/ext tests, are read
with open("fanoExc"+fname+".txt","r") as f:
    exc = [i[0:-1] for i in f.readlines()]
with open("fanoInh"+fname+".txt","r") as f:
    inh = [i[0:-1] for i in f.readlines()]
with open("fanoExt"+fname+".txt","r") as f:
    ext = [i[0:-1] for i in f.readlines()]
with open("../ffnames.txt","r") as f:
    labels = [i[0:-1] for i in f.readlines()]

data = Data([
    Scatter3d(
        x= exc,
        y= inh,
        z= ext,
        mode='markers+text',
        name="FanoFactor",
        text= labels
    )
])
layout = Layout(
    showlegend=False,
    autosize=True,
    width=1368,
    height=781,
    xaxis=XAxis(
        title='exc',
        type='linear'
    ),
    yaxis=YAxis(
        title='inh',
        type='linear'
    ),
    scene=Scene(
        xaxis=XAxis(
            title='exc',
            type='linear'
        ),
        yaxis=YAxis(
            title='inh',
            type='linear'
        ),
        zaxis=ZAxis(
            title='ext',
            type='linear'
        ),
        cameraposition=[[0.27309410183979566, 0.6269146374994026, 0.6564905213984571, -0.3184616839484621], [0, 0, 0], 2.206892958728639]
    )
)
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)