import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np

from dash.dependencies import Input, Output
from plotly import graph_objs as go
from plotly.graph_objs import *
from datetime import datetime as dt

import json
import urllib.request
import time
import os, sys
import shutil
import socket


app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# Plotly mapbox public token
mapbox_access_token = "pk.eyJ1IjoibW9ycGhldXMyNyIsImEiOiJja2Zrd3J0dWMwb2pxMnlwY3g0Zmtza3ZuIn0.obFXuRnZeFgcHdzxq-Co4g"


# Initialize data frame
df = pd.read_csv( "new_data_sno1_predict.csv",dtype=object)

df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M")
df.index = df["time"]
df.drop("time", 1, inplace=True)

totalList = []
for month in df.groupby(df.index.month):
    dailyList = []
    for day in month[1].groupby(month[1].index.day):
        dailyList.append(day[1].bemp)
    totalList.append(dailyList)
totalList = np.array(totalList)

ndf = df.bemp.astype(float).astype(int)

def getStationInformation():
    ubike_url = "https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json"
    #urllib.request.urlretrieve(ubike_url, "YouBikeTP.json")
    json_file = open("YouBikeTP.json")
    ubike_data = json.load(json_file)
    text_list = []
    json_info_list = []
    i = 0
    for key,value in ubike_data['retVal'].items():
        i = i + 1
        sno = value['sno']
        sna = value['sna']
        tot = value['tot']
        sbi = value['sbi']
        sarea = value['sarea']
        mday = value['mday']
        lat = value['lat']
        lng = value['lng']
        ar = value['ar']
        sareaen = value['sareaen']
        snaen = value['snaen']
        aren = value['aren']
        bemp = value['bemp']
        act = value['act']
        str_station = '[{sno}] {sna}<br> bike:{sbi}<br> empty:{bemp}<br> total:{tot}'.format(sno=sno,sna=sna,sbi=sbi,bemp=bemp,tot=tot)
        text_list = text_list + [str_station]
        value['idx'] = i
        json_info_list = json_info_list + [value]

    return json_info_list, text_list

station_info_context, txt_list_tmp = getStationInformation()


graph_div = html.Div(
    className = "graph-div",
    children=[
        dcc.Graph(
            id='map-graph',
        ),
        dcc.Graph(
            id='line-chart',
        ),
    ]
)

def value_block(title, value, blk_id, wper):
    block_obj = html.Div(
        className = 'div-value-block',
        style={'float':'left', 'width': wper},
        children = [
            html.P(title, style={'text-align':'left','font-size':'15px'}),
            html.P(value, style={'text-align':'center','font-size':'30px'}, id=blk_id)
        ]
    )
    return block_obj


user_control_div = html.Div (
        className = "div-user-control",
        children=[
            html.H5("Please enter the informations below"),
            dcc.Dropdown(
                id="location-dropdown",
                options=[ {"value":data['idx'] , "label": "[{sno}] {name}".format(sno=data['sno'],name=data['sna'])} for data in station_info_context],
                placeholder="Select a location",
            ),

            dcc.DatePickerSingle(
                id="date-picker",
                min_date_allowed=dt(2018, 2, 8),
                max_date_allowed=dt(2018, 12, 30),
                initial_visible_month=dt(2018, 2, 8),
                date=dt(2018, 2, 8).date(),
                display_format="MMM D, YYYY",
            ),
            dcc.Dropdown(
                id="bar-selector",
                options=[
                    {
                        "label": str(n) + ":00",
                        "value": str(n),
                    } for n in range(24)
                ],
                multi=False,
                placeholder="Select certain hours",
            )
        ],
)

search_result = html.Div(
    className = 'search-result-div',
    children=[
        value_block('True Empty','100', 'truth_empty_value',"40%"),
        value_block('True Bike','100','truth_bike_value',"40%"),
        value_block('Predict Empty','100', 'pred_empty_value',"40%"),
        value_block('Predict Bike','100','pred_bike_value',"40%")
    ]
)

current_status_div = html.Div(
    className = 'current-station-div',
    children = [
        html.P("No.",id='station-no'),
        html.H3("Station Name",id='station-name'),
        html.P("Station Address",id='station-addr'),
        html.P("Status",id='station-status'),
        html.Div(
            children = [
                value_block("Bike",'--',"station-bike-value",'80px'),
                value_block("Empty",'--',"station-empty-value",'80px') ,
                value_block("Total",'--',"station-total-value","80px")
            ]
        )]
)

info_up_div = html.Div(
    className = "info-up-div",
    children=[
        user_control_div,
        current_status_div,

    ]
)

info_down_div = html.Div(
    className = "info-down-div",
    children=[
        search_result
    ]
)

info_div = html.Div(
    className = "info-div",
    children=[
        info_up_div,
        info_down_div
    ]
)

body_div = html.Div(
    className = "body-div",
    children=[
        info_div,
        graph_div
    ]
)

header_div = html.Div(
    className = "header-div",
    children=[
        html.H1(children='Youbike Preidction')
    ]
)

app.layout = html.Div(
    className = "main-div",
    children=[
        header_div,
        body_div,
])



# Get the Coordinates of the chosen months, dates and times
def getLatLonColor(selectedData, datepiked):
    listCoords = ndf[datepiked]

    # No times selected, output all times for chosen month and date
    if selectedData is None or len(selectedData) is 0:
        return listCoords.sum()

    selectnum = 0
    for time in selectedData:
        selectnum = selectnum + listCoords[listCoords.index.hour == time]

    return selectnum


title = 'Main Source for News'
labels = ['Television', 'Newspaper', 'Internet', 'Radio']
colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']

mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]
x_data = np.vstack((np.arange(2001, 2014),)*4)
y_data = np.array([
    [74, 82, 80, 74, 73, 72, 74, 70, 70, 66, 66, 69],
    [45, 42, 50, 46, 36, 36, 34, 35, 32, 31, 31, 28],
    [13, 14, 20, 24, 20, 24, 24, 40, 35, 41, 43, 50],
    [18, 21, 18, 21, 16, 14, 13, 18, 17, 16, 19, 23],
])


# Update Map Graph based on date-picker, selected data on histogram and location dropdown
@app.callback(
    Output("station-empty-value", "children"),
    Output("station-bike-value", "children"),
    Output("station-total-value", "children"),
    Output("station-no", "children"),
    Output("station-name", "children"),
    Output("station-addr", "children"),
    Output("station-status", "children"),
    Output("map-graph", "figure"),
    Output("line-chart", "figure"),
    [
        Input("date-picker", "date"),
        Input("bar-selector", "value"),
        Input("location-dropdown", "value"),
    ],
)
def update_graph(datePicked, selectedData, selectedLocation):

    zoom = 12.0
    latInitial = 25.0408578889
    lonInitial = 121.567904444
    bearing = 0

    total_num = '--'
    bike_num = '--'
    empty_num = '--'
    station_no = "No. "
    station_name = "Station Name"
    station_addr = "Address"
    station_status = "Status: Unknown"

    # read from where?
    json_info,txt_list = getStationInformation()
    station_data ={}


    if selectedLocation:
        zoom = 15.0
        station_data = json_info[selectedLocation-1]
        latInitial = float(station_data["lat"])
        lonInitial = float(station_data["lng"])
        total_num = station_data['tot']
        bike_num = station_data['sbi']
        empty_num = station_data['bemp']
        station_no = station_no + station_data['sno']
        station_name = station_data['sna']
        station_addr =  station_data['ar']
        if(int(station_data['act']) == 1):
            station_status = "Status: Active"
        else:
            station_status = "Status: Inactive"


    date_picked = dt.strptime(datePicked, "%Y-%m-%d")
    monthPicked = date_picked.month
    dayPicked = date_picked.day

    listCoords = getLatLonColor(selectedData, datePicked)
    print("selectedData:",selectedData)
    print("datePicked:",datePicked)

    fig = go.Figure(
        data=[
            # Plot of important locations on the map
            Scattermapbox(
                lat=[json_info[i]["lat"] for i in range(len(json_info))],
                lon=[json_info[i]["lng"] for i in range(len(json_info))],
                mode="markers",
                hoverinfo="text",
                text= txt_list,
                marker=dict(size=15, symbol='bicycle'),
            ),
        ],
        layout=Layout(
            autosize=True,
            margin=go.layout.Margin(l=0, r=0, t=0, b=0),
            showlegend=False,
            height=400,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
                style='mapbox://styles/mapbox/streets-v11',
                bearing=bearing,
                zoom=zoom,
            ),
            updatemenus=[
                dict(
                    buttons=(
                    [
                        dict(
                            args=[
                                {
                                    "mapbox.zoom": 12,
                                    "mapbox.center.lon": "121.567904444",
                                    "mapbox.center.lat": "25.0408578889",
                                    "mapbox.bearing": 0,
                                    "mapbox.style": "dark",
                                }
                            ],
                        label="Reset Zoom",
                        method="relayout",
                    )
                    ]
                    ),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#323130",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )

    fig2 = go.Figure()

    for i in range(0, 4):
        fig2.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))

        # endpoints
        fig2.add_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[y_data[i][0], y_data[i][-1]],
            mode='markers',
            marker=dict(color=colors[i], size=mode_size[i])
        ))
    fig2.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=10,
            r=10,
            t=10,
        ),
        showlegend=False,
        plot_bgcolor='white',
        height=400,
    )

    return empty_num, bike_num, total_num,station_no,station_name,station_addr,station_status, fig,fig2


if __name__ == '__main__':
    app.run_server(debug=True)
