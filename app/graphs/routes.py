# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import math
from . import blueprint
# Flask modules
import plotly.subplots as sp
from flask import render_template, request, url_for, redirect, send_from_directory, jsonify, make_response
from flask_table import Table, Col, LinkCol
from functools import partial
import json
import time
from scipy.spatial.distance import pdist
pw_jaccard_func = partial(pdist, metric='jaccard')
import scipy.cluster.hierarchy as sch
import random
from wordcloud import WordCloud, STOPWORDS
from pytimeparse.timeparse import timeparse
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
import plotly.express as px
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import gensim
from plotly.subplots import make_subplots
import os
from os import path
from time import sleep
from matplotlib.figure import Figure
from dash_holoniq_wordcloud import DashWordcloud
from dash import Dash, dcc, html
from bs4 import BeautifulSoup
import plotly.graph_objs as go
import csv
from collections import defaultdict
import string
stop_words = set(stopwords.words("english"))
from collections import OrderedDict
from plotly import exceptions, optional_imports
from plotly.graph_objs import graph_objs
from distinctipy import distinctipy
np = optional_imports.get_module("numpy")
scp = optional_imports.get_module("scipy")
sch = optional_imports.get_module("scipy.cluster.hierarchy")
scs = optional_imports.get_module("scipy.spatial")
import numpy as np
np.random.seed(1)
import plotly
import pandas as pd
import pickle
import seaborn as sns 
import os
import sys
import pymongo
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from collections import namedtuple
from json import JSONEncoder
from bigtree import Node, print_tree, dataframe_to_tree, tree_to_dataframe
from datetime import datetime
from datetime import timezone
import datetime
import networkx as nx
from scipy.spatial.distance import jaccard, squareform
from bertopic import BERTopic
from plotly.graph_objs import *
global threshold
global glo_dataframes
from eventregistry import *
from chart_studio import plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from dotenv import load_dotenv
load_dotenv()

threshold = 0.2
dendro_clusters = 0
color_ran = []
selected_event = ""
selected_barrier = ""
dbfileName = 'data/ForPropagationNetworkNew2.csv'
total_clusters = 0
clustered_dataframes = pd.DataFrame()
client = pymongo.MongoClient(os.getenv("DB_URL"))
db = client.test



colors = ["blue",  "black",  "brown", "gray", "green", "orange", "purple", "red", "white", "yellow", "bisque",
                "burlywood", "chartreuse", "chocolate", "coral", "cornsilk", "crimson", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", 
                "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
                "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
                "dimgray", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro",
                "ghostwhite", "gold", "goldenrod",  "greenyellow", "honeydew", "hotpink", "indianred",
                "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", 
                "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink", "lightsalmon", 
                "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue", "lightyellow", "lime",
                "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
                "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue",
                "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", 
                "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff",
                "peru", "pink", "plum", "powderblue",  "rosybrown", "royalblue", "rebeccapurple", "saddlebrown",
                "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray",
                "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "violet"] 


KEY = "EVENTR_KEY"
er = EventRegistry(apiKey=KEY)


@blueprint.route('/users.html')
@blueprint.route('/<path>')
def index(path):
    content = None
    try:
        return render_template('layouts/default.html',
                               content=render_template('pages/' + path))

    except:
        return render_template('layouts/auth-default.html',
                               content=render_template('pages/404.html'))

# Return sitemap
@blueprint.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, '../static'), 'sitemap.xml')

@blueprint.route('/quantitative')
def quantitative():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/index.html'))


@blueprint.route('/traffic')
def traffic():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/Traffic.html'))


@blueprint.route('/cascade')
def cascade():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/Cascade.html'))


@blueprint.route('/followers')
def followers():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/Followers.html'))


@blueprint.route('/forensic')
def forensic():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/Forensic.html'))


@blueprint.route('/')
def users():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/users.html'))

@blueprint.route('/posts')
def posts():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/posts.html'))

@blueprint.route('/surveys')
def surveys():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/surveys.html'))

@blueprint.route('/ranker')
def ranker():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/ranker.html'))

@blueprint.route('/bots')
def bots():
    #print("ad dashboard")
    return render_template('layouts/default.html', content=render_template('pages/bots.html'))



########################### FIRST PAGE ###########################################
@blueprint.route('/usercomparison', methods=['GET', 'POST'])
def usercomparison():
    usersData = pd.DataFrame(list(db.users.find()))
    dfg = usersData.groupby('isAdmin').count().reset_index()
    dfg = dfg.rename(columns={"isAdmin": "User is Admin?", "username": "Total Users"})
    fig = px.bar(dfg,x='User is Admin?',y='Total Users',title='<b>Admin users vs. Other users</b>')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@blueprint.route('/postsovertime', methods=['GET', 'POST'])
def postsovertime():
    postsData = pd.DataFrame(list(db.posts.find()))
    postsData['createdAt'] = postsData['createdAt'].astype('datetime64[ns]')
    postsData['MD'] = postsData['createdAt'].dt.strftime('%Y-%m-%d')
    
    dfg = postsData.groupby('MD').count().reset_index()
    dfg = dfg.rename(columns={"MD": "Time", "desc": "Total Posts"})

    #dfg = dfg.rename(columns={"isAdmin": "User is Admin?", "username": "Total Users"})
    fig = px.bar(dfg, x='Time',y='Total Posts',title='<b>Volume of posts (daily) over time</b>')
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/userssovertime', methods=['GET', 'POST'])
def userssovertime():
    postsData = pd.DataFrame(list(db.users.find()))
    postsData['createdAt'] = postsData['createdAt'].astype('datetime64[ns]')
    postsData['MD'] = postsData['createdAt'].dt.strftime('%Y-%m-%d')
    
    dfg = postsData.groupby('MD').count().reset_index()
    dfg = dfg.rename(columns={"MD": "Time", "desc": "Total Users"})

    #dfg = dfg.rename(columns={"isAdmin": "User is Admin?", "username": "Total Users"})
    fig = px.bar(dfg, x='Time',y='Total Users',title='<b>Volume of users (daily) over time</b>')
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/postsbyUsers', methods=['GET', 'POST'])
def postsbyUsers():
    postsData = pd.DataFrame(list(db.posts.find()))
    postsData = postsData.rename(columns={"_id": "postId"}) 
    postsData = postsData.dropna(subset=['postId'], axis=0)
    postsData = postsData.dropna(subset=['userId'], axis=0)
    postsData = postsData.dropna(subset=['desc'], axis=0)
    
    users = pd.DataFrame(list(db.users.find()))
    users = users.rename(columns={"_id": "userId"}) 
    users = users.dropna(subset=['username'], axis=0)
    users = users.dropna(subset=['userId'], axis=0)
    
    users = users.dropna(subset=['userId'], axis=0)
    postsData['userId'] = postsData['userId'].astype(str)
    users['userId'] = users['userId'].astype(str)
    
    users = users[["userId", "username"]].reset_index(drop=True)
    postsData = postsData[["postId", "userId","desc"]].reset_index(drop=True)
    
    df_merged = pd.merge(users, postsData, on ='userId', how ="left")
    dfg = df_merged.groupby(['username', "postId"]).count().reset_index()
    dfg = dfg.rename(columns={"username": "Username", "desc":"Total Posts"})
    
    fig = px.bar(dfg, x="Username",y="Total Posts", title='<b>Total posts</b>')
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/viewershipsOfPosts', methods=['GET', 'POST'])
def viewershipsOfPosts():
    
    readposts = pd.DataFrame(list(db.readposts.find()))
    readposts = readposts.dropna(subset=['userId'], axis=0)
    readposts = readposts[["postId", "userId"]].reset_index(drop=True)
    
    users = pd.DataFrame(list(db.users.find()))
    users = users.rename(columns={"_id": "userId"}) 
    users = users.dropna(subset=['username'], axis=0)
    users = users.dropna(subset=['userId'], axis=0)
    
    users = users.dropna(subset=['userId'], axis=0)
    readposts['userId'] = readposts['userId'].astype(str)
    users['userId'] = users['userId'].astype(str)
    
    users = users[["userId", "username", "email"]].reset_index(drop=True)
    readposts = readposts[["postId", "userId"]].reset_index(drop=True)
    
    df_merged = pd.merge(users,readposts , on ='userId', how ="left")
    totalViewership = df_merged.groupby(['username', "userId", "postId"]).count().reset_index()
    totalViewership = totalViewership.rename(columns={"username":"Username", "email":"Total posts read"})
    
    fig = px.bar(totalViewership, x="Username",y="Total posts read", title='<b>Total posts read by each user</b>')
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@blueprint.route('/userActivityHome', methods=['GET', 'POST'])
def userActivityHome():
    activities = pd.DataFrame(list(db.timemes.find()))
    activities = activities.dropna(how='any',axis=0)
    activities = activities[activities["page"] == "Home"]  
    
    users = pd.DataFrame(list(db.users.find()))
    users = users.rename(columns={"_id": "userId"}) 
    users = users.dropna(subset=['username'], axis=0) 
    
    df_merged = activities.merge(users, on='userId', how = 'left').reset_index()
    df_merged = df_merged[["seconds", "userId","username", "createdAt_x", "page"]] 
    df_merged['seconds'] = df_merged['seconds'].astype(float).astype(int)
    
    #df_merged['datetime'] = pd.to_datetime(df_merged['seconds'], unit='s')
    #df_merged['datetime_string'] = df_merged['datetime'].dt.strftime('%S')
    
    df_merged['createdAt_x'] = df_merged['createdAt_x'].astype('datetime64[ns]')
    df_merged = df_merged.rename(columns={"createdAt_x": "Time", "seconds": "Time Spent (seconds)"})
    
    dfg = df_merged.groupby(['Time', 'Time Spent (seconds)', 'username']).count().reset_index()
    fig = px.line(dfg, x="Time", y="Time Spent (seconds)", color='username',title='<b>Time spent by each user on home screen</b>', line_shape='linear')
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/postDetailActivity', methods=['GET', 'POST'])
def postDetailActivity():
    activities = pd.DataFrame(list(db.timemes.find()))
    activities = activities.dropna(how='any',axis=0)
    activities = activities[activities["page"] == "DetailPage"]  
    
    users = pd.DataFrame(list(db.users.find()))
    users = users.rename(columns={"_id": "userId"}) 
    users = users.dropna(subset=['username'], axis=0) 
    
    df_merged = activities.merge(users, on='userId', how = 'left').reset_index()
    df_merged = df_merged[["seconds", "userId","username", "createdAt_x", "page"]] 
    df_merged['seconds'] = df_merged['seconds'].astype(float).astype(int)
    
    #df_merged['datetime'] = pd.to_datetime(df_merged['seconds'], unit='s')
    #df_merged['datetime_string'] = df_merged['datetime'].dt.strftime('%S')
    
    df_merged['createdAt_x'] = df_merged['createdAt_x'].astype('datetime64[ns]')
    df_merged = df_merged.rename(columns={"createdAt_x": "Time", "seconds": "Time Spent (seconds)"})
    
    dfg = df_merged.groupby(['Time', 'Time Spent (seconds)', 'username']).count().reset_index()
    fig = px.line(dfg, x="Time", y="Time Spent (seconds)", color='username',title='<b>Time spent by each user on Detail screen</b>', line_shape='linear')
    fig.update_layout(font=dict(family="Helvetica", size=18, color="black"))
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



@blueprint.route('/postviewovertime', methods=['GET', 'POST'])
def postviewovertime():
    postsData = pd.DataFrame(list(db.posts.find()))
    postsData['createdAt'] = postsData['createdAt'].astype('datetime64[ns]')
    postsData['MD'] = postsData['createdAt'].dt.strftime('%Y-%m-%d')
    
    dfg = postsData.groupby('MD').count().reset_index()
    dfg = dfg.rename(columns={"MD": "Time", "desc": "Total Posts"})

    #dfg = dfg.rename(columns={"isAdmin": "User is Admin?", "username": "Total Users"})
    fig = px.bar(dfg, x='Time',y='Total Posts',title='Volume of posts over time')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/postreadovertime', methods=['GET', 'POST'])
def postreadovertime():
    postsData = pd.DataFrame(list(db.posts.find()))
    postsData['createdAt'] = postsData['createdAt'].astype('datetime64[ns]')
    postsData['MD'] = postsData['createdAt'].dt.strftime('%Y-%m-%d')
    
    dfg = postsData.groupby('MD').count().reset_index()
    dfg = dfg.rename(columns={"MD": "Time", "desc": "Total Posts"})

    #dfg = dfg.rename(columns={"isAdmin": "User is Admin?", "username": "Total Users"})
    fig = px.bar(dfg, x='Time',y='Total Posts',title='Volume of posts over time')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/presurveysubmissiontime', methods=['GET', 'POST'])
def presurveysubmissiontime():
    usersData = pd.DataFrame(list(db.users.find()))
    presurData = pd.DataFrame(list(db.presurveys.find()))
    usersData['uniqueId'] = usersData['uniqueId'].astype(str)
    presurData['uniqueId'] = presurData['uniqueId'].astype(str)
    usersData = usersData[["_id", "username", "uniqueId"]].reset_index(drop=True)
    presurData = presurData[["uniqueId", "createdAt"]].reset_index(drop=True)
    df_merged = pd.merge(presurData, usersData, on ='uniqueId', how ="left")
    df_merged = df_merged.rename(columns={"createdAt":"time"})
    fig = px.bar(df_merged, x="username",y="time", title='Users who submitted the pre-survey?')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/postsurveysubmissiontime', methods=['GET', 'POST'])
def postsurveysubmissiontime():
    usersData = pd.DataFrame(list(db.users.find()))
    presurData = pd.DataFrame(list(db.postsurveys.find()))
    usersData['_id'] = usersData['_id'].astype(str)
    presurData['userId'] = presurData['userId'].astype(str)
    usersData = usersData[["_id", "username", "uniqueId"]].reset_index(drop=True)
    usersData = usersData.rename(columns={"_id":"userId"})
    presurData = presurData[["userId", "createdAt"]].reset_index(drop=True)
    df_merged = pd.merge(presurData, usersData, on ='userId', how ="left")
    
    df_merged = df_merged.rename(columns={"createdAt":"time"})
    fig = px.bar(df_merged, x="username",y="time", title='Users who submitted the post-survey?')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@blueprint.route('/totalRankedPosts', methods=['GET', 'POST'])
def totalRankedPosts():
    ppl_flower = pd.DataFrame(list(db.posts.find()))
    ppl_flower['userId'] = ppl_flower['userId'].astype(str)
    usersData = pd.DataFrame(list(db.users.find()))
    usersData['_id'] = usersData['_id'].astype(str)
    usersData = usersData.rename(columns={"_id":"userId"})

    df_merged = pd.merge(ppl_flower, usersData, on ='userId', how ="left")
    df_sorted = df_merged.sort_values(by='updatedAt_x', ascending=False)
    df_sorted['MD'] = df_sorted['updatedAt_x'].dt.strftime('%Y-%m-%dT%H:%M')
    df_sorted = df_sorted.rename(columns={"MD":"Time"})

    df_sorted = df_sorted.head(10)
    fig = px.bar(df_sorted, x='Time', y='rank', color='username', title='Ten latest ranked posts')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
    

@blueprint.route('/BotsTotalRepliesRoute', methods=['GET', 'POST'])
def BotsTotalRepliesRoute():
    one = db.users.find({'username': 'Weißes Walross'})
    two = db.users.find({'username': 'Weißer Hase'})
    three = db.users.find({'username': 'Schwarzer Ninja'})
    four = db.users.find({'username': 'Oranger Ninja'})
    five = db.users.find({'username': 'Lila Walross'})
    six = db.users.find({'username': 'Lila Krähe'})
    seven = db.users.find({'username': 'Lila Blume'})
    eight = db.users.find({'username': 'Grünes Kaninchen'})
    nine = db.users.find({'username': 'Grauer Otter'})
    ten = db.users.find({'username': 'Graue Krähe'})
    ele = db.users.find({'username': 'Gelber Roboter'})
    twe = db.users.find({'username': 'Blaues Siegel'})
    thir = db.users.find({'username': 'Blaues Huhn'})
    fou = db.users.find({'username': 'Blauer Ninja'})
    fif = db.users.find({'username': 'Blauer Biber'})
    
    one_latest = pd.DataFrame()
    two_latest = pd.DataFrame()
    three_latest = pd.DataFrame()
    four_latest = pd.DataFrame()
    five_latest = pd.DataFrame()
    six_latest = pd.DataFrame()
    seven_latest = pd.DataFrame()
    eight_latest = pd.DataFrame()
    nine_latest = pd.DataFrame()
    ten_latest = pd.DataFrame()
    ele_latest = pd.DataFrame()
    twe_latest = pd.DataFrame()
    thir_latest = pd.DataFrame()
    fou_latest = pd.DataFrame()
    fif_latest = pd.DataFrame()
    
    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":one[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        one_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":two[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        two_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":three[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        three_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":four[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        four_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":five[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        five_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":six[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        six_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
        
    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":seven[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        seven_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":eight[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        eight_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":nine[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        nine_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":ten[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ten_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":ele[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ele_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":twe[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        twe_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":thir[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        thir_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":fou[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fou_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.comments.find({"userId":fif[0]["_id"]})))
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fif_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
        
        
        
    try:
        dfs = [one_latest, two_latest, three_latest, four_latest, five_latest, six_latest, seven_latest, eight_latest, nine_latest, ten_latest, ele_latest, twe_latest, thir_latest, fou_latest, fif_latest]
        dfs_defined = [df for df in dfs if df is not None] 

        if dfs_defined:
            concatenated_df = pd.concat(dfs_defined)
            concatenated_df.reset_index(drop=True, inplace=True)
            concatenated_df['index'] = concatenated_df.reset_index().index + 1
            print(len(concatenated_df))
            if len(concatenated_df) > 0:
                print(concatenated_df.columns)
                concatenated_df = concatenated_df.sort_values(by='updatedAt', ascending=False)
        
                latest_updates = concatenated_df.groupby('username').tail(5)
                latest_updates['MD'] = latest_updates['updatedAt'].dt.strftime('%Y-%m-%dT%H:%M')
                latest_updates = latest_updates.rename(columns={"MD":"Time"})
                #latest_updates = latest_updates.head(15)
                fig = px.bar(latest_updates, x='Time', y='index', color='username', barmode='group', title='Last five comments on posts by the bots')
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
            else:
                return ""
    except Exception as e:
        print("An error occurred:", e) 
        return ""
        
        
@blueprint.route('/BotsTotalLikedComentsRoute', methods=['GET', 'POST'])
def BotsTotalLikedComentsRoute():
    one = db.users.find({'username': 'Weißes Walross'})
    two = db.users.find({'username': 'Weißer Hase'})
    three = db.users.find({'username': 'Schwarzer Ninja'})
    four = db.users.find({'username': 'Oranger Ninja'})
    five = db.users.find({'username': 'Lila Walross'})
    six = db.users.find({'username': 'Lila Krähe'})
    seven = db.users.find({'username': 'Lila Blume'})
    eight = db.users.find({'username': 'Grünes Kaninchen'})
    nine = db.users.find({'username': 'Grauer Otter'})
    ten = db.users.find({'username': 'Graue Krähe'})
    ele = db.users.find({'username': 'Gelber Roboter'})
    twe = db.users.find({'username': 'Blaues Siegel'})
    thir = db.users.find({'username': 'Blaues Huhn'})
    fou = db.users.find({'username': 'Blauer Ninja'})
    fif = db.users.find({'username': 'Blauer Biber'})
    
    one_latest = pd.DataFrame()
    two_latest = pd.DataFrame()
    three_latest = pd.DataFrame()
    four_latest = pd.DataFrame()
    five_latest = pd.DataFrame()
    six_latest = pd.DataFrame()
    seven_latest = pd.DataFrame()
    eight_latest = pd.DataFrame()
    nine_latest = pd.DataFrame()
    ten_latest = pd.DataFrame()
    ele_latest = pd.DataFrame()
    twe_latest = pd.DataFrame()
    thir_latest = pd.DataFrame()
    fou_latest = pd.DataFrame()
    fif_latest = pd.DataFrame()
    
    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":one[0]["_id"]})))
        ppl_flower["username"] = "Weißes Walross"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        one_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":two[0]["_id"]})))
        ppl_flower["username"] = "Weißer Hase"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        two_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":three[0]["_id"]})))
        ppl_flower["username"] = "Schwarzer Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        three_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":four[0]["_id"]})))
        ppl_flower["username"] = "Oranger Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        four_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":five[0]["_id"]})))
        ppl_flower["username"] = "Lila Walross"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        five_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":six[0]["_id"]})))
        ppl_flower["username"] = "Lila Krähe"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        six_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
        
    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":seven[0]["_id"]})))
        ppl_flower["username"] = "Lila Blume"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        seven_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":eight[0]["_id"]})))
        ppl_flower["username"] = "Grünes Kaninchen"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        eight_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":nine[0]["_id"]})))
        ppl_flower["username"] = "Grauer Otter"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        nine_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":ten[0]["_id"]})))
        ppl_flower["username"] = "Graue Krähe"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ten_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":ele[0]["_id"]})))
        ppl_flower["username"] = "Gelber Roboter"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ele_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":twe[0]["_id"]})))
        ppl_flower["username"] = "Blaues Siegel"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        twe_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":thir[0]["_id"]})))
        ppl_flower["username"] = "Blaues Huhn"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        thir_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":fou[0]["_id"]})))
        ppl_flower["username"] = "Blauer Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fou_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.commentlikes.find({"userId":fif[0]["_id"]})))
        ppl_flower["username"] = "Blauer Biber"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fif_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
        
    try:
        dfs = [one_latest, two_latest, three_latest, four_latest, five_latest, six_latest, seven_latest, eight_latest, nine_latest, ten_latest, ele_latest, twe_latest, thir_latest, fou_latest, fif_latest]
        dfs_defined = [df for df in dfs if df is not None] 

        if dfs_defined:
            concatenated_df = pd.concat(dfs_defined)
            concatenated_df.reset_index(drop=True, inplace=True)
            concatenated_df['index'] = concatenated_df.reset_index().index + 1
            print(len(concatenated_df))
            if len(concatenated_df) > 0:
                print(concatenated_df.columns)
                concatenated_df = concatenated_df.sort_values(by='updatedAt', ascending=False)
        
                latest_updates = concatenated_df.groupby('username').tail(5)
                latest_updates['MD'] = latest_updates['updatedAt'].dt.strftime('%Y-%m-%dT%H:%M')
                latest_updates = latest_updates.rename(columns={"MD":"Time"})
                #latest_updates = latest_updates.head(15)
                fig = px.bar(latest_updates, x='Time', y='index', color='username', barmode='group', title='Last five likes on comments by the bots')
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
            else:
                return ""
    except Exception as e:
        print("An error occurred:", e)
        return ""


@blueprint.route('/BotsTotalLikedPostsRoute', methods=['GET', 'POST'])
def BotsTotalLikedPostsRoute():
    one = db.users.find({'username': 'Weißes Walross'})
    two = db.users.find({'username': 'Weißer Hase'})
    three = db.users.find({'username': 'Schwarzer Ninja'})
    four = db.users.find({'username': 'Oranger Ninja'})
    five = db.users.find({'username': 'Lila Walross'})
    six = db.users.find({'username': 'Lila Krähe'})
    seven = db.users.find({'username': 'Lila Blume'})
    eight = db.users.find({'username': 'Grünes Kaninchen'})
    nine = db.users.find({'username': 'Grauer Otter'})
    ten = db.users.find({'username': 'Graue Krähe'})
    ele = db.users.find({'username': 'Gelber Roboter'})
    twe = db.users.find({'username': 'Blaues Siegel'})
    thir = db.users.find({'username': 'Blaues Huhn'})
    fou = db.users.find({'username': 'Blauer Ninja'})
    fif = db.users.find({'username': 'Blauer Biber'})
    
    one_latest = pd.DataFrame()
    two_latest = pd.DataFrame()
    three_latest = pd.DataFrame()
    four_latest = pd.DataFrame()
    five_latest = pd.DataFrame()
    six_latest = pd.DataFrame()
    seven_latest = pd.DataFrame()
    eight_latest = pd.DataFrame()
    nine_latest = pd.DataFrame()
    ten_latest = pd.DataFrame()
    ele_latest = pd.DataFrame()
    twe_latest = pd.DataFrame()
    thir_latest = pd.DataFrame()
    fou_latest = pd.DataFrame()
    fif_latest = pd.DataFrame()
    
    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":one[0]["_id"]})))
        ppl_flower["username"] = "Weißes Walross"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        one_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":two[0]["_id"]})))
        ppl_flower["username"] = "Weißer Hase"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        two_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":three[0]["_id"]})))
        ppl_flower["username"] = "Schwarzer Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        three_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":four[0]["_id"]})))
        ppl_flower["username"] = "Oranger Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        four_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":five[0]["_id"]})))
        ppl_flower["username"] = "Lila Walross"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        five_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":six[0]["_id"]})))
        ppl_flower["username"] = "Lila Krähe"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        six_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
        
    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":seven[0]["_id"]})))
        ppl_flower["username"] = "Lila Blume"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        seven_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":eight[0]["_id"]})))
        ppl_flower["username"] = "Grünes Kaninchen"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        eight_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":nine[0]["_id"]})))
        ppl_flower["username"] = "Grauer Otter"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        nine_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":ten[0]["_id"]})))
        ppl_flower["username"] = "Graue Krähe"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ten_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":ele[0]["_id"]})))
        ppl_flower["username"] = "Gelber Roboter"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        ele_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":twe[0]["_id"]})))
        ppl_flower["username"] = "Blaues Siegel"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        twe_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
        
    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":thir[0]["_id"]})))
        ppl_flower["username"] = "Blaues Huhn"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        thir_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":fou[0]["_id"]})))
        ppl_flower["username"] = "Blauer Ninja"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fou_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)

    try:
        ppl_flower = pd.DataFrame(list(db.postlikes.find({"userId":fif[0]["_id"]})))
        ppl_flower["username"] = "Blauer Biber"
        df_sorted = ppl_flower.sort_values(by='updatedAt', ascending=False)
        fif_latest = df_sorted.head(2)
    except Exception as e:
        print("An error occurred:", e)
    
    try:
        dfs = [one_latest, two_latest, three_latest, four_latest, five_latest, six_latest, seven_latest, eight_latest, nine_latest, ten_latest, ele_latest, twe_latest, thir_latest, fou_latest, fif_latest]
        dfs_defined = [df for df in dfs if df is not None] 

        if dfs_defined:
            concatenated_df = pd.concat(dfs_defined)
            concatenated_df.reset_index(drop=True, inplace=True)
            concatenated_df['index'] = concatenated_df.reset_index().index + 1
            print(len(concatenated_df))
            if len(concatenated_df) > 0:
                print(concatenated_df.columns)
                concatenated_df = concatenated_df.sort_values(by='updatedAt', ascending=False)
        
                latest_updates = concatenated_df.groupby('username').tail(5)
                latest_updates['MD'] = latest_updates['updatedAt'].dt.strftime('%Y-%m-%dT%H:%M')
                latest_updates = latest_updates.rename(columns={"MD":"Time"})
                #latest_updates = latest_updates.head(15)
                fig = px.bar(latest_updates, x='Time', y='index', color='username', barmode='group', title='Last five likes on posts by the bots')
                graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return graphJSON
            else:
                return ""
        
    except Exception as e:
        print("An error occurred:", e) 
        return ""   
    


    