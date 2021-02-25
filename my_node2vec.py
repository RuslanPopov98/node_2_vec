import random
import numpy as np
import pandas as pd
import networkx as nx
import os
from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec
#import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_walks(d_graph, length_walk, num_walks):
                
    walks = list()
    
    for i in range(num_walks):
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)
        
        for source in shuffled_nodes:
            walk = [source]
            walk_length = length_walk
            
            while len(walk) < walk_length:
                walk_options = d_graph[walk[-1]].get('neighbors', None)
                
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]]['first_waight']
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]]['probabilities'][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                    
                walk.append(walk_to)

            walk = list(map(str, walk))
            walks.append(walk)
    return walks

def change_graph(graph: nx.Graph, p: float=1., q: float=1.):
    dict_graph=defaultdict(dict)

    for sourse in graph.nodes:

        if 'probabilities' not in dict_graph[sourse]:
            dict_graph[sourse]['probabilities'] = {}
        
        for cur_node in list(graph.neighbors(sourse)):
            
            w = []
            
            if 'probabilities' not in dict_graph[cur_node]:
                dict_graph[cur_node]['probabilities'] = {}
                
            for dest in list(graph.neighbors(cur_node)):
                loc_weight = 0.
                if dest == sourse:
                    loc_weight = 1/p
                elif dest in graph[sourse]:
                    loc_weight = 1.
                else:
                    loc_weight = 1/q
                
                w.append(loc_weight)
                
            list_weights = np.array(w)
            dict_graph[cur_node]['probabilities'][sourse] = list_weights/list_weights.sum()
                    
        first_travel_weights = []
        for dest in list(graph.neighbors(sourse)):
            first_travel_weights.append(1)
            
        first_travel_weights = np.array(first_travel_weights)
        dict_graph[sourse]['first_waight'] = first_travel_weights/first_travel_weights.sum()
        
        dict_graph[sourse]['neighbors'] = list(graph.neighbors(sourse))
        
    return dict_graph

'''
a = nx.barabasi_albert_graph(5,2)
b = nx.barabasi_albert_graph(10,2)
c = nx.barabasi_albert_graph(18,2)

graph1 = nx.union(a, b, rename=('a-', 'b-'))
graph = nx.union(graph1,c, rename=('', 'c-'))
graph.add_edge('a-0', 'b-0')
nx.draw_networkx(graph,with_labels=True,node_size=70)
plt.show()

dict_graph = change_graph(graph)
walks = generate_walks(dict_graph, 16, 100)

model = Word2Vec(walks, size = 20, min_count=1, window=10)
embeddings = np.array([model.wv[str(x)] for x in graph.nodes])

tsne = TSNE(n_components=2, random_state=7, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

team_colors = {'a':'r', 'b':'b', 'c':'teal'}
df = pd.DataFrame(list(model.wv.vocab.items()), columns=['node','adress'])
df['color'] = df['node'].apply(lambda x: team_colors[x[0]])

node_colors = dict(zip(df['node'], df['color']))
colors = [node_colors[x] for x in graph.nodes]
figure = plt.figure(figsize=(11, 9))
ax = figure.add_subplot(111)
ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)
'''

'''
----------------------TEST ON FIFA------------------
'''
df=pd.read_csv(r'C:\Users\79776\Desktop\algothitm\Node2vec\FullData.csv',usecols=['Name', 'Club', 'Club_Position', 'Rating'])
df.columns=list(map(str.lower, df.columns))
reformat_string=lambda x:str.lower(x).replace(' ', '_')
df['name']=df['name'].apply(reformat_string)
df['club']=df['club'].apply(reformat_string)
df['club_position']=df['club_position'].str.lower()
df = df[(df['club_position'] != 'sub') & (df['club_position'] != 'res')]
fix_positions = {'rcm' : 'cm', 'lcm': 'cm', 'rcb': 'cb', 'lcb': 'cb', 'ldm': 'cdm', 'rdm': 'cdm'}
df['club_position'] = df['club_position'].apply(lambda x: fix_positions.get(x, x))

clubs = {'real_madrid', 'manchester_utd', 
         'manchester_city', 'chelsea', 'juventus', 
         'fc_bayern', 'napoli'}
df=df[df['club'].isin(clubs)]

assert all(n_players == 11 for n_players in df.groupby('club')['name'].nunique())

FORMATIONS = {'4-3-3_4': {'gk': ['cb_1', 'cb_2'],                           # Real madrid
                          'lb': ['lw', 'cb_1', 'cm_1'],
                          'cb_1': ['lb', 'cb_2', 'gk'],
                          'cb_2': ['rb', 'cb_1', 'gk'],
                          'rb': ['rw', 'cb_2', 'cm_2'],
                          'cm_1': ['cam', 'lw', 'cb_1', 'lb'],
                          'cm_2': ['cam', 'rw', 'cb_2', 'rb'],
                          'cam': ['cm_1', 'cm_2', 'st'],
                          'lw': ['cm_1', 'lb', 'st'],
                          'rw': ['cm_2', 'rb', 'st'],
                          'st': ['cam', 'lw', 'rw']},
              '5-2-2-1': {'gk': ['cb_1', 'cb_2', 'cb_3'],                   # Chelsea
                          'cb_1': ['gk', 'cb_2', 'lwb'],
                          'cb_2': ['gk', 'cb_1', 'cb_3', 'cm_1', 'cb_2'],
                          'cb_3': ['gk', 'cb_2', 'rwb'],
                          'lwb': ['cb_1', 'cm_1', 'lw'],
                          'cm_1': ['lwb', 'cb_2', 'cm_2', 'lw', 'st'],
                          'cm_2': ['rwb', 'cb_2', 'cm_1', 'rw', 'st'],
                          'rwb': ['cb_3', 'cm_2', 'rw'],
                          'lw': ['lwb', 'cm_1', 'st'],
                          'st': ['lw', 'cm_1', 'cm_2', 'rw'],
                          'rw': ['st', 'rwb', 'cm_2']},
              '4-3-3_2': {'gk': ['cb_1', 'cb_2'],                           # Man UTD / CITY
                          'lb': ['cb_1', 'cm_1'],
                          'cb_1': ['lb', 'cb_2', 'gk', 'cdm'],
                          'cb_2': ['rb', 'cb_1', 'gk', 'cdm'],
                          'rb': ['cb_2', 'cm_2'],
                          'cm_1': ['cdm', 'lw', 'lb', 'st'],
                          'cm_2': ['cdm', 'rw', 'st', 'rb'],
                          'cdm': ['cm_1', 'cm_2', 'cb_1', 'cb_2'],
                          'lw': ['cm_1', 'st'],
                          'rw': ['cm_2', 'st'],
                          'st': ['cm_1', 'cm_2', 'lw', 'rw']},              # Juventus, Bayern
              '4-2-3-1_2': {'gk': ['cb_1', 'cb_2'],
                            'lb': ['lm', 'cdm_1', 'cb_1'],
                            'cb_1': ['lb', 'cdm_1', 'gk', 'cb_2'],
                            'cb_2': ['rb', 'cdm_2', 'gk', 'cb_1'],
                            'rb': ['cb_2', 'rm', 'cdm_2'],
                            'lm': ['lb', 'cdm_1', 'st', 'cam'],
                            'rm': ['rb', 'cdm_2', 'st', 'cam'],
                            'cdm_1': ['lm', 'cb_1', 'rb', 'cam'],
                            'cdm_2': ['rm', 'cb_2', 'lb', 'cam'],
                            'cam': ['cdm_1', 'cdm_2', 'rm', 'lm', 'st'],
                            'st': ['lm', 'rm', 'cam']},
              '4-3-3': {'gk': ['cb_1', 'cb_2'],                             # Napoli
                        'lb': ['cb_1', 'cm_1'],
                        'cb_1': ['lb', 'cb_2', 'gk', 'cm_2'],
                        'cb_2': ['rb', 'cb_1', 'gk', 'cm_2'],
                        'rb': ['cb_2', 'cm_3'],
                        'cm_1': ['cm_2', 'lw', 'lb'],
                        'cm_3': ['cm_2', 'rw', 'rb'],
                        'cm_2': ['cm_1', 'cm_3', 'st', 'cb_1', 'cb_2'],
                        'lw': ['cm_1', 'st'],
                        'rw': ['cm_3', 'st'],
                        'st': ['cm_2', 'lw', 'rw']}}

add_club_suffix = lambda x, c: x + '_{}'.format(c)
         
import networkx as nx
from collections import deque  
graph=nx.Graph()
formatted_positions = set()

def club2graph(club_name, formation, graph):
    club_data = df[df['club'] == club_name]
    
    club_formation = FORMATIONS[formation]
    
    club_positions = dict()
    
    # Assign positions to players
    available_positions = deque(club_formation)
    available_players = set(zip(club_data['name'], club_data['club_position']))
    
    roster = dict()  # Here we will store the assigned players and positions
    
    while available_positions:
        position = available_positions.pop()
        name, pos = [(name, position) for name, p in available_players if position.startswith(p)][0]        
        
        roster[name] = pos
        
        available_players.remove((name, pos.split('_')[0]))
        
    reverse_roster = {v: k for k, v in roster.items()}
    # Build the graph
    for name, position in roster.items():
        # Connect to team name
        graph.add_edge(name, club_name)
        
        # Inter team connections
        for teammate_position in club_formation[position]:
            # Connect positions
            graph.add_edge(add_club_suffix(position, club_name),
                           add_club_suffix(teammate_position, club_name))
            
            # Connect player to teammate positions
            graph.add_edge(name, add_club_suffix(teammate_position, club_name))
            
            # Connect player to teammates
            graph.add_edge(name, reverse_roster[teammate_position])
            
            # Save for later trimming
            formatted_positions.add(add_club_suffix(position, club_name))
            formatted_positions.add(add_club_suffix(teammate_position, club_name))
            
    return graph

teams = [('real_madrid', '4-3-3_4'), 
         ('chelsea', '5-2-2-1'),
         ('manchester_utd', '4-3-3_2'),
         ('manchester_city', '4-3-3_2'),
         ('juventus', '4-2-3-1_2'),
         ('fc_bayern', '4-2-3-1_2'),
         ('napoli', '4-3-3')]                                                          
               
graph = club2graph('real_madrid', '4-3-3_4', graph)

for team, formation in teams:
    graph = club2graph(team, formation, graph)

fix_formatted_positions = lambda x: x.split('_')[0] if x in formatted_positions else x

dict_graph = change_graph(graph)
walks = generate_walks(dict_graph, 16, 100)

reformatted_walks = [list(map(fix_formatted_positions, walk)) for walk in walks]
walks = reformatted_walks
model = Word2Vec(walks, size = 20, min_count=1, window=10)
player_nodes = [x for x in model.wv.vocab if len(x) > 3 and x not in clubs]
embeddings = np.array([model.wv[x] for x in player_nodes])

tsne = TSNE(n_components=2, random_state=7, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

team_colors = {
    'real_madrid': 'lightblue',
    'chelsea': 'b',
    'manchester_utd': 'r',
    'manchester_city': 'teal',
    'juventus': 'gainsboro',
    'napoli': 'deepskyblue',
    'fc_bayern': 'tomato' }

df['color'] = df['club'].apply(lambda x: team_colors[x])
player_colors = dict(zip(df['name'], df['color']))
colors = [player_colors[x] for x in player_nodes]

figure = plt.figure(figsize=(11, 9))

ax = figure.add_subplot(111)

ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

team_patches = [mpatches.Patch(color=color, label=team) for team, color in team_colors.items()]
ax.legend(handles=team_patches);




















'''
def fit(walks, dimensions, **skip_gram_params) -> gensim.models.word2vec:
    skip_gram_params['workers'] = 2
    skip_gram_params['size'] = dimensions
    return gensim.models.word2vec(walks, **skip_gram_params)

model= fit(walks, 5)
print(model)


d_graph = {0: {'probabilities': {2: [0.25, 0.25, 0.25, 0.25], 
                                     3: [0.25, 0.25, 0.25, 0.25], 
                                     4: [0.25, 0.25, 0.25, 0.25], 
                                     5: [0.25, 0.25, 0.25, 0.25]}, 
                   'first_travel_key': [0.25, 0.25, 0.25, 0.25], 
                   'neighbors': [2, 3, 4, 5]}, 
               2: {'probabilities': {0: [0.33333333, 0.33333333, 0.33333333], 
                                     1: [0.33333333, 0.33333333, 0.33333333], 
                                     3: [0.33333333, 0.33333333, 0.33333333]}, 
                   'first_travel_key': [0.33333333, 0.33333333, 0.33333333], 
                   'neighbors': [0, 1, 3]}, 
               3: {'probabilities': {0: [0.2, 0.2, 0.2, 0.2, 0.2], 
                                     2: [0.2, 0.2, 0.2, 0.2, 0.2], 
                                     4: [0.2, 0.2, 0.2, 0.2, 0.2], 
                                     5: [0.2, 0.2, 0.2, 0.2, 0.2], 
                                     7: [0.2, 0.2, 0.2, 0.2, 0.2]}, 
                   'first_travel_key': [0.2, 0.2, 0.2, 0.2, 0.2], 
                   'neighbors': [0, 2, 4, 5, 7]}, 
               4: {'probabilities': {0: [0.33333333, 0.33333333, 0.33333333], 
                                     3: [0.33333333, 0.33333333, 0.33333333], 
                                     6: [0.33333333, 0.33333333, 0.33333333]}, 
                   'first_travel_key': [0.33333333, 0.33333333, 0.33333333], 
                   'neighbors': [0, 3, 6]}, 
               5: {'probabilities': {0: [0.33333333, 0.33333333, 0.33333333], 
                                     3: [0.33333333, 0.33333333, 0.33333333], 
                                     6: [0.33333333, 0.33333333, 0.33333333]}, 
                   'first_travel_key': [0.33333333, 0.33333333, 0.33333333], 
                   'neighbors': [0, 3, 6]}, 
               1: {'probabilities': {2: [1.]}, 
                   'first_travel_key': [1.], 
                   'neighbors': [2]}, 
               7: {'probabilities': {3: [0.5, 0.5], 
                                     6: [0.5, 0.5]}, 
                   'first_travel_key': [0.5, 0.5], 
                   'neighbors': [3, 6]}, 
               6: {'probabilities': {4: [0.33333333, 0.33333333, 0.33333333], 
                                     5: [0.33333333, 0.33333333, 0.33333333], 
                                     7: [0.33333333, 0.33333333, 0.33333333]}, 
                   'first_travel_key': [0.33333333, 0.33333333, 0.33333333], 
                   'neighbors': [4, 5, 7]}}

model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=1)
'''













