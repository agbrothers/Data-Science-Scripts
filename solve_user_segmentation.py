import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import argparse
import random
import os
from tqdm import tqdm



# Clustering Analysis to segment users by playstyle and identify retention/monitization differences.  Using a custom built k-means algorithm.  

# CONTEXT:
# Within the game users can watch story content (interactive video) or play match (candy crush style puzzles)
# Playing match helps users earn more stars to consume video content, some users consume stars immediately 
# while others binge match to build up a lot of stars and then binge story.  This algorithm analyzes millions of
# user events to identify paths and segment users into playstyle cohorts.  

# -> Output is a csv with user_id's and associated cluster.  


def listdir_ignore_hidden(directory):
    files = [file for file in os.listdir(directory) if file[0] != '.']
    return files


def load_events(directory):
    tqdm.write('\n\nLOADING EVENT DATA')
    event_data = pd.DataFrame() 
    fields = ['user_id','event_type','event_time']
    for folder in listdir_ignore_hidden(directory):
        for file in tqdm(listdir_ignore_hidden(os.path.join(directory,folder))):
            df = pd.read_csv(os.path.join(directory, folder, file), usecols=fields)
            event_data = event_data.append(df, ignore_index=True)  
            
    key_events = ['m3_level_end', 'story_node_completed']
    event_data = event_data[event_data['event_type'].isin(key_events)]
    event_data = event_data.replace(['story_node_completed','m3_level_end'],[0,1])
    event_data = event_data.sort_values(by=['user_id','event_time'], ascending=[True, True])
    return event_data


def path_metric(events):
    consecutive = []
    count = 1
    for i in range(1,len(events)):
        if events[i] == events[i-1]:
            count+=1
        else:
            consecutive.append(count)
            count=1
    consecutive.append(count)
    return np.mean(consecutive)


def path_behavior(events):
    tqdm.write('\n\nAGGREGATING USER PATHS')
    users = pd.DataFrame()
    for user in tqdm(events.user_id.unique()):
        path = events[events['user_id'] == user].event_type.values
        if len(path) > 10:
            users = users.append({'user_id':user, 'path':path_metric(path)}, ignore_index=True)
    return users


def kmeans(paths,k=3):
    means = np.array([paths['path'].quantile(i/k) for i in range(k)])
    paths['cluster'] = np.zeros(len(paths))
    max_iter = 100
    i = 0
    print('K-means Iterations:')
    while i < max_iter:
        for j,user in paths.iterrows():
            paths.loc[j, 'cluster'] = int(np.argmin(np.abs(means-user.path)))
        prior_means = means
        means = np.array(paths.groupby('cluster')['path'].mean().values)
        if all(means == prior_means):
            break
        i+=1
        print(i,' ', end='', flush=False)
    return paths


def to_coords(path):
    x = [0]
    y = [0]
    for i,event in enumerate(path):
        x.append(x[-1]+(1-event))
        y.append(y[-1]+(event))
    return x,y


def random_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


def draw_paths(users):
    fig,ax = plt.subplots(figsize=(15,11))
    k = len(users.cluster.unique())
    colors = [random_color() for i in range(k)]
    for i,user in users.iterrows():
        plt.plot(user.x, user.y, alpha=0.3, lw=3, color=colors[user.cluster])
    ax.set_xlim([0,200])
    ax.set_ylim([0,100])
    legend = list(np.linspace(1,k,k).astype(int).astype(str))
    ax.legend(handler_map=dict(zip(legend,colors)))
    plt.tight_layout()
        
        
def plot_clusters(events, clusters):
    plots = pd.DataFrame()
    for i,user in clusters.iterrows():
        path = events[events['user_id'] == user.user_id]['event_type'].values
        x,y = to_coords(path)
        plots = plots.append({'user_id':user.user_id,'x':x,'y':y,'cluster':user.cluster}, ignore_index=True)
    draw_paths(plots)    


if __name__ == "__main__":
    # Read inputs from terminal
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_dir', '-s', type=str, default='events',
                        help='Source directory for raw event data')
    parser.add_argument('--dst_dir', '-d', type=str, default='events',
                        help='Destination directory for resulting user segment csv')
    args, _ = parser.parse_known_args()
    
    
    # Execute the script
    directory = '/Users/greysonbrothers/Desktop/ /work/SOLVE/data/events'
    events = load_events(args.src_dir) # args.src_dir
    behavior = path_behavior(events)
    clusters = kmeans(behavior.copy(),k=3)
    clusters[['user_id','cluster']].to_csv(args.dst_dir, index=False)
    
    # Visualize results
    print('Users Per Cluster: \n', clusters.groupby('cluster').count())
    print('Means: \n', clusters.groupby('cluster').mean())
    plot_clusters(events, clusters)
    
    
    