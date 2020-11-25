import networkx as nx
import pandas as pd
import os
from ast import literal_eval
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.io_util import save_obj
from utils.relatedness import evaluate_katz_relatedness
from utils.timer import timer

similarities_file = '{0}_{1}_{2}_similarities.txt'

yahoo_movies = 'yahoo_movies'
yahoo_movies_2_hops = 'yahoo_movies_2_hops'
small_library_thing = 'SmallLibraryThing'
small_library_thing_2_hops = 'SmallLibraryThing2Hops'

project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
selection_types = ['categorical', 'ontological', 'factual']
datasets = [small_library_thing, small_library_thing_2_hops, yahoo_movies, yahoo_movies_2_hops]
dataset = datasets[0]
topk = 10
top_k_similar_items = 0.25

# Katz Parameters
alpha = 0.25
# 0.25 is the best value in:
# Path-based Semantic Relatedness on Linked Data and its use to Word and Entity Disambiguation
# authored by Ioana Hulpu¸s, Narumol Prangnawarat, Conor Hayes


def build():
    df_map = pd.read_csv(os.path.join(project_dir, 'data', dataset, 'df_map.csv'))
    df_selected_features = pd.read_csv(os.path.join(project_dir, 'data', dataset, 'selected_features.csv'))
    df_features = pd.read_csv(os.path.join(project_dir, 'data', dataset, "features.tsv"), sep='\t', header=None)
    df_target_items = pd.read_csv(os.path.join(project_dir, 'data', dataset, "target_items.csv"))
    df_ratings = pd.read_csv(os.path.join(project_dir, 'data', dataset, "ratings.csv"))
    df_ratings = df_ratings.iloc[:, :3]
    df_ratings.columns = ['userId', 'itemId', 'rating']

    items = df_ratings.itemId.unique()
    df_map = df_map[df_map['item'].isin(items)]

    for i in range(len(df_selected_features)):
        start_type = time.time()
        id_features, selection_type = literal_eval(df_selected_features.iloc[i].features), df_selected_features.iloc[
            i].type
        print(selection_type)
        # Take all available features
        df_feature_uris = df_features[df_features[0].isin(id_features)]
        df_item_features = df_map[df_map['feature'].isin(id_features)]
        # Create Graph
        G = nx.Graph()

        # Add items nodes
        for item in items:
            G.add_node(item)

        # Add Object Nodes
        indexer_subjects = {}  # Indexer to Simply the MAP Creation
        subject_id = 0
        for i, row in df_feature_uris.iterrows():
            uri = row[1]
            uri = uri.split('><')[1][:-1]
            if uri not in indexer_subjects:
                while subject_id in items:
                    subject_id += 1
                indexer_subjects[uri] = subject_id
                # print(subject_node)
                G.add_node(subject_id)
                subject_id += 1
            # Else it means that we have already a subject with the same URI but coming from a different property

        # Add edges
        for i, row in df_item_features.iterrows():
            item_id, feature_id = int(row['item']), int(row['feature'])
            uri = df_feature_uris[df_feature_uris[0] == feature_id][1].values[0]
            uri = uri.split('><')[1][:-1]
            G.add_edge(item_id, indexer_subjects[uri])

        print('Start Path Exploration of {}'.format(selection_type))
        start_path_exploration = time.time()
        # Evaluate the Shortest Path with Dijkstra
        hashmap_shortest_paths = {}
        for target_item_id in df_target_items.itemId.tolist():
            start_target_item_id = time.time()
            print("\tTarget Item: {}".format(target_item_id))
            hashmap_shortest_paths[target_item_id] = {}
            for num_item, item in enumerate(df_item_features[df_item_features['item'] != target_item_id].item.unique()):
                # print("\t{}".format(item))
                item = int(item)
                hashmap_shortest_paths[target_item_id][item] = []
                all_simple_paths = nx.shortest_simple_paths(G, target_item_id, item)
                # for num, path in enumerate(reversed_iterator(all_simple_paths)):
                for num, path in enumerate(all_simple_paths):
                    # print("\t\t{}".format(len(path[1:-1])))
                    hashmap_shortest_paths[target_item_id][item].append(path[1:-1])
                    if num == topk:
                        break

                if num_item % 500 == 0 and num_item > 0:
                    print("\t\tExplored: {0}/{1}".format(num_item, len(df_item_features.item.unique()) - 1))

            print("\t--> End Exploration in {}".format(timer(start_target_item_id, time.time())))
        print("--> End Path Exploration of {} in {}".format(selection_type, timer(start_path_exploration, time.time())))

        # Store Object

        save_obj(hashmap_shortest_paths,
                 os.path.join(project_dir, 'data', dataset, 'similarities',
                              'dict_{0}_exploration'.format(selection_type)))
        save_obj(indexer_subjects,
                 os.path.join(project_dir, 'data', dataset, 'similarities',
                              'subject_uri_indexer_{0}_exploration'.format(selection_type)))

        # Measure Metrics
        similar_items = evaluate_katz_relatedness(hashmap_shortest_paths, alpha, top_k_similar_items)
        similar_items.to_csv(os.path.join(project_dir, 'data', dataset, 'similarities',
                                          similarities_file.format('katz{0}'.format(alpha), 'target', selection_type)), index=None)

        print("\n\n{0} Katz relatedness file WRITTEN on {1}".format(selection_type, dataset))

        print("**** END in {} ****".format(timer(start_type, time.time())))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    build()
