import pandas as pd
import numpy as np


## Template
# itemId, similar_items
# 7733.0, "[109, 121313, 120219, 119605, 122450]"
# num similar :int(len(similarities_for_item.argsort()[0])*cfg.top_k_similar_items)

def evaluate_katz_relatedness(hashmap_shortest_paths, alpha, top_k_similar_items):
    """

    :param hashmap_shortest_paths:
    :param alpha:  the effectiveness of a link between two nodes is governed by a known, constant probability, α
    :param top_k_similar_items: The quartile of items to save
    :return:
    """
    similar_items = pd.DataFrame()

    for target_item in hashmap_shortest_paths.keys():
        rel_target_end_at1, rel_target_end_at5, rel_target_end_at10 = [], [], []
        for end_item in hashmap_shortest_paths[target_item].keys():
            top_k_paths = hashmap_shortest_paths[target_item][end_item]
            # +1 since we have remove the head and tail of the path and the length is the number of edges
            rel_target_end_at1.append(np.mean([np.power(alpha, len(p) + 1) for p in top_k_paths[:1]]))
            rel_target_end_at5.append(np.mean([np.power(alpha, len(p) + 1) for p in top_k_paths[:5]]))
            rel_target_end_at10.append(np.mean([np.power(alpha, len(p) + 1) for p in top_k_paths[:10]]))

        # For now we are storing the list based on katz_rel_at_10
        end_items = list(hashmap_shortest_paths[target_item].keys())
        top_similar_index_ids = np.argsort(rel_target_end_at10)[::-1][
                                :int(len(end_items) * top_k_similar_items)]  # Sort from bigger to smaller
        similar_items = similar_items.append({
            'itemId': int(target_item),
            'similar_items': [end_items[top_similar_index_id] for top_similar_index_id in
                              top_similar_index_ids]
        }, ignore_index=True)

    return similar_items

