import Part1
import numpy as np

def ExtractCB(ST, SV, k=20, l=20, T=10, eps=0.01):
    users_profiles, items_profiles = Part1.build_df('ratings.csv')
    u = np.random.randint(k, users_profiles.shape[0])
    v = np.random.randint(l, items_profiles.shape[0])
    B = calculate_code_book(k, l, u, v)
    t = 1
    rmse = np.inf()
    while t < T or rmse > eps:
        for user_id in range(users_profiles.shape[0]): #todo: first user is 1 not 0 ?
            u[user_id] = get_cluster(user_id, v, B, k, get_user_error)
        B = calculate_code_book(k, l, u, v)
        for item_id in range(items_profiles.shape[0]): #todo: first item is 1 not 0 ?
            v[item_id] = get_cluster(item_id, u, B, l, get_item_error)
        B = calculate_code_book(k, l, u, v)
        rmse = evaluate(SV)
        t =t+1
    return u, v, B


def get_cluster_members(cluster_id, vector):
    return [index for index, cluster in enumerate(vector) if cluster == cluster_id]


def calculate_code_book_cell(user_cluster_id, item_cluster_id, u, v):
    users = get_cluster_members(user_cluster_id, u)
    items = get_cluster_members(item_cluster_id, v)
    ratings = intersection(users, items)
    return np.mean(ratings)


def calculate_code_book(k, l, u, v):
    B = np.zeros((k, l))
    for i in range(k):
        for j in range(l):
            B[i][j] = calculate_code_book_cell(i, j, u, v)
    return B


def get_user_error(user_id, v, B, user_cluster):
    users_profiles, items_profiles = Part1.build_df('ratings.csv')
    sum = 0
    for i, item in enumerate(users_profiles[user_id][0]):
        item_cluster = v[item]
        rating = users_profiles[1][i]
        sum += (rating-B[user_cluster][item_cluster])**2
    return sum


def get_item_error(item_id, u, B, item_cluster):
    users_profiles, items_profiles = Part1.build_df('ratings.csv')
    sum = 0
    for i, user in enumerate(items_profiles[item_id][0]):
        user_cluster = u[user]
        rating = items_profiles[1][i]
        sum += (rating - B[user_cluster][item_cluster]) ** 2
    return sum


def get_cluster(id, vector, B, num_clusters, get_error):
    min_cluster = 0
    min_error = np.inf()
    for cluster in range(num_clusters):
        error = get_error(id, vector, B, cluster)
        if min_error > error:
            min_error = error
            min_cluster = cluster
    return min_cluster


def intersection(users, items):
    users_profiles, items_profiles = Part1.build_df('ratings.csv')
    ratings = []
    if len(users)<len(items):
        for user in users:
            user_ratings = users_profiles[user]
            ratings += [user_ratings[1][i] for i,item in enumerate(user_ratings[0]) if item in items]
    else:
        for item in items:
            item_ratings = items_profiles[item]
            ratings += [item_ratings[1][i] for i,user in enumerate(item_ratings[0]) if user in users]
    return ratings


def evaluate(SV):
    return 0 # todo!!!