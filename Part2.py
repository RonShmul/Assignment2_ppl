import Part1
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def ExtractCB(ST, SV, k=20, l=20, T=10, eps=0.01):
    global users_profiles
    global items_profiles
    global total_average
    users_profiles = Part1.build_user_profiles(ST)
    items_profiles = Part1.build_item_profiles(ST)
    total_average = np.mean(ST[:, 2])
    u = np.random.randint(k, size=users_profiles.shape[0])
    v = np.random.randint(l, size=items_profiles.shape[0])
    B = calculate_code_book(k, l, u, v)
    t = 1
    rmse = np.inf
    while t < T or rmse > eps:
        for user_id in range(users_profiles.shape[0]): #todo: first user is 1 not 0 ?
            if users_profiles[user_id] is not np.ndarray:
                continue
            u[user_id] = get_cluster(user_id, v, B, k, get_user_error)
        B = calculate_code_book(k, l, u, v)
        for item_id in range(items_profiles.shape[0]): #todo: first item is 1 not 0 ?
            if items_profiles[item_id] is not np.ndarray:
                continue
            v[item_id] = get_cluster(item_id, u, B, l, get_item_error)
        B = calculate_code_book(k, l, u, v)
        rmse = evaluate(SV, u, v, B)
        t =t+1
    return u, v, B


def get_cluster_members(cluster_id, vector, profile):
    return [index for index, cluster in enumerate(vector) if cluster == cluster_id and type(profile[index]) is np.ndarray]


def calculate_code_book_cell(user_cluster_id, item_cluster_id, u, v):
    users = get_cluster_members(user_cluster_id, u, users_profiles)
    items = get_cluster_members(item_cluster_id, v, items_profiles)
    ratings = intersection(users, items)
    if len(ratings) == 0:
        return total_average
    return np.mean(ratings)


def calculate_code_book(k, l, u, v):
    B = np.zeros((k, l))
    for i in range(k):
        for j in range(l):
            B[i][j] = calculate_code_book_cell(i, j, u, v)
    return B


def get_user_error(user_id, v, B, user_cluster):

    sum = 0
    for i, item in enumerate(users_profiles[user_id][0]):
        item_cluster = v[item]
        rating = users_profiles[1][i]
        sum += (rating-B[user_cluster][item_cluster])**2
    return sum


def get_item_error(item_id, u, B, item_cluster):
    sum = 0
    for i, user in enumerate(items_profiles[item_id][0]):
        user_cluster = u[user]
        rating = items_profiles[1][i]
        sum += (rating - B[user_cluster][item_cluster]) ** 2
    return sum


def get_cluster(id, vector, B, num_clusters, get_error):
    min_cluster = 0
    min_error = np.inf
    for cluster in range(num_clusters):
        error = get_error(id, vector, B, cluster)
        if min_error > error:
            min_error = error
            min_cluster = cluster
    return min_cluster


def intersection(users, items):
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


def evaluate(SV, u, v, B):
    prediction = predict(SV, u, v, B)
    return np.sqrt(mean_squared_error(SV[:, 2], prediction))


def predict(SV, u, v, B):
    predictions = []
    for index in range(SV.shape[0]):
        user_cluster = u[int(SV[index, 0])]
        item_cluster = v[int(SV[index, 1])]
        predictions.append(B[user_cluster, item_cluster])
    return predictions


data = np.genfromtxt('ratings.csv', delimiter=',')
np.random.shuffle(data)
ST, SV = train_test_split(data, test_size=0.2, random_state=42)
ExtractCB(ST, SV)
