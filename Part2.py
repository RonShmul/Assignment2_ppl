import Part1
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask, jsonify, request, current_app
import sys


app = Flask(__name__)

global u
global v
global B


def create_ST_SV(path_file):
    data = np.genfromtxt(path_file, delimiter=',', skip_header=1)

    np.random.shuffle(data)
    ST, SV = train_test_split(data, test_size=0.2, random_state=42)
    return ST, SV


def ExtractCB(path_file, k=20, T=10, eps=0.01):
    ST, SV = create_ST_SV(path_file)
    l=k
    global users_profiles
    global items_profiles
    global total_average
    global u
    global v
    global B
    users_profiles = Part1.build_user_profiles(ST)
    items_profiles = Part1.build_item_profiles(ST)
    total_average = np.mean(ST[:, 2])
    u = np.random.randint(k, size=users_profiles.shape[0])
    v = np.random.randint(l, size=items_profiles.shape[0])
    B = calculate_code_book(k, l, u, v)
    t = 1
    rmse = np.inf
    while t < T and rmse > eps:
        user_ids = np.unique(ST[:, 0], axis=0)
        for user_id in user_ids:
            u[int(user_id)] = get_cluster(user_id, v, B, k, get_user_error)
        B = calculate_code_book(k, l, u, v)
        item_ids = np.unique(ST[:, 1], axis=0)
        for item_id in item_ids:
            v[int(item_id)] = get_cluster(item_id, u, B, l, get_item_error)
        B = calculate_code_book(k, l, u, v)
        rmse = evaluate(SV, u, v, B)
        t = t + 1
    pd.DataFrame(u).to_csv('u.csv', header=False)
    pd.DataFrame(v).to_csv('v.csv', header=False)
    pd.DataFrame(B[:,1:]).to_csv('B.csv', header=False)
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
    for i, item in enumerate(users_profiles[int(user_id)][0]):
        item_cluster = v[int(item)]
        rating = users_profiles[int(user_id)][1][i]
        sum += (rating-B[user_cluster][item_cluster])**2
    return sum


def get_item_error(item_id, u, B, item_cluster):
    sum = 0
    for i, user in enumerate(items_profiles[int(item_id)][0]):
        user_cluster = u[int(user)]
        rating = items_profiles[int(item_id)][1][i]
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
        try:
            user_cluster = u[int(SV[index, 0])]
            item_cluster = v[int(SV[index, 1])]
            predictions.append(B[user_cluster, item_cluster])
        except KeyError:
            predictions.append(total_average)
        except IndexError:
            predictions.append(total_average)
    return predictions


def get_recommendations(user_id, ST, n=10):
    u = np.genfromtxt('u.csv', delimiter=',')
    v = np.genfromtxt('v.csv', delimiter=',')
    B = np.genfromtxt('B.csv', delimiter=',')
    rated_items = users_profiles[user_id][0]
    items = np.unique(ST[:, 1], axis=0)
    not_rated_items = [item for item in items if item not in rated_items]
    recommendations = pd.DataFrame(data=not_rated_items, columns=['items'])
    user_cluster = u[int(user_id)]
    recommendations['ratings'] = recommendations['items'].apply(lambda item_id: B[user_cluster][v[int(item_id)]])
    return recommendations.sort_values(by=['ratings'], ascending=False).iloc[0:n, 0].tolist()


@app.route('/', methods=['POST'])
def get_predictions():
    data = request.form
    user_id = data['userid']
    n = data['n']
    ST, SV = create_ST_SV(sys.argv[1])
    return jsonify(get_recommendations(user_id, ST, n))


if __name__ == '__main__':
    print('start train')
    ExtractCB(sys.argv[1], k=10)
    print('done train')
    with app.app_context():
        print(current_app.name)
    app.run()
