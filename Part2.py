import Part1

def ExtractCB():
    pass

def get_cluster_members(cluster_id, vector):
    return [index for index, cluster in enumerate(vector) if cluster == cluster_id]

def calculate_code_book(user_id, item_id, u, v):
    user_cluster_id = u[user_id]
    item_cluster_id = v[item_id]
    users = get_cluster_members(user_cluster_id, u)
    items = get_cluster_members(item_cluster_id, v)


def intersection(users, items):
    users_profiles, items_profiles = Part1.build_df('ratings.csv')
    if len(users)<len(items):
        for user in users:
            user_ratings = users_profiles.iloc[users_profiles['userId'] == user]
            #user_ratings['']