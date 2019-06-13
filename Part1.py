import numpy as np
import pandas as pd
import csv


def ExtractProfiles(ratings_file, users_output_dir, items_output_dir):
    data = np.genfromtxt(ratings_file, delimiter=',')
    user_profiles = build_user_profiles(data)
    item_profiles = build_item_profiles(data)
    save_to_csv(user_profiles, users_output_dir + '/' + 'user.csv', ['userId', 'movieId', 'rating'])
    save_to_csv(item_profiles, items_output_dir + '/' + 'item.csv', ['movieId', 'userId', 'rating'])


def build_df(ratings_file):
    data = np.genfromtxt(ratings_file, delimiter=',')
    df = pd.DataFrame(data)
    df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    df = df.drop(0)
    df = df.drop(['timestamp'], axis=1)
    profiles = df.groupby('userId').agg(lambda x: list(x))
    items = df.groupby('movieId').agg(lambda x: list(x))
    return profiles, items


# ExtractProfiles('ratings.csv', 'outputs', 'outputs')


def build_user_profiles(data):
    user_ids = np.unique(data[:, 0], axis=0)
    user_ids = user_ids[~np.isnan(user_ids)]
    profiles = np.zeros(int(np.max(user_ids))+1)
    profiles = profiles.astype(np.object)
    for user_id in user_ids:
        user_data = data[np.where(data[:, 0] == user_id)]
        user_data = np.stack((user_data[:,1], user_data[:,2]))
        profiles[int(user_id)] = user_data
    return profiles


def build_item_profiles(data):
    item_ids = np.unique(data[:, 1], axis=0)
    item_ids = item_ids[~np.isnan(item_ids)]
    profiles = np.zeros(int(np.max(item_ids))+1)
    profiles = profiles.astype(np.object)
    for item_id in item_ids:
        item_data = data[np.where(data[:, 1] == item_id)]
        item_data = np.stack((item_data[:,0], item_data[:,2]))
        profiles[int(item_id)] = item_data
    return profiles


def save_to_csv(profile, file_name, column_names):
    with open(file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(column_names)
        for index in range(profile.shape[0]):
            if type(profile[index]) is not np.ndarray and profile[index] == 0:
                continue
            csv_writer.writerow([index, profile[index][0].tolist(), profile[index][1].tolist()])



ExtractProfiles('ratings.csv', 'outputs', 'outputs')