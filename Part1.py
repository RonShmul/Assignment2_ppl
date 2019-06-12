import numpy as np
import pandas as pd


def ExtractProfiles(ratings_file, profiles_output_dir, items_output_dir):
    profiles, items = build_df(ratings_file)
    profiles.to_csv(profiles_output_dir +'/'+'profile.csv')
    items.to_csv(items_output_dir+'/'+'item.csv')

def build_df(ratings_file):
    data = np.genfromtxt(ratings_file, delimiter=',')
    df = pd.DataFrame(data)
    df.columns = ['userId', 'movieId', 'rating', 'timestamp']
    df = df.drop(0)
    df = df.drop(['timestamp'], axis=1)
    profiles = df.groupby('userId').agg(lambda x: list(x))
    items = df.groupby('movieId').agg(lambda x: list(x))
    return profiles, items

ExtractProfiles('ratings.csv', 'outputs', 'outputs')