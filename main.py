import os
import time

import numpy as np
import pandas as pd

from graphs import Graphs


def main():
    df = pd.read_csv('data/movies_metadata.csv')
    print(df[['title', 'original_title']].dropna())
    # merge_data()
    # graph = Graphs()
    # graph.create_graphs('example')
    pass

def merge_data(verbose=False):
    # Will have to merge critic_reviews into tomatoes_movies on rotten_tomatoes_link, then into master on title
    
    # Nathan please comment this so it looks a bit more readable instead of just dense code
    credits_df = pd.read_csv('data/credits.csv', usecols=['id', 'cast'], low_memory=False)
    links_df = pd.read_csv('data/links.csv', usecols=['movieId', 'imdbId'], low_memory=False)
    expert_df = pd.read_csv('data/expert_ratings.csv')

    if ~os.path.exists('data/rating_averages.csv'):
        rating_df = pd.read_csv('data/ratings.csv.gz', usecols=['movieId', 'userId', 'rating'], compression='gzip', low_memory=False)
        rating_avg_df = rating_df.groupby('movieId')['rating'].mean()
        rating_avg_df.to_csv('data/rating_averages.csv', index=False)
    else:
        rating_avg_df = pd.read_csv('data/rating_averages.csv')
    rating_avg_df = rating_avg_df.apply(lambda n: n * 20)
    print(f'Ratings average df has data for {len(rating_avg_df)} movies') if verbose else None

    metadata_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
    metadata_df['id'] = metadata_df['id'].astype('int32')
    metadata_df.drop(['belongs_to_collection', 'homepage', 'overview', 'poster_path', 'status', 
                      'spoken_languages', 'tagline', 'Unnamed: 0'], axis='columns', inplace=True)

    master_df = pd.merge(credits_df, metadata_df, on='id')
    print(f'After FIRST  merge, length is {len(master_df)}') if verbose else None
    
    master_df = pd.merge(master_df, rating_avg_df, left_on='id', right_on='movieId', how='left')
    master_df.rename(columns={'rating': 'user_rating'}, inplace=True)
    print(f'After SECOND merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    master_df = pd.merge(master_df, links_df, left_on='id', right_on='movieId', how='left')
    print(f'After THIRD  merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    master_df = pd.merge(master_df, expert_df, left_on='original_title', right_on='title', how='left').drop(['title_x', 'title_y'], axis='columns')
    master_df.rename(columns={'rating': 'expert_rating'}, inplace=True)
    print(f'After FOURTH merge, length is {len(master_df)}\nColumns are: {master_df.columns}') if verbose else None
    
    # Save merged dataframe to CSV
    master_df = master_df.loc[:, ['id', 'imdb_id', 'original_title', 'release_date', 'adult', 'budget', 'runtime', 'revenue', 'user_rating', 'expert_rating', 'vote_average', 'vote_count', 'genre', 'original_language', 'popularity', 'production_companies', 'production_countries', 'cast', ]]
    master_df.to_csv("data/master_dataset.csv", index=False)

# Method to test the time difference between opening a csv file vs a gzipped csv file
def test_times():
    for file in ['credits', 'keywords', 'links', 'movies_metadata', 'ratings']:
        t1 = time.time()
        credits = pd.read_csv(f'data/gzips/{file}.csv.gz', compression='gzip')
        t2 = time.time()
        tot1 = t2-t1
        print(f'Took {tot1} seconds to open the GZIP File')

        credits.to_csv(f'data/{file}.csv', index=False)

        t3 = time.time()
        credits2 = pd.read_csv(f'data/{file}.csv')
        t4 = time.time()
        tot2 = t4-t3
        print(f'Took {tot2} seconds to open the CSV File')

        print(f'Opening a CSV is {tot1-tot2} seconds faster than a GZIP')

if __name__ == "__main__":
    main()
