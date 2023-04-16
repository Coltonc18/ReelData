import pandas as pd
import os
import numpy as np
import time

def main():
    # merge_data()
    pass

def merge_data():
    credits_df = pd.read_csv('data/credits.csv.gz', usecols=['id', 'cast'], low_memory=False)
    links_df = pd.read_csv('data/links.csv.gz', usecols=['movieId', 'imdbId'], low_memory=False)

    if ~os.path.exists('data/rating_averages.csv'):
        rating_df = pd.read_csv('data/ratings.csv.gz', usecols=['movieId', 'userId', 'rating'], compression='gzip', low_memory=False)
        rating_avg_df = rating_df.groupby('movieId')['rating'].mean()
        rating_avg_df.to_csv('data/rating_averages.csv', index=False)
    else:
        rating_avg_df = pd.read_csv('data/rating_averages.csv')
    print(f'Ratings average df has data for {len(rating_avg_df)} movies')

    metadata_df = pd.read_csv('data/movies_metadata.csv.gz', low_memory=False)
    metadata_df['id'] = metadata_df['id'].astype('int32')
    metadata_df.drop(['belongs_to_collection', 'homepage', 'overview', 'poster_path', 'status', 
                      'spoken_languages', 'tagline', 'Unnamed: 0'], axis='columns', inplace=True)

    master_df = pd.merge(credits_df, metadata_df, on='id')
    # print(f'After FIRST  merge, length is {len(master_df)}')
    master_df = pd.merge(master_df, rating_avg_df, left_on='id', right_on='movieId', how='left')
    # print(f'After SECOND merge, length is {len(master_df)}')
    master_df = pd.merge(master_df, links_df, left_on='id', right_on='movieId', how='left')
    # print(f'After THIRD  merge, length is {len(master_df)}, columns are: {master_df.columns}')
    
    # Save merged dataframe to CSV
    master_df.to_csv("data/master_dataset.csv", index=False)


# Method to test the time difference between opening a csv file vs a gzipped csv file
def test_times():
    for file in ['credits', 'keywords', 'links', 'movies_metadata', 'ratings']:
        t1 = time.time()
        credits = pd.read_csv(f'data/{file}.csv.gz', compression='gzip')
        t2 = time.time()
        tot1 = t2-t1
        # print(f'Took {tot1} seconds to open the GZIP File')

        credits.to_csv(f'data/{file}.csv', index=False)

        t3 = time.time()
        credits2 = pd.read_csv(f'data/{file}.csv')
        t4 = time.time()
        tot2 = t4-t3
        # print(f'Took {tot2} seconds to open the CSV File')

        print(f'Opening a CSV is {tot1-tot2} seconds faster than a GZIP')

if __name__ == "__main__":
    main()
