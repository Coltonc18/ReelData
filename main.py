import pandas as pd
import os

def main():
    # DO NOT UNCOMMENT THE FOLLOWING LINE WITHOUT CAUTION
    # compress_csv()
    merge_data()
    pass

def merge_data():
    # Most datasets have id as the common denominator to merge on
    # ratings.csv has UserId and MovieId, MovieId is the same as id, but will need to be groupby'ed before merging to find avg rating for each movie
    # Metadata.csv contains the correct imdbId for webscraping as well as the usual id
    credits_df = pd.read_csv('data/credits.csv.gz', compression='gzip')
    links_df = pd.read_csv('data/links.csv.gz', compression='gzip')
    rating_df = pd.read_csv('data/ratings.csv.gz', compression='gzip')
    metadata_df = pd.read_csv('data/movies_metadata.csv.gz', compression='gzip', low_memory=False)

    metadata_df = metadata_df[metadata_df['id'].str.isnumeric()]
    
    metadata_df['id'] = metadata_df['id'].astype('int64')

    rating_avg_df = rating_df.groupby('movieId')['rating'].mean().reset_index()

    master_df = pd.merge(credits_df, metadata_df, on='id')

    master_df = pd.merge(master_df, rating_avg_df, left_on='id', right_on='movieId')

    master_df = pd.merge(master_df, links_df, on='movieId')

    # Drop unnecessary columns
    master_df.drop(['id', 'movieId'], axis=1, inplace=True)
    
    # Save merged dataframe to CSV
    master_df.to_csv("data/master_dataset.csv", index=False)

def compress_csv():
    for filename in os.listdir('data'):
        filename = os.path.join('data', filename)
        df = pd.read_csv(filename, low_memory=False)
        df.to_csv(f'{filename}')

if __name__ == "__main__":
    main()
