import os
import time

import numpy as np
import pandas as pd

from graphs import Graphs


def main():
    merge_data()
    # graph = Graphs()
    # graph.create_graphs('example')
    pass

def merge_data(verbose=False):
    # Will have to merge critic_reviews into tomatoes_movies on rotten_tomatoes_link, then into master on title
    
    # TODO: #1 Nathan please comment this so it looks a bit more readable instead of just dense code
    credits_df = pd.read_csv('data/credits.csv', usecols=['id', 'cast'], low_memory=False)
    links_df = pd.read_csv('data/links.csv', usecols=['movieId', 'imdbId'], low_memory=False)

    # Think we are done using this:
    # expert_df = pd.read_csv('data/expert_ratings.csv')

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
    
    # RT Critic Ratings
    critic_df = pd.read_csv('data/rotten_tomatoes_critic_reviews.csv')
    critic_df['review_score'] = critic_df.apply(_convert_ratings, axis='columns')

    # RT Audience Ratings and Movie Data
    audience_df = pd.read_csv('data/rotten_tomatoes_movies.csv')
    audience_df = audience_df.drop(axis='columns', labels=['movie_info', 'critics_consensus', 'runtime', 'genres'])

    # All RT Data
    rotten_tomatoes_df = pd.DataFrame(data=critic_df.groupby('rotten_tomatoes_link')['review_score'].mean()).reset_index()
    rotten_tomatoes_df['review_type'] = rotten_tomatoes_df['review_score'].apply(lambda rating: 'Fresh' if rating >= 60 else 'Rotten')
    rotten_tomatoes_df = rotten_tomatoes_df.merge(audience_df, on='rotten_tomatoes_link')

    master_df = pd.merge(credits_df, metadata_df, on='id')
    print(f'After FIRST  merge, length is {len(master_df)}') if verbose else None
    
    master_df = pd.merge(master_df, rating_avg_df, left_on='id', right_on='movieId', how='left')
    master_df.rename(columns={'rating': 'user_rating'}, inplace=True)
    print(f'After SECOND merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    master_df = pd.merge(master_df, links_df, left_on='id', right_on='movieId', how='left')
    print(f'After THIRD  merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    master_df = pd.merge(master_df, rotten_tomatoes_df, left_on='title', right_on='movie_title', how='left')
    print(f'After FOURTH merge, length is {len(master_df)}\nColumns are: {master_df.columns}') if verbose else None
    
    # Reorder the columns and filter out a few
    master_df = master_df.loc[:, ['id', 'imdb_id', 'rotten_tomatoes_link', 'title', 'budget', 'revenue', 
                                  'review_score', 'review_type', 'release_date', 'streaming_release_date', 
                                  'adult', 'runtime', 'user_rating', 'vote_average', 'vote_count', 
                                  'original_language', 'popularity', 'production_companies', 
                                  'production_countries', 'directors', 'authors', 'actors', 'cast', 
                                  'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 
                                  'audience_status', 'audience_rating', 'audience_count', 
                                  'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 
                                  'tomatometer_rotten_critics_count']]
    
    # Save the dataframe to a file
    master_df.to_csv("data/master_dataset.csv", index=False)

'''
Takes a row from the rotten_tomatoes_critic_reviews.csv dataset and converts the expert ratings from
fraction or letter value into a score out of 100 points. If there is no value, but only a "Fresh" or
"Rotten" label, the value is set to 80/100 and 40/100, respectively
'''
def _convert_ratings(row):
    # Pull the rating value out of the column
    value = row['review_score']

    # Normalize some letter score values
    if isinstance(value, str):
        value = value.replace(' ', '')

    # Convert fractional ratings into a score/100 rating
    if '/' in str(value):
        fraction = value.split('/')
        try:
            return (float(fraction[0]) / float(fraction[1])) * 100
        except ZeroDivisionError:
            return 80 if row['review_type'] == 'Fresh' else 40
    
    # Convert letter grades into scores based on a normalized scoring pattern
    letter_values = {'A': 100, 'A-': 93, 'B+': 88, 'B': 84, 'B-': 80, 'C+': 78,
                     'C': 74, 'C-': 70, 'D+': 68, 'D': 64, 'D-': 60, 'F': 50}
    if value in letter_values.keys():
        return letter_values[value]
    
    # If the value is NaN, return 80 or 40 based on review_type (Fresh or Rotten)
    if isinstance(value, float) and np.isnan(value):
        return 80 if row['review_type'] == 'Fresh' else 40
    
    # If nothing else, the value is already set, and will be returned as is
    return float(value)

'''
DEPRECATED: Method to test the time difference between opening a csv file vs a gzipped csv file
'''
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
