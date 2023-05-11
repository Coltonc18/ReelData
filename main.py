import json
import os
import pickle
import re
import time

import numpy as np
import pandas as pd

from graphs import Graphs
from webscraping import scrape_top_tier_actors


def main():
    merge_data()

    # graph = Graphs()
    # graph.create_graphs('example')
    pass

def merge_data(verbose=False):
    # Read the 'credits.csv' file and select only the 'id' and 'cast' columns
    # Set low_memory=False to avoid warning about mixed dtypes in the 'cast' column
    credits_df = pd.read_csv('data/credits.csv', usecols=['id', 'cast'], low_memory=False)

    # Convert cast column to comma separated list
    credits_df['cast'] = credits_df['cast'].apply(json_to_columns, args=('name',))

    # Temporarily set the columns to zeros since school computers cannot scrape
    credits_df['a_list'] = np.zeros(len(credits_df), dtype=np.int8)
    credits_df['top_100'] = np.zeros(len(credits_df), dtype=np.int8)
    credits_df['top_1k'] = np.zeros(len(credits_df), dtype=np.int8)
    
    # Create new columns in credits_df for each catagory of actor: A-List, Top 100, and Top 1000
    # Before making each column, assure the file exists containing the set of actors, and if it does not, scrape the web for it
    # A-List actors
    if not os.path.exists('data/alist_actors.pickle'):
        print('File Not Found: Scraping A-List actors') if verbose else None
        scrape_top_tier_actors(pages=['alist'])
    with open('data/alist_actors.pickle', 'rb') as file:
        alist_set = pickle.load(file)
    credits_df['a_list'] = credits_df['cast'].apply(lambda actors : 1 if any(actor in alist_set for actor 
                                                                             in actors.split(', ')) else 0)
    # Top-100 actors
    if not os.path.exists('data/top_100_actors.pickle'):
        print('File Not Found: Scraping Top 100 actors') if verbose else None
        scrape_top_tier_actors(pages=['top_100'])
    with open('data/top_100_actors.pickle', 'rb') as file:
        top_100_set = pickle.load(file)
    credits_df['top_100'] = credits_df['cast'].apply(lambda actors : 1 if any(actor in top_100_set for actor 
                                                                              in actors.split(', ')) else 0)
    # Top-1000 actors
    if not os.path.exists('data/top_1k_actors.pickle'):
        print('File Not Found: Scraping Top 1000 actors') if verbose else None
        scrape_top_tier_actors(pages=['top_1k'])
    with open('data/top_1k_actors.pickle', 'rb') as file:
        top_1k_set = pickle.load(file)
    credits_df['top_1k'] = credits_df['cast'].apply(lambda actors : 1 if any(actor in top_1k_set for actor 
                                                                             in actors.split(', ')) else 0)
    # Read the 'links.csv' file and select only the 'movieId' and 'imdbId' columns
    links_df = pd.read_csv('data/links.csv', usecols=['movieId', 'imdbId'], low_memory=False)

    # Check if the 'rating_averages.csv' file exists
    if not os.path.exists('data/rating_averages.csv'):
        # If it doesn't exist, read the 'ratings.csv.gz' file and select only the 'movieId' and 'rating' columns
        rating_df = pd.read_csv('data/ratings.csv.gz', usecols=['movieId', 'rating'], compression='gzip', low_memory=False)
        # Calculate the average user rating of each movie as a Series
        rating_avgs = rating_df.groupby('movieId')['rating'].mean()
        # Multiply the mean rating by 20 to convert it to a 100-point scale and round to 3 decimal places
        rating_avgs = rating_avgs.apply(lambda n: round(n * 20, 3))
        print(f'Ratings average df has data for {len(rating_avgs)} movies') if verbose else None
        # Save the resulting dataframe to a new 'rating_averages.csv' file
        rating_avgs.to_csv('data/rating_averages.csv')
    else:
        # If the 'rating_averages.csv' file exists, read it instead
        rating_avgs = pd.read_csv('data/rating_averages.csv') 

    # Load movie metadata from csv file
    metadata_df = pd.read_csv('data/movies_metadata.csv', low_memory=False)

    # Remove rows that are not movies
    metadata_df = metadata_df[metadata_df['video'] == False]
    metadata_df = metadata_df[metadata_df['status'] == 'Released']

    # Cast 'id' column to int32 data type
    metadata_df['id'] = metadata_df['id'].astype('int32')

    # Remove unnecessary columns from metadata
    metadata_df.drop(['adult', 'belongs_to_collection', 'homepage', 'overview', 'poster_path', 'status', 
                    'spoken_languages', 'tagline', 'Unnamed: 0'], axis='columns', inplace=True)
    
    # Convert production_companies, production_countries, and genres columns to comma separated lists
    for column in ['production_companies', 'production_countries', 'genres']:
        metadata_df[column] = metadata_df[column].apply(json_to_columns, args=('name',))
    
    # RT Critic Ratings
    # review_score column created with the converted rating values
    critic_df = pd.read_csv('data/rotten_tomatoes_critic_reviews.csv')
    critic_df['review_score'] = critic_df.apply(_convert_ratings, axis='columns')

    # RT Audience Ratings and drop unused columns
    audience_df = pd.read_csv('data/rotten_tomatoes_movies.csv')
    audience_df = audience_df.drop(axis='columns', labels=['movie_info', 'critics_consensus', 'runtime', 'genres'])

    # All RT Data
    # Create a DataFrame containing the average review score for each movie
    # by grouping the critic reviews DataFrame by the 'rotten_tomatoes_link' column and computing the mean
    # Then reset the index so 'rotten_tomatoes_link' becomes a column again
    rotten_tomatoes_df = pd.DataFrame(data=critic_df.groupby('rotten_tomatoes_link')['review_score'].mean()).reset_index()

    # Create a new column 'review_type' in the rotten_tomatoes_df DataFrame based on the 'review_score' column
    # Use a lambda function to assign 'Fresh' if rating is greater than or equal to 60, otherwise 'Rotten'
    rotten_tomatoes_df['review_type'] = rotten_tomatoes_df['review_score'].apply(lambda rating: 'Fresh' if rating >= 60 else 'Rotten')

    # Merge the audience DataFrame into the rotten_tomatoes_df DataFrame on the 'rotten_tomatoes_link' column
    # This adds the audience ratings and other data (except for movie_info, critics_consensus, runtime, and genres) 
    # to the DataFrame containing the average review score for each movie
    rotten_tomatoes_df = rotten_tomatoes_df.merge(audience_df, on='rotten_tomatoes_link')

    # Merge credits and metadata dataframes on the 'id' column
    master_df = pd.merge(credits_df, metadata_df, on='id')

    # If verbose is true, print the length of the merged dataframe: should stay constant throughout upcoming merges
    print(f'After FIRST merge, length is {len(master_df)}') if verbose else None

    # Merge the new dataframe with the rating_avg dataframe on the 'id' and 'movieId' columns
    master_df = pd.merge(master_df, rating_avgs, left_on='id', right_on='movieId', how='left')

    # Rename the 'rating' column to 'user_rating'
    master_df.rename(columns={'rating': 'user_rating'}, inplace=True)

    # If verbose is true, print the length and columns of the merged dataframe
    print(f'After SECOND merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    # Merge the new dataframe with the links dataframe on the 'id' and 'movieId' columns
    master_df = pd.merge(master_df, links_df, left_on='id', right_on='movieId', how='left')

    # If verbose is true, print the length and columns of the merged dataframe
    print(f'After THIRD merge, length is {len(master_df)}, cols are {master_df.columns}') if verbose else None

    # Merge the new dataframe with the rotten_tomatoes_df dataframe on the 'title' and 'movie_title' columns
    master_df = pd.merge(master_df, rotten_tomatoes_df, left_on='title', right_on='movie_title', how='left')

    # If verbose is true, print the length and columns of the merged dataframe
    print(f'After FOURTH merge, length is {len(master_df)}\nColumns are: {master_df.columns}') if verbose else None
    
    # Reorder the columns and filter out a few
    master_df = master_df.loc[:, ['id', 'imdb_id', 'rotten_tomatoes_link', 'title', 'content_rating', 'budget', 
                                  'revenue', 'review_score', 'review_type', 'release_date', 'streaming_release_date', 
                                  'runtime', 'user_rating', 'genres', 'vote_average', 'vote_count', 'original_language', 
                                  'production_companies', 'production_countries', 'directors', 'authors', 
                                  'cast', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 
                                  'audience_status', 'audience_rating', 'audience_count', 'tomatometer_fresh_critics_count', 
                                  'tomatometer_rotten_critics_count', 'a_list', 'top_100', 'top_1k']]
    
    # Changed names of a few columns for better understanding
    master_df.rename({'review_score': 'calc_RT_rating', 'review_type': 'RT_expert_class', 'tomatometer_rating': 'RT_expert_rating'}, axis='columns', inplace=True)
    
    # Save the dataframe to a file
    master_df.to_csv("data/master_dataset.csv", index=False)

def _convert_ratings(row):
    '''
    Converts ratings from fraction or letter values into a percentage score out of 100 points. 
    If there is no value, but only a "Fresh" or "Rotten" label, the value is set to 80% and 40% respectively.

        Parameters:
                row (Series): Row from a DataFrame which will have its expert_rating value normalized

        Returns:
                value (float): New calculated rating for that row
    '''
    # Pull the rating value out of the column
    value = row['review_score']

    # Normalize some letter score values
    if isinstance(value, str):
        value = value.replace(' ', '')

    # Convert fractional ratings into a score / 100 rating
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

def json_to_columns(cell, key):
    '''
    Intended to be used in a ``df.apply(json_to_columns)`` application.
    Takes a cell from the ``DataFrame`` and converts the stringified JSON in that cell to a Comma-Separated list of values

        Parameters:
                cell (str): Stringified JSON which is to be converted
                key (str): The key value in the JSON dictionary that will return the indended value

        Returns:
                values (str): Comma-Separated list in the format "item1, item2, item3" of all ``values`` in the JSON with the indicated ``key``
    '''
    try:
        # Use regex look aheads and look behinds to avoid replacing apostrophes in the middle of
        # a word with a double quotation, which would mess up the json parsing
        cell = re.sub("(?<=[A-Za-z])'(?=[A-Za-z])", "<apostrophe>", cell)

        # Replace all single quotes with double quotes
        cell = cell.replace("'", '"')

        # Replace occurrences of None with "None" because it messes up the JSON parsing
        cell = cell.replace('None', '"None"')

        # Turn the apostrophes back into single quotes
        cell = cell.replace("<apostrophe>", "'")

        # Load the json string into a list of dictionaries
        json_obj = json.loads(cell)
        values = ''

        # Iterate through each list index and convert the json format into a comma separated list
        for item in json_obj:
            # Don't add any duplicates: it will mess up the indexing of the ML dataset
            if f'{item[key]}' not in values:
                values += f'{item[key]}, '

        # Remove any left over quotes, and slice the string down by two indeces to deal with the extra comma at the end
        values = values[:-2].replace('"', '')

        # Values are returned in the format "item1, item2, item3, ..."
        return values
    
    except json.decoder.JSONDecodeError:
        # In certain cases, there are apostrophes that cannot be caught with the regex expression above,
        # which causes the JSON parser to throw an error as the strings aren't closed properly. In that 
        # rare case, just return an empty string
        return ''

def test_times():
    '''
    DEPRECATED
    ----------

    Tests the time difference between opening a csv file vs a gzipped csv file of the same data
    '''
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
