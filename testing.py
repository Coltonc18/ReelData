import os
import pickle

import pandas as pd

from cse163_utils import assert_equals
from learning import get_learning_dataset
import main
from webscraping import scrape_top_tier_actors


def run_all_tests():
    # create_test_datasets()
    # test_merge()
    # test_webscraping()
    # test_master()
    # test_learning()
    test_graphs()


def create_test_datasets():
    print('Compiling test datasets...')
    titles = ['Toy Story', 'Jumanji', 'Toy Story 2', 'Monsters, Inc.', 'Cars', 'Avatar', 'Toy Story 3', 'Cars 2', 'Cars 3']

    metadata = pd.read_csv('data/movies_metadata.csv')
    metadata = metadata[metadata['title'].isin(titles)]
    metadata.to_csv('data/tests/movies_metadata.csv')

    movies = metadata[metadata['title'].isin(titles)].set_index('id')['title'].to_dict()

    credits = pd.read_csv('data/credits.csv', usecols=['cast', 'crew', 'id'])
    credits = credits[credits['id'].isin(movies.keys())]
    credits.to_csv('data/tests/credits.csv')

    RT_movies = pd.read_csv('data/rotten_tomatoes_movies.csv')
    RT_movies = RT_movies[RT_movies['movie_title'].isin(movies.values())]
    RT_movies.to_csv('data/tests/RT_movies.csv')

    critics = pd.read_csv('data/rotten_tomatoes_critic_reviews.csv')
    critics = critics[critics['rotten_tomatoes_link'].isin(RT_movies['rotten_tomatoes_link'])]
    critics.to_csv('data/tests/RT_critics.csv')


def test_learning():
    features, labels = get_learning_dataset('expert', remake_data=True, prefix='data/tests/')
    learning = pd.read_csv('data/tests/learning.csv')

    # Check the dimensions of the DataFrame
    assert_equals(9, len(learning))
    assert_equals(31, len(learning.columns))

    # Check the properties of the returned DF and Series (features and labels)
    assert_equals(len(features), len(labels))
    assert_equals(False, ('RT_expert_rating' in features.columns.to_list()))
    assert_equals(True, 'RT_expert_rating'.__eq__(labels.name))

    # Check a few locations in the DF against hand-computed values
    assert_equals(100, learning.loc[0, 'RT_expert_rating'])
    assert_equals(1, learning.loc[2, 'audience_status_Upright'])
    assert_equals(1, learning.loc[7, 'genres_Animation'])

    # Test that all columns are of int or float type
    # This means the One-Hot Encoding was completed successfully
    for column in learning.columns.to_list():
        assert_equals(True, (str(type(learning[column].to_list()[0])) in ["<class 'int'>", "<class 'float'>"]))

    print('Passed Machine Learning Testing')


def test_webscraping():
    # Read the scraped actors and cross reference them against what is seen on the site

    if not os.path.exists('data/tests/alist_actors.pickle'):
        scrape_top_tier_actors(pages=['alist'], prefix='data/tests/')
    with open('data/tests/alist_actors.pickle', 'rb') as file:
        alist_set = pickle.load(file)

    for actor in {'Ryan Reynolds', 'Tom Cruise', 'Leonardo DiCaprio', 'Denzel Washington', 'Tom Hanks', 'Bradley Cooper', 
                  'Christian Bale', 'Hugh Jackman', 'Dwayne Johnson', 'Robert Downey Jr.', 'Brad Pitt', 'Mark Wahlberg', 'Ryan Gosling'}:
        assert_equals(True, actor in alist_set)

    print('Passed Webscraping Testing')


def test_merge():
    # Merge the small datasets
    main.merge_data(verbose=False, prefix='data/tests/')
    master = pd.read_csv('data/tests/master_dataset.csv')

    # Check the dimensions of the DataFrame
    assert_equals(9, len(master))
    assert_equals(33, len(master.columns))

    # Check for columns from each of the files that were merged
    assert_equals(True, ('id' in master.columns.to_list()))
    assert_equals(True, ('rotten_tomatoes_link' in master.columns.to_list()))
    assert_equals(True, ('revenue' in master.columns.to_list()))
    assert_equals(True, ('user_rating' in master.columns.to_list()))
    assert_equals(True, ('audience_rating' in master.columns.to_list()))

    # Titles and ids calculated from create_test_datasets
    assert_equals(['Toy Story', 'Jumanji', 'Toy Story 2', 'Monsters, Inc.', 'Cars', 'Avatar', 'Toy Story 3', 'Cars 2', 'Cars 3'], master.loc[:, 'title'].to_list())
    assert_equals([862, 8844, 863, 585, 920, 19995, 10193, 49013, 260514], master.loc[:, 'id'].to_list())

    print('Passed Merge Testing')


def test_master():
    titles = ['Toy Story', 'Jumanji', 'Toy Story 2', 'Monsters, Inc.', 'Cars', 'Avatar', 'Toy Story 3', 'Cars 2', 'Cars 3']
    master = pd.read_csv('data/master_dataset.csv')
    master = master[master['title'].isin(titles)]
    master.to_csv('data/tests/master_dataset.csv')


def test_graphs():
    # Test Method companies_totalRevenue in Graphs.py
    # Check if method filtered out the different companies correctly
    companies = pd.read_csv('data/tests/master_dataset.csv')
    # Extract the name of each production company from the dictionary and explode the column
    companies['production_companies'] = companies['production_companies'].str.split(", ")
    companies = companies.explode('production_companies')
    # Filter to get the top 30 production companies based on total revenue and put it into a list
    top_producers = companies.groupby('production_companies')['revenue'].sum().sort_values(ascending=False).head(15).index.tolist()
    companies = companies[companies['production_companies'].isin(top_producers)]
    assert_equals(9, len(companies['production_companies'].unique()))
    # Test if sum of revenue for company is correct
    assert_equals(3872712463.0, companies[companies['production_companies'] == 'Pixar Animation Studios']['revenue'].sum())

    # Test Method genres_totalRevenue in Graphs.py
    genres = pd.read_csv('data/tests/master_dataset.csv')
    # Filter out movies with zero revenue and missing genres
    genres = genres.query('revenue > 0')
    genres = genres.dropna(subset=['genres'])
    # Explode the genres column to make a row for each genre in a movie
    genres = genres.assign(genres=genres['genres'].str.split(',')).explode('genres')
    # Remove duplicates from genres column
    genres['genres'] = genres['genres'].str.strip()
    genres = genres.drop_duplicates(subset=['genres', 'imdb_id'])
    # Calculate the total revenue for each genre
    genre_revenue_sum = genres.groupby('genres')['revenue'].sum()
    # Test if sum of revenue for company is correct
    assert_equals(2787965087.0, genre_revenue_sum['Action'])

    print('Passed Graphs Testing')


if __name__ == '__main__':
    run_all_tests()
