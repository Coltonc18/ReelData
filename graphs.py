# pyright: reportGeneralTypeIssues=false

# To import Vega-Altair into anaconda, run the commands below in the conda command line:
# conda install -c conda-forge altair vega_datasets altair_viewer vega
import altair as alt
import pandas as pd
from vega_datasets import data
from scipy.signal import savgol_filter

class Graphs:

    def __init__(self, *args):
        '''
        Initializes a GraphGenerator object with a dictionary of available graph functions and creates the specified graphs.

        Args:
            *args: A variable number of string arguments that correspond to the available graph function names.
        '''
        self._functions = {
            'budget_expertRating': Graphs.budget_expertRating,
            'budget_userRating': Graphs.budget_userRating,
            'releaseDate_totalRevenue': Graphs.releaseDate_totalRevenue,
            'userRating_expertRating': Graphs.userRating_expertRating,
            'companies_totalRevenue': Graphs.companies_totalRevenue,
            'genres_userRating': Graphs.genres_userRating,
            'genres_expertRating': Graphs.genres_expertRating,
            'genres_totalRevenue': Graphs.genres_totalRevenue,
            'genres_averageRevenue': Graphs.genres_averageRevenue,
            'actorType_averageRevenue': Graphs.actorType_averageRevenue,
            'actorType_averageRating': Graphs.actorType_averageRating,
            'actorType_expertRating': Graphs.actorType_expertRating,
            'releaseDate_avgRating': Graphs.releaseDate_avgRating,
            'budget_avgRevenue' : Graphs.budget_avgRevenue
            }
        self.create_graphs(*args)

    def create_graphs(self, *args, all=False):
        '''
        Creates the specified graphs based on the given arguments.

        Args:
            *args: A variable number of string arguments that correspond to the available graph function names.
            all (bool, optional): If True, creates all available graphs. Defaults to False.
            '''
        if all:
            for function in self._functions.keys():
                self._functions[function]()

        for arg in args:
            try:
                self._functions[arg]()
            except KeyError:
                print(f'Function {arg} does not exist')

    @staticmethod
    def budget_expertRating():
        '''
        Generates a scatter plot showing the relationship between movie budget and expert rating, using data from the
        'master_dataset.csv' file. Movies with budgets less than $10,000 or more that $300000000 and 
        budgets/ratings equal to 0 are excluded from the plot.
        The resulting chart is saved to 'graphs/budget_expertRating.html'.
        '''

        # Used to diable the default bahvior of Altiar that limited the max # of rows that can be displayed
        alt.data_transformers.disable_max_rows()

        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Filter out movies with budgets less than $10,000 and budgets over $300000000 to get rid of outliers
        data = data[(data['budget'] >= 10000) & (data['budget'] <= 300000000)]

        # Filter out movies with budgets and ratings that are 0
        data = data.query('budget > 0')
        data = data.query('RT_expert_rating > 0')

        # Define color scheme
        color_scheme = {
            'Certified Fresh': '#E0B713',  
            'Fresh': '#b30000',  
            'Rotten': '#444444'  
        }

        # Create the scatter plot 
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('budget', axis=alt.Axis(title='Budget')),
            y=alt.Y('RT_expert_rating', axis=alt.Axis(title='Expert Rating')),
            color=alt.Color('tomatometer_status:N', legend=alt.Legend(title='Tomato Status'), scale=alt.Scale(range=list(color_scheme.values()))),
            tooltip=['title','budget', 'RT_expert_rating']
        ).properties(
            title='Movie Budget vs Expert Ratings',
            width=800,
            height=400
        ).interactive()
        # Display the chart
        chart.save('graphs/budget_expertRating.html')


    @staticmethod
    def budget_userRating():
        '''
        Generates a scatter plot showing the relationship between movie budget and user rating, using data from the
        'master_dataset.csv' file. Movies with budgets less than $10,000 or more than $300000000 to get rid of outliers 
        and budgets/ratings equal to 0 are excluded from the plot.
        The resulting chart is saved to 'graphs/budget_userRating.html'.
        '''
        
        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Filter out movies with budgets less than $10,000 and budgets over $300000000 to get rid of outliers
        data = data[(data['budget'] >= 10000) & (data['budget'] <= 300000000)
        ]

        # Filter out budgets that are zero
        data = data.query('budget > 0')
        data = data.query('audience_rating > 0')
        data = data.dropna(subset=['audience_status'])

        color_scheme = {
            'Spilled': '#444444',
            'Upright': '#b30000'  
        }

        # Create the scatter plot
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('budget', axis=alt.Axis(title='Budget')),
            y=alt.Y('audience_rating', axis=alt.Axis(title='Audience Rating')),
            color=alt.Color('audience_status:N', legend=alt.Legend(title='Tomato Status'), scale=alt.Scale(range=list(color_scheme.values()))),
            tooltip=['title','budget', 'audience_rating']
        ).properties(
            title='Movie Budget vs Audience Ratings',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/budget_userRating.html')


    @staticmethod
    def releaseDate_totalRevenue():
        '''
        Generates a bar graph showing the total revenue of movies by release year, using data from the 'master_dataset.csv' file.
        Only movies released between 1930 and 2017 are included in the plot. The resulting chart is saved to 
        'graphs/releaseDate_totalRevenue.html'.
        '''
        alt.data_transformers.disable_max_rows()

        # Load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter data by year 1930 - 2017
        data = data.query("release_date >= '1930-01-01' and release_date < '2017-01-01'")

        # Create a bar chart showing relationships between revenue and year
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('year(release_date):T', axis=alt.Axis(title='Release Date'), scale=alt.Scale(domain=(1930, 2016))),
            y=alt.Y('sum(revenue)', axis=alt.Axis(title='Total Revenue')),
            color=alt.Color('sum(revenue)', scale=alt.Scale(scheme='goldred'), legend=None),
            tooltip=['year(release_date):T', 'sum(revenue)'],
        ).properties(
            title='Revenue by Year',
            width=800,
            height=400
        ).interactive()

        # display the chart
        chart.save('graphs/releaseDate_totalRevenue.html')


    @staticmethod
    def userRating_expertRating():
        '''
        Create a scatter plot showing the relationship between audience rating and expert rating for movies in the dataset.
        Load the dataset from a CSV file and filter out any movies with zero budgets or no expert rating data.
        Then create a scatter plot using the audience rating and expert rating data, with each point representing a movie.
        Save the chart as an HTML file.
        '''

        # load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter out budgets that are zero
        data = data.query('budget > 0')

        # filter out rows where RT_expert_rating is 0 audience ratings not n/a
        data = data[(data['RT_expert_rating'] != 0) & (data['audience_rating'].notna())]

        # create a scatter plot showing the relationship between audience rating and expert rating
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('user_rating', axis=alt.Axis(title='Audience Rating')),
            y=alt.Y('RT_expert_rating', axis=alt.Axis(title='Expert Rating')),
            tooltip=['title','user_rating', 'RT_expert_rating', 'budget']
        ).properties(
            width=800,
            height=400,
            title='Audience Rating vs Expert Rating'
        )

        # Display the chart
        chart.save('graphs/userRating_expertRating.html')


    @staticmethod
    def companies_totalRevenue():
        '''
        Create a bar chart showing the top production companies by total revenue.
        Load the dataset from a CSV file and group the data by production company.
        Extract the top 15 production companies based on total revenue and create a bar chart showing their revenue.
        Save the chart as an HTML file.
        '''

        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Extract the name of each production company from the dictionary and explode the column
        data['production_companies'] = data['production_companies'].str.split(", ")
        data = data.explode('production_companies')

        # Filter to get the top 15 production companies based on total revenue and put it into a list
        top_producers = data.groupby('production_companies')['revenue'].sum().sort_values(ascending=False).head(15).index.tolist()
        data = data[data['production_companies'].isin(top_producers)]

        # Define the color scale as a gradient with the desired number of colors
        num_colors = len(top_producers)
        color_scale = alt.Scale(scheme='goldred', domain=top_producers)


        # Create a chart for all selected production companies, sorting in decsending order
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('production_companies:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Producion Companies')),
            y=alt.Y('sum(revenue):Q', axis=alt.Axis(title='Total Revenue')),
            color=alt.Color('production_companies:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=color_scale, legend=None),
            tooltip=['production_companies:N', 'sum(revenue):Q']
        ).properties(
            title='Top 10 Production Companies vs Total Revenue',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/companies_totalRevenue.html')


    @staticmethod
    def genres_userRating():
        '''
        Loads movie data from a CSV file, filters it by year, calculates the mean user rating
        for each year, and creates a line graph showing the average user rating over the years.
        The resulting chart is saved as an HTML file.
        '''

        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Split the genres column and explode the column
        data['genres'] = data['genres'].str.split(', ')
        data = data.explode('genres')

        # Filter to get the genres and sort in decsending order
        top_genres = data.groupby('genres').size().sort_values(ascending=False).index
        data = data[data['genres'].isin(top_genres)]

        # Group the data by genre and calculate the average user rating for each genre
        genre_ratings = data.groupby('genres').agg({'user_rating': 'mean'}).reset_index()

        # Define the color scale as a gradient with the desired number of colors
        num_colors = len(top_genres)
        color_scale = alt.Scale(scheme='goldred', domain=list(range(num_colors)))

        # Sort the data by average user rating in descending order and reset the index
        genre_ratings = genre_ratings.sort_values('user_rating', ascending=False).reset_index(drop=True)

        # Assign a rank to each genre based on its index in the sorted data
        genre_ratings['rank'] = genre_ratings.index

        # Create a stacked bar chart showing the average user rating for each genre
        chart = alt.Chart(genre_ratings).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('user_rating:Q', axis=alt.Axis(title='Average Audience Rating')),
            color=alt.Color('rank:O', scale=color_scale, legend=None)
        ).properties(
            title='Genre vs Average Audience Rating',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/genres_userRating.html')


    @staticmethod
    def genres_expertRating():
        '''
        Loads movie data from a CSV file, filters it by year, calculates the mean expert rating
        for each year, and creates a line graph showing the average expert rating over the years.
        The resulting chart is saved as an HTML file.
        '''

        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Split the genres column and explode the column
        data['genres'] = data['genres'].str.split(', ')
        data = data.explode('genres')

        # Filter to get the genres and sort in decsending order
        top_genres = data.groupby('genres').size().sort_values(ascending=False).index
        data = data[data['genres'].isin(top_genres)]

        # Group the data by genre and calculate the average expert rating for each genre
        genre_ratings = data.groupby('genres').agg({'RT_expert_rating': 'mean'}).reset_index()

        # Define the color scale as a gradient with the desired number of colors
        num_colors = len(top_genres)
        color_scale = alt.Scale(scheme='goldred', domain=list(range(num_colors)))

        # Sort the data by average expert rating in descending order and reset the index
        genre_ratings = genre_ratings.sort_values('RT_expert_rating', ascending=False).reset_index(drop=True)

        # Assign a rank to each genre based on its index in the sorted data
        genre_ratings['rank'] = genre_ratings.index

        # Create a stacked bar chart showing the average expert rating for each genre
        chart = alt.Chart(genre_ratings).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('RT_expert_rating:Q', axis=alt.Axis(title='Average Expert Rating')),
            color=alt.Color('rank:O', scale=color_scale, legend=None)
        ).properties(
            title='Genre vs Expert Rating',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/genres_expertRating.html')


    @staticmethod
    def genres_totalRevenue():
        '''
        Loads movie data from a CSV file and creates a bar chart showing the total revenue for each genre.
        The resulting chart is saved as an HTML file.
        '''

        # Load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter out movies with zero revenue and missing genres
        data = data.query('revenue > 0')
        data = data.dropna(subset=['genres'])

        # Explode the genres column to make a row for each genre in a movie
        data = data.assign(genres=data['genres'].str.split(',')).explode('genres')

        # Remove duplicates from genres column
        data['genres'] = data['genres'].str.strip()
        data = data.drop_duplicates(subset=['genres', 'imdb_id'])

        # Calculate the total revenue for each genre
        genre_revenue_sum = data.groupby('genres')['revenue'].sum().reset_index()

        # Create the bar graph showing revenue by genre
        chart = alt.Chart(genre_revenue_sum).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('revenue:Q', axis=alt.Axis(title='Total Revenue')),
            color=alt.Color('genres:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=alt.Scale(scheme='goldred', reverse=False), legend=None),
            tooltip=['genres:N', 'revenue:Q']
        ).properties(
            title='Genre vs Total Revenue',
            width=800,
            height=400
        ).interactive()

        # display the chart
        chart.save('graphs/genres_totalRevenue.html')


    @staticmethod
    def genres_averageRevenue():
        '''
        Loads movie data from a CSV file, filters it by non-zero revenue and non-missing genres,
        calculates the average revenue for each genre, sorts the genres by revenue in descending order, 
        and creates a bar chart showing average revenue by genre. The chart is saved as an HTML file.
        '''

        # Used to diable the default bahvior of Altiar that limited the max # of rows that can be displayed
        alt.data_transformers.disable_max_rows()

        # Load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter out movies with zero revenue and missing genres
        data = data.query('revenue > 0')
        data = data.dropna(subset=['genres'])

        # Explode the genres column to make a row for each genre in a movie
        data = data.assign(genres=data['genres'].str.split(',')).explode('genres')

        # Remove duplicates from genres column
        data['genres'] = data['genres'].str.strip()
        data = data.drop_duplicates(subset=['genres', 'imdb_id'])

        # Calculate the average revenue for each genre
        genre_revenue = data.groupby('genres')['revenue'].mean().reset_index()

        # Sort the genres by revenue in descending order
        genre_revenue = genre_revenue.sort_values('revenue', ascending=False)

        # Create the bar chart showing average revenue by genre
        chart = alt.Chart(genre_revenue).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('revenue:Q', axis=alt.Axis(title='Average Revenue')),
            color=alt.Color('genres:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=alt.Scale(scheme='goldred', reverse=False), legend=None),
            tooltip=['genres:N', 'revenue:Q']
        ).properties(
            title='Genre vs Average Revenue',
            width=800,
            height=400
        ).interactive()

        # display the chart
        chart.save('graphs/genres_averageRevenue.html')


    @staticmethod
    def actorType_averageRevenue():
        '''
        Loads movie data from a CSV file and calculates the average revenue for movies with top actors 
        categorized as A List, Top 100, Top 1K, and No Top Actors. It creates a bar chart showing the 
        average revenue by actor type and saves the chart as an HTML file.
        '''

        # load data
        data = pd.read_csv('data/master_dataset.csv')

        # calculate average revenue for movies with top 100 actors
        alist_avg_revenue = data[data['a_list'] == 1]['revenue'].mean()

        # calculate average revenue for movies with top 100 actors
        top100_avg_revenue = data[data['top_100'] == 1]['revenue'].mean()

        # calculate average revenue for movies with top 1k actors
        top1k_avg_revenue = data[data['top_1k'] == 1]['revenue'].mean()

        # calculate average revenue for movies with no top actors
        no_top_avg_revenue = data[(data['a_list'] == 0) & (data['top_100'] == 0) & (data['top_1k'] == 0)]['revenue'].mean()

        # create a DataFrame to use for plotting
        plot_data = pd.DataFrame({
            'actor_type': pd.Categorical(['A List', 'Top 100', 'Top 1K', 'No Top Actors'], categories=['A List', 'Top 100', 'Top 1K', 'No Top Actors'], ordered=True),
            'average_revenue': [alist_avg_revenue, top100_avg_revenue, top1k_avg_revenue, no_top_avg_revenue],
        })

        # define the desired colors for each bar
        color_scheme = {
            'A List': '#b30000',
            'Top 100': '#E0B713',
            'Top 1K': '#13A3E0',
            'No Top Actors': '#444444'
        }

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type', title='Actor Type', axis=alt.Axis(labelAngle=0), sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_revenue', title='Average Revenue'),
            color=alt.Color('actor_type:N', legend=None, scale=alt.Scale(domain=list(color_scheme.keys()), range=list(color_scheme.values()))),
            tooltip=['actor_type:N', 'average_revenue:Q']
        ).properties(
            title='Type of Actor vs Average Revenue',
            width=400,
            height=400
        ).interactive()

        # display chart
        chart.save('graphs/actorType_averageRevenue.html')


    @staticmethod
    def actorType_averageRating():
        '''
        Loads movie data from a CSV file and calculates the average audience rating for movies with top 
        actors categorized as A List, Top 100, Top 1K, and No Top Actors. It creates a bar chart showing 
        the average audience rating by actor type and saves the chart as an HTML file.
        '''

        # load data
        data = pd.read_csv('data/master_dataset.csv')

        # calculate average audience rating for movies with A-list actors
        alist_avg_rating = data[data['a_list'] == 1]['audience_rating'].mean()

        # calculate average audience rating for movies with top 100 actors
        top100_avg_rating = data[data['top_100'] == 1]['audience_rating'].mean()

        # calculate average audience rating for movies with top 1k actors
        top1k_avg_rating = data[data['top_1k'] == 1]['audience_rating'].mean()

        # calculate average audience rating for movies without top actors
        no_top_avg_rating = data[(data['a_list'] == 0) & (data['top_100'] == 0) & (data['top_1k'] == 0)]['user_rating'].mean()

        # create a DataFrame to use for plotting
        plot_data = pd.DataFrame({
            'actor_type': pd.Categorical(['A List', 'Top 100', 'Top 1K', 'No Top Actors'], categories=['A List', 'Top 100', 'Top 1K', 'No Top Actors'], ordered=True),
            'average_rating': [alist_avg_rating, top100_avg_rating, top1k_avg_rating, no_top_avg_rating],
        })

        # define the desired colors for each bar
        color_scheme = {
            'A List': '#b30000',
            'Top 100': '#E0B713',
            'Top 1K': '#13A3E0',
            'No Top Actors': '#444444'
        }

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type', title='Actor Type', axis=alt.Axis(labelAngle=0),sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_rating', title='Average Audience Rating'),
            color=alt.Color('actor_type:N', legend=None, scale=alt.Scale(domain=list(color_scheme.keys()), range=list(color_scheme.values()))),
            tooltip=['actor_type:N', 'average_rating:Q']
        ).properties(
            title='Top Actors vs Audience Ratings',
            width=400,
            height=400
        ).interactive()

        # display chart
        chart.save('graphs/actorType_averageRating.html')
    

    @staticmethod
    def actorType_expertRating():
        '''
        Loads movie data from a CSV file, calculates the average expert rating for movies
        with A-list actors, top 100 actors, top 1k actors, and movies without top actors, and creates a
        bar chart using Altair to visualize the results. The chart is saved in an HTML file.
        '''
        # load data
        data = pd.read_csv('data/master_dataset.csv')

        # calculate average expert rating for movies with A-list actors
        alist_avg_rating = data[data['a_list'] == 1]['RT_expert_rating'].mean()

        # calculate average expert rating for movies with top 100 actors
        top100_avg_rating = data[data['top_100'] == 1]['RT_expert_rating'].mean()

        # calculate average expert rating for movies with top 1k actors
        top1k_avg_rating = data[data['top_1k'] == 1]['RT_expert_rating'].mean()

        # calculate average expert rating for movies without top actors
        no_top_avg_rating = data[(data['a_list'] == 0) & (data['top_100'] == 0) & (data['top_1k'] == 0)]['RT_expert_rating'].mean()

        # create a DataFrame to use for plotting
        plot_data = pd.DataFrame({
            'actor_type': pd.Categorical(['A List', 'Top 100', 'Top 1K', 'No Top Actors'], categories=['A List', 'Top 100', 'Top 1K', 'No Top Actors'], ordered=True),
            'average_rating': [alist_avg_rating, top100_avg_rating, top1k_avg_rating, no_top_avg_rating],
        })

        # defines the color for each bar
        color_scheme = {
            'A List': '#b30000',
            'Top 100': '#E0B713',
            'Top 1K': '#13A3E0',
            'No Top Actors': '#444444'
        }

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type', title='Actor Type', axis=alt.Axis(labelAngle=0), sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_rating:Q', title='Average Expert Rating'),
            color=alt.Color('actor_type:N', legend=None, scale=alt.Scale(domain=list(color_scheme.keys()), range=list(color_scheme.values()))),
            tooltip=['actor_type:N', 'average_rating:Q']
        ).properties(
            title='Top Actors vs Expert Ratings',
            width=400,
            height=400
        ).interactive()

        # display chart
        chart.save('graphs/actorType_expertRating.html')


    @staticmethod
    def releaseDate_avgRating():
        '''
        Loads movie data from a CSV file, filters it by year, calculates the mean expert rating
        for each year, and creates a line graph showing the average expert rating over the years.
        The resulting chart is saved as an HTML file.
        '''

        # Load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Convert release_date to datetime type
        data['release_date'] = pd.to_datetime(data['release_date'])

        # Filter data by year 1930 - 2017
        data = data.query("release_date >= '1930-01-01' and release_date < '2017-01-01'")

        # Calculate the mean expert rating for each year
        avg_rating = data.groupby(pd.Grouper(key='release_date', freq='Y'))['RT_expert_rating'].mean(numeric_only=True).reset_index()

        avg_rating['RT_expert_rating'] = savgol_filter(avg_rating['RT_expert_rating'], 50, 7)

        # Create a line graph showing average expert rating over the years
        chart = alt.Chart(avg_rating).mark_line(color='#b30000').encode(
            x=alt.X('year(release_date):T', axis=alt.Axis(title='Release Date'), scale=alt.Scale(domain=(1930,2016))),
            y=alt.Y('RT_expert_rating', axis=alt.Axis(title='Average Expert Rating')),
            tooltip=['year(release_date):T', 'RT_expert_rating'],
        )

        # set chart properties
        chart = chart.properties(
            width=800,
            height=400,
            title='Average Expert Rating per Year'
        )

        # display the chart
        chart.save('graphs/releaseDate_avgRating.html')


    @staticmethod
    def budget_avgRevenue():
        '''
        Generates a line plot showing the average revenue for each budget value in the movie dataset.
        Saves the chart as an HTML file.
        '''

        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Filter out budgets less than $10,000
        data = data[(data['budget'] >= 10000) & (data['budget'] <= 40000000)]

        # Filter out budgets and revenues that are zero
        data = data.query('budget > 0')
        data = data.query('revenue > 0')

        # Calculate the average revenue for each budget value
        avg_revenue = data.groupby('budget')['revenue'].mean().reset_index()
        avg_revenue['revenue'] = savgol_filter(avg_revenue['revenue'], 70, 10)

        # Create the line plot
        chart = alt.Chart(avg_revenue).mark_line(color='#E0B713', interpolate='bundle').encode(
            x=alt.X('budget', axis=alt.Axis(title='Budget')),
            y=alt.Y('revenue', axis=alt.Axis(title='Average Revenue')),
            tooltip=['budget', 'revenue']
        ).properties(
            title='Movie Budget vs Average Revenue',
            width=800,
            height=400
        ).interactive()

        # display the chart
        chart.save('graphs/budget_avgRevenue.html')
