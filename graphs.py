# pyright: reportGeneralTypeIssues=false

# To import Vega-Altair into anaconda, run the commands below in the conda command line:
# conda install -c conda-forge altair vega_datasets altair_viewer vega
import altair as alt
import pandas as pd
from vega_datasets import data

class Graphs:

    def __init__(self, *args):
        self._functions = {
            'budget_expertRating': Graphs.budget_expertRating,
            'budget_userRating': Graphs.budget_userRating,
            'releaseDate_totalRevenue': Graphs.releaseDate_totalRevenue,
            'userRating_expertRating': Graphs.userRating_expertRating,
            'companies_totalRevenue': Graphs.companies_totalRevenue,
            'companies_averageRevenue': Graphs.companies_averageRevenue,
            'genres_userRating': Graphs.genres_userRating,
            'genres_expertRating': Graphs.genres_expertRating,
            'genres_totalRevenue': Graphs.genres_totalRevenue,
            'genres_averageRevenue': Graphs.genres_averageRevenue,
            'actorType_averageRevenue': Graphs.actorType_averageRevenue,
            'actorType_averageRating': Graphs.actorType_averageRating,
            'actorType_expertRating': Graphs.actorType_expertRating,
            }
        self.create_graphs(*args)

    def create_graphs(self, *args, all=False):
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
        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Filter out movies with budgets less than $10,000
        data = data[data['budget'] >= 10000]

        # filter out movies with budgets and ratings that are 0
        data = data.query('budget > 0')
        data = data.query('RT_expert_rating > 0')

        # Create the scater plot 
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('budget', axis=alt.Axis(title='Budget')),
            y=alt.Y('RT_expert_rating', axis=alt.Axis(title='Expert Rating')),
            color=alt.Color('tomatometer_status', legend=alt.Legend(title='Tomato Status')),
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
        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Filter out budgets less than $10,000
        data = data[data['budget'] >= 10000]

        # Filter out budgets that are zero
        data = data.query('budget > 0')
        data = data.query('audience_rating > 0')
        data = data.dropna(subset=['audience_rating'])

        # Create the scatter plot
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('budget', axis=alt.Axis(title='Budget')),
            y=alt.Y('audience_rating', axis=alt.Axis(title='User Rating')),
            color=alt.Color('audience_status', legend=alt.Legend(title='Tomato Status')),
            tooltip=['title','budget', 'audience_rating']
        ).properties(
            title='Movie Budget vs User Ratings',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/budget_userRating.html')

    @staticmethod
    def releaseDate_totalRevenue():
        # Load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter data by year 1930 - 2017
        data = data.query("release_date >= '1930-01-01' and release_date < '2017-01-01'")

        # Create a bar graph showing relationships between revenue and year
        chart = alt.Chart(data).mark_bar(color='green').encode(
            x=alt.X('year(release_date):T', axis=alt.Axis(title='Release Date'), scale=alt.Scale(domain=(1930,2016))),
            y=alt.Y('sum(revenue)', axis=alt.Axis(title='Total Revenue')),
            tooltip=['year(release_date):T', 'sum(revenue)'],
        )

        # set chart properties
        chart = chart.properties(
            width=800,
            height=400,
            title='Revenue by Year'
        )

        # display the chart
        chart.save('graphs/releaseDate_totalRevenue.html')

    @staticmethod
    def userRating_expertRating():
        # load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter out budgets that are zero
        data = data.query('budget > 0')

        # filter out rows where RT_expert_rating is 0
        data = data[(data['RT_expert_rating'] != 0) & (data['audience_rating'].notna())]

        # create a scatter plot showing the relationship between user rating and expert rating
        chart = alt.Chart(data).mark_point().encode(
            x=alt.X('user_rating', axis=alt.Axis(title='User Rating')),
            y=alt.Y('RT_expert_rating', axis=alt.Axis(title='Expert Rating')),
            tooltip=['title','user_rating', 'RT_expert_rating', 'budget']
        ).properties(
            width=800,
            height=400,
            title='User Rating vs Expert Rating'
        )

        # Display the chart
        chart.save('graphs/userRating_expertRating.html')

    @staticmethod
    def companies_totalRevenue():
        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')

        # Extract the name of each production company from the dictionary and explode the column
        data['production_companies'] = data['production_companies'].str.split(", ")
        data = data.explode('production_companies')

        # Filter to get the top 30 production companies based on total revenue and put it into a list
        top_producers = data.groupby('production_companies')['revenue'].sum().sort_values(ascending=False).head(15).index.tolist()
        data = data[data['production_companies'].isin(top_producers)]

        # Define the color scale as a gradient with the desired number of colors
        num_colors = len(top_producers)
        color_scale = alt.Scale(scheme='yellowgreenblue', domain=top_producers)


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
    def companies_averageRevenue():
        # Load the data from the CSV file
        data = pd.read_csv('data/master_dataset.csv')
        top_comps = set(data.groupby('production_companies')['revenue'].sum().head(15).index.to_list())

        # Extract the name of each production company from the dictionary and explode the column
        data['production_companies'] = data['production_companies'].apply(lambda x: set(str(x).split(', ')))
        data = data.loc[[any(company in top_comps for company in row) for row in data['production_companies']]]
        data = data.explode('production_companies')

        # Filter to get the top 30 production companies based on average revenue
        top_producers = data.groupby('production_companies')['revenue'].mean().sort_values(ascending=False).index.tolist()

        data = data[data['production_companies'].isin(top_producers)]
        # Define the color scale as a gradient with the desired number of colors
        num_colors = len(top_producers)
        color_scale = alt.Scale(scheme='yellowgreenblue', domain=top_producers)



        # Create a chart for all selected production companies and sort in decsending order
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('production_companies:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Producion Companies')),
            y=alt.Y('mean(revenue):Q', axis=alt.Axis(title='Average Revenue')),
            color=alt.Color('production_companies:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=color_scale, legend=None),
            tooltip=['production_companies:N', 'mean(revenue):Q']
        ).properties(
            title='Top 10 Production Companies vs Average Revenue',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/companies_averageRevenue.html')

    @staticmethod
    def genres_userRating():
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
        color_scale = alt.Scale(scheme='yellowgreenblue', domain=list(range(num_colors)))

        # Sort the data by average user rating in descending order and reset the index
        genre_ratings = genre_ratings.sort_values('user_rating', ascending=False).reset_index(drop=True)

        # Assign a rank to each genre based on its index in the sorted data
        genre_ratings['rank'] = genre_ratings.index

        # Create a stacked bar chart showing the average user rating for each genre
        chart = alt.Chart(genre_ratings).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('user_rating:Q', axis=alt.Axis(title='Average User Rating')),
            color=alt.Color('rank:O', scale=color_scale, legend=None)
        ).properties(
            title='Genre vs Average User Rating',
            width=800,
            height=400
        ).interactive()

        # Display the chart
        chart.save('graphs/genres_userRating.html')

    @staticmethod
    def genres_expertRating():
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
        color_scale = alt.Scale(scheme='yellowgreenblue', domain=list(range(num_colors)))

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
        # load the data
        data = pd.read_csv("data/master_dataset.csv")

        # Filter out movies with zero revenue and missing genres
        data = data.query('revenue > 0')
        data = data.dropna(subset=['genres'])

        # Explode the genres column to make a row for each genre in a movie
        data = data.assign(genres=data['genres'].str.split(',')).explode('genres')

        # Remove duplicates from genres column
        data['genres'] = data['genres'].str.strip()
        data = data.drop_duplicates(subset=['genres'])

        # Calculate the total revenue for each genre
        genre_revenue = data.groupby('genres')['revenue'].sum().reset_index()

        # Create the bar graph showing revenue by genre
        chart = alt.Chart(genre_revenue).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('revenue:Q', axis=alt.Axis(title='Revenue')),
            color=alt.Color('genres:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=alt.Scale(scheme='yellowgreenblue', reverse=False), legend=None),
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
        # load the data
        data = pd.read_csv("data/master_dataset.csv")

        # filter out movies with zero revenue and missing genres
        data = data.query('revenue > 0')
        data = data.dropna(subset=['genres'])

        # explode the genres column to make a row for each genre in a movie
        data = data.assign(genres=data['genres'].str.split(',')).explode('genres')

        # Remove duplicates from genres column
        data['genres'] = data['genres'].str.strip()
        data = data.drop_duplicates(subset=['genres'])

        # calculate the average revenue for each genre
        genre_revenue = data.groupby('genres')['revenue'].mean().reset_index()

        # sort the genres by revenue in descending order
        genre_revenue = genre_revenue.sort_values('revenue', ascending=False)

        # create the bar chart showing average revenue by genre
        chart = alt.Chart(genre_revenue).mark_bar().encode(
            x=alt.X('genres:N', sort='-y', axis=alt.Axis(labelAngle=45, title='Genres')),
            y=alt.Y('revenue:Q', axis=alt.Axis(title='Revenue')),
            color=alt.Color('genres:N', sort=alt.EncodingSortField('revenue', order='descending'),
                            scale=alt.Scale(scheme='yellowgreenblue', reverse=False), legend=None),
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
            'color': ['red', 'green', 'blue', 'gray']
        })

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type', title='Actor Type', axis=alt.Axis(labelAngle=0), sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_revenue', title='Average Revenue'),
            color=alt.Color('color', legend=None),
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
        # load data
        data = pd.read_csv('data/master_dataset.csv')

        # calculate average user rating for movies with A-list actors
        alist_avg_rating = data[data['a_list'] == 1]['audience_rating'].mean()

        # calculate average user rating for movies with top 100 actors
        top100_avg_rating = data[data['top_100'] == 1]['audience_rating'].mean()

        # calculate average user rating for movies with top 1k actors
        top1k_avg_rating = data[data['top_1k'] == 1]['audience_rating'].mean()

        # calculate average user rating for movies without top actors
        no_top_avg_rating = data[(data['a_list'] == 0) & (data['top_100'] == 0) & (data['top_1k'] == 0)]['user_rating'].mean()

        # create a DataFrame to use for plotting
        plot_data = pd.DataFrame({
            'actor_type': ['A List', 'Top 100', 'Top 1K', 'No Top Actors'],
            'average_rating': [alist_avg_rating, top100_avg_rating, top1k_avg_rating, no_top_avg_rating]
        })

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type', title='Actor Type', axis=alt.Axis(labelAngle=0),sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_rating', title='Average User Rating'),
            color=alt.Color('actor_type', legend=None, scale=alt.Scale(domain=['A List', 'Top 100', 'Top 1K', 'No Top Actors'], range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])),
            tooltip=['actor_type:N', 'average_rating:Q']
        ).properties(
            title='Top Actors vs User Ratings',
            width=400,
            height=400
        ).interactive()


        # display chart
        chart.save('graphs/actorType_averageRating.html')
    
    @staticmethod
    def actorType_expertRating():
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
            'actor_type': ['A List', 'Top 100', 'Top 1K', 'No Top Actors'],
            'average_rating': [alist_avg_rating, top100_avg_rating, top1k_avg_rating, no_top_avg_rating]
        })

        # create bar chart
        chart = alt.Chart(plot_data).mark_bar().encode(
            x=alt.X('actor_type:N', title='Actor Type', axis=alt.Axis(labelAngle=0), sort=['A List', 'Top 100', 'Top 1K', 'No Top Actors']),
            y=alt.Y('average_rating:Q', title='Average Expert Rating'),
            color=alt.Color('actor_type:N', legend=None),
            tooltip=['actor_type:N', 'average_rating:Q']
        ).properties(
            title='Top Actors vs Expert Ratings',
            width=400,
            height=400
        ).interactive()

        # display chart
        chart.save('graphs/actorType_expertRating.html')
