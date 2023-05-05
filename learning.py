import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor

sns.set()

def create_learning_dataset(remake=False):
    if not remake:
        if os.path.exists('data/learning_dataset.csv'):
            return pd.read_csv('data/learning_dataset.csv')

    # Read in and filter the master dataset to the desired columns for Machine Learning
    df = pd.read_csv('data/master_dataset.csv')
    df = df.loc[:, ['content_rating', 'budget', 'revenue', 'calc_RT_rating', 'release_date', 
                    'runtime', 'user_rating', 'genres', 'original_language', 'production_companies', 
                    'production_countries', 'directors', 'authors', 'cast', 'tomatometer_status', 'RT_expert_rating', 
                    'tomatometer_count', 'audience_status', 'audience_rating', 'audience_count', 'tomatometer_fresh_critics_count', 
                    'tomatometer_rotten_critics_count', 'a_list', 'top_100', 'top_1k']]
    

    # Convert tomatometer_fresh/rotten_critics_count to percentages
    total_counts = df['tomatometer_rotten_critics_count'] + df['tomatometer_fresh_critics_count']
    df['tomatometer_fresh_percentage'] = df['tomatometer_fresh_critics_count'] / total_counts * 100
    df['tomatometer_rotten_percentage'] = df['tomatometer_rotten_critics_count'] / total_counts * 100
    # Remove the old columns, as they will likely confuse the model
    df.drop(['tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count'], axis='columns', inplace=True)

    # Convert each actor, author, etc into their own column with manual One-Hot-Encoding
    encoded_df = pd.DataFrame(index=range(len(df)))
    # 'authors', 'directors', 'genres', 'production_companies', 'production_countries'
    for column in ['genres', 'directors', 'production_countries']:
        # Convert NaN values to empty strings to avoid type errors
        df[column] = df[column].fillna('')

        # Take each column (Series) and apply a lambda function to it, which returns a new Series of ones with the index as the name
        # Then replace any NaN values with 0 and cast the Series to an int0 (0 or 1 only) which should use the lowest memory
        # The list(set(x.split(', '))) portion converts the comma separated list (string) into a list, and removes any duplicates
        encoded_column = df[column].apply(lambda x: pd.Series([1] * len(list(set(x.split(', ')))), index=list(set(x.split(', '))))).fillna(0, downcast='infer').astype(np.int8) # type: ignore
        # Add a prefix to the name so that it is clear which column the name came from
        encoded_column = encoded_column.add_prefix(f'{column}_')
        # Append the encoded column to the DataFrame
        encoded_df = pd.concat([encoded_df, encoded_column], axis=1)
    
    # Convert the release date from a string to datetime type
    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d')
    # Dates will be stored as an ordinal (time since the year zero)
    df['release_date'] = df['release_date'].apply(lambda date: datetime.date.toordinal(date))

    # Drop the columns that have already been manually encoded
    df = df.drop(['authors', 'directors', 'cast', 'genres', 'production_companies', 'production_countries'], axis='columns')
    # Complete the One-Hot-Encoding on the rest of the classification columns and concatenate the manually encoded ones
    df = pd.concat([pd.get_dummies(df), encoded_df], axis=1)

    # Save to a csv file
    df.to_csv('data/learning.csv', index=False)
    # And return the modified dataframe
    return df


def regressive_model(label_column, error=0.2):
    learning_df = create_learning_dataset(remake=False)

    if 'revenue' in label_column:
        filtered_df = learning_df[learning_df['revenue'] != 0.0]
    elif 'expert' in label_column:
        filtered_df = learning_df[learning_df['RT_expert_rating'].notna()]
        filtered_df = learning_df[learning_df['RT_expert_rating'] != 0.0]
        label_column = 'RT_expert_rating'
        filtered_df = filtered_df.drop(['calc_RT_rating', 'tomatometer_status_Fresh', 'tomatometer_status_Rotten', 
                                        'tomatometer_status_Certified-Fresh', 'tomatometer_count'], axis='columns')
    elif 'audience' in label_column:
        filtered_df = learning_df[learning_df['audience_rating'].notna()]
        filtered_df = learning_df[learning_df['audience_rating'] != 0.0]
        label_column = 'audience_rating'
        filtered_df = filtered_df.drop(['audience_status_Spilled', 'audience_status_Upright', 'audience_count', 'user_rating'], axis='columns')
    else:
        return f'{label_column} is not a valid metric to train on, please pick revenue, expert, or audience'
    
    filtered_df = pd.get_dummies(filtered_df).dropna()

    features = filtered_df.drop([label_column], axis='columns')
    labels = filtered_df[label_column]
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

    accuracies = []
    for depth in range(8, 13, 1):

        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(features_train, labels_train)

        # Compute training accuracy
        train_predictions = model.predict(features_train)
        close = 0
        for prediction, actual in zip(train_predictions, labels_train):
            if abs(prediction / actual - 1) <= error:
                close += 1
        train_acc = close/len(train_predictions) * 100
        # print('Train Accuracy:', train_acc, '%')

        # Compute test accuracy
        test_predictions = model.predict(features_test)
        close = 0
        for prediction, actual in zip(test_predictions, labels_test):
            # print(f'Prediction: {prediction}, Actual: {actual}')
            if abs(float(prediction) / actual - 1) <= error:
                close += 1
        test_acc = close/len(test_predictions) * 100
        # print('Test  Accuracy:', test_acc, '%')

        accuracies.append({'max depth': depth, 'train accuracy': train_acc, 
                       'test accuracy': test_acc})
        
    accuracies = pd.DataFrame(accuracies)
    plot_accuracies(accuracies, 'train accuracy', 'Train', f'accuracy_graphs/{label_column}_RegressorTrain', save=False)
    plot_accuracies(accuracies, 'test accuracy', 'Test', f'accuracy_graphs/{label_column}_RegressorTest', save=False)

def plot_accuracies(accuracies, column, name, filepath, save=True):
    sns.relplot(kind='line', x='max depth', y=column, data=accuracies)
    plt.title(f'{name} Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel(f'{name} Accuracy')
    plt.ylim(-1, 101)

    if save:
        plt.savefig(filepath)
    plt.show()  # Display the graph
    

def neural_network():
    pass

if __name__ == '__main__':
    # print(create_learning_dataset().columns)
    # for col in ['revenue', 'user', 'expert']:
    regressive_model('expert', error=0.1)