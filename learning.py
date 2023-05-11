import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# Initialize the seaborn library for graphing
sns.set()

def get_learning_dataset(label_column, remake_data=False):
    '''
    Gets or creates the Machine Learning specific dataset, then returns as a tuple of (features, labels)
    
        Parameters:
                label_column (str): The column to be used as labels in the ML Model (Options are ``"expert"``, ``"user/audience"``, or ``"revenue"``)
                remake_data (bool): Default False, defines whether or not the method should recreate and overwrite the existing dataset
        Returns:
                (features (DataFrame), labels (Series)) (tuple): Dataset slices that represent the features and labels to be used by a ML model
    '''

    # Remake the dataset if remake_data is True
    if remake_data:
        # Read in and filter the master dataset to the desired columns for Machine Learning
        df = pd.read_csv('data/master_dataset.csv')
        df = df.loc[:, ['content_rating', 'budget', 'revenue', 'calc_RT_rating', 'release_date', 
                        'runtime', 'user_rating', 'genres', 'original_language', 'production_companies', 
                        'production_countries', 'directors', 'authors', 'cast', 'tomatometer_status', 
                        'RT_expert_rating', 'tomatometer_count', 'audience_status', 'audience_rating', 
                        'audience_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count', 
                        'a_list', 'top_100', 'top_1k']]
        
        
        # Convert tomatometer_fresh/rotten_critics_count to percentages
        total_counts = df['tomatometer_rotten_critics_count'] + df['tomatometer_fresh_critics_count']
        df['tomatometer_fresh_percentage'] = df['tomatometer_fresh_critics_count'] / total_counts * 100
        df['tomatometer_rotten_percentage'] = df['tomatometer_rotten_critics_count'] / total_counts * 100
        # Remove the old columns, as they will likely confuse the model
        df.drop(['tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count'], axis='columns', inplace=True)

        # Convert each actor, author, etc into their own column with manual One-Hot-Encoding
        encoded_df = pd.DataFrame(index=range(len(df)))
        # To add more columns to be encoded, add to this list
        for column in ['genres']:
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

        # Save new dataset to a csv file
        df.to_csv('data/learning.csv', index=False)

    else:
        # If we don't need to recreate the data, just read it in from the .csv file
        df = pd.read_csv('data/learning.csv', low_memory=False)

    # Filter down the columns of the DataFrame based on which label_column was chosen
    if 'revenue' in label_column:
        filtered_df = df[df['revenue'] != 0.0]
    elif 'expert' in label_column:
        filtered_df = df[df['RT_expert_rating'].notna()]
        filtered_df = df[df['RT_expert_rating'] != 0.0]
        label_column = 'RT_expert_rating'
        filtered_df = filtered_df.drop(['calc_RT_rating', 'tomatometer_fresh_percentage', 
                                        'tomatometer_rotten_percentage'], axis='columns')
    elif ('audience' in label_column) or ('user' in label_column):
        filtered_df = df[df['audience_rating'].notna()]
        filtered_df = df[df['audience_rating'] != 0.0]
        label_column = 'audience_rating'
        filtered_df = filtered_df.drop(['user_rating'], axis='columns')
    else:
        raise NameError(f'{label_column} is not a valid metric to train on, please pick revenue, expert, or audience')
    
    # Compute One-Hot Encoding on the dataframe, and drop NaN values to prepare for ML
    filtered_df = pd.get_dummies(filtered_df).dropna()

    # Separate the DataFrame into separate features and labels
    features = filtered_df.drop([label_column], axis='columns')
    labels = filtered_df[label_column]

    # Return the datasets as a tuple
    return features, labels

def regressive_model(label_column, error=10, remake_data=False):
    '''
    Uses a Sci-Kit Learn ``DecisionTreeRegressor`` in an attempt to predict the ratings and revenue of unseen movies based on their other factors

        Parameters:
                label_column (str): The column to be used as labels in the ML Model (Options are ``"expert"``, ``"user/audience"``, or ``"revenue"``)
                                    Will be passed to ``get_learning_dataset()``.
                error (int): Percentage between ``0`` and ``100`` (default ``10``) is used to set the maximum error bound to be considered a *correct guess* 
                             by the ML model when making predictions.
                remake_data (bool): Default False, defines whether or not the method should recreate and overwrite the existing dataset
        
        Returns:
                None
    '''

    # Get the features and labels datasets from get_learning_dataset
    features, labels = get_learning_dataset(label_column, remake_data)
    # Split the data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

    # Loop through a range of numbers to determine the best accuracy based on the max_depth parameter of the model
    accuracies = []
    for depth in range(1, 20, 1):

        # Instantiate the DecisionTreeRegressor Model with variable depth
        model = DecisionTreeRegressor(max_depth=depth)
        # Fit the model to the training data
        model.fit(features_train, labels_train)

        # Compute training accuracy
        train_predictions = model.predict(features_train)
        close = 0
        for prediction, actual in zip(train_predictions, labels_train):
            if label_column in ['audience_rating', 'RT_expert_rating']:
                # If model is working with ratings (user or expert), the error is the difference between the actual and predicted ratings
                if abs(prediction - actual) <= error:
                    close += 1
            else:
                # Otherwise the model is working with revenue, and the error is the percentage difference between actual and predicted
                # Calculated by dividing the two values to see their difference
                if abs(prediction / actual - 1) * 100 <= error:
                    close += 1
        train_acc = close/len(train_predictions) * 100
        # print('Train Accuracy:', train_acc, '%')

        # Compute test accuracy using the same methods as above for training
        test_predictions = model.predict(features_test)
        close = 0
        for prediction, actual in zip(test_predictions, labels_test):
            if label_column in ['audience_rating', 'RT_expert_rating']:
                if abs(prediction - actual) <= error:
                    close += 1
            else:
                if abs(prediction / actual - 1) * 100 <= error:
                    close += 1
        test_acc = close/len(test_predictions) * 100
        # print('Test  Accuracy:', test_acc, '%')

        # Add to the accuracies list a new row of dictionaries with the accuracies
        accuracies.append({'max depth': depth, 'train accuracy': train_acc, 
                       'test accuracy': test_acc})
        
    # Convert accuracies into a pandas DataFrame
    accuracies = pd.DataFrame(accuracies)

    # Plot the accuracy of the model based on the max_depth parameter to visualize the best-fit model
    plot_accuracies(accuracies, 'train accuracy', 'Train', f'accuracy_graphs/{label_column}_RegressorTrain')
    plot_accuracies(accuracies, 'test accuracy', 'Test', f'accuracy_graphs/{label_column}_RegressorTest')

def neural_network(label_column, remake_data=False):
    '''
    Uses a Sci-Kit Learn ``MultiLayer Perceptron`` in an attempt to predict the ratings and revenue of unseen movies based on their other factors

        Parameters:
                label_column (str): The column to be used as labels in the ML Model (Options are ``"expert"``, ``"user/audience"``, or ``"revenue"``)
                                    Will be passed to ``get_learning_dataset()``.
                remake_data (bool): Default False, defines whether or not the method should recreate and overwrite the existing dataset
        
        Returns:
                testing_accuracy, training_accuracy (float, float): Calculates and returns the accuracy of the model on both the training and testing data
    '''

    # Get the features and labels from get_learning_dataset
    features, labels = get_learning_dataset(label_column, remake_data)

    # Identify the target column
    target_column = [label_column]
    # Scales all data down to be between 0 and 1
    features = features/features.max()

    # Print information about the DataFrame
    # print(df.describe().transpose())

    # Convert features into a numpy array
    features = features.dropna(axis=1).values
    # Convert and reshape the labels into a numpy array
    labels = labels.values
    labels = labels.reshape((len(labels),)) # type: ignore

    # TODO: REMOVE RANDOM STATE ONCE WORKING
    # Separate the data into training and testing datasets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=1)

    # Create a grid of hyper-parameters to test with the MLPRegressor
    param_grid = {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (10,),(20,),(25,),(35,),(50,),(100,),(10,10,10),(10,10),(50,50),(25,25)
            ],
        'learning_rate_init' : [0.01, 0.0075, 0.005, 0.0025, 0.001]
    }
    # Create a GridSearchCV object, which will test the given model with each hyperparameter
    # grid = GridSearchCV(MLPRegressor(), param_grid, refit=True, verbose=3)
    # grid.fit(features_train, labels_train)
    # Output the best parameters as found by the GridSearchCV
    # print(f'Best parameters found on development set: {grid.best_params_}')

    # Create and fit the Neural Network (Multi-Layer Perceptron) model
    mlp = MLPRegressor(hidden_layer_sizes=(25,), activation='tanh', solver='sgd', learning_rate_init=0.005, max_iter=500)
    mlp.fit(features_train, labels_train)

    # Predict values for both training and testing features set
    predict_train = mlp.predict(features_train)
    predict_test = mlp.predict(features_test)

    # Compare actual label values with the predicted values
    # print(f'Training difference: {abs(predict_train - labels_train).astype(int)}  Training: {round(abs(predict_train - labels_train).mean(), 2)}')
    # print(f'Testing  difference: {abs(predict_test  -  labels_test).astype(int)}   Testing: {round(abs(predict_test  -  labels_test).mean(), 2)}')

    # Return the average difference of the predictions vs labels for both testing and training data
    return abs(predict_test - labels_test).mean(), abs(predict_train - labels_train).mean() # type: ignore

def plot_accuracies(accuracies, column, name, filepath, save=True):
    '''
    Plots a linechart showing the accuracy of the DecisionTreeRegressor Model for different max_depth values.
    Saves to file if requested.

        Parameters:
                accuracies (DataFrame): Contains the columns "max depth" and ``column`` as well as the accuracy of the model for each depth
                column (str): Column name in ``accuracies`` that correlates to a percentage (usually test accuracy or train accuracy)
                name (str): String to be put in the title of the graph such as "Expert Ratings" or "Revenue"
                filepath (str): Filepath of image to be saved of the graph
                save (bool): Default True, whether to save the image or not

        Returns:
                None
    '''
    # Make a line relplot of the max_depth of the model vs the accuracy
    sns.relplot(kind='line', x='max depth', y=column, data=accuracies)
    # Add title and axis labels
    plt.title(f'{name} Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel(f'{name} Accuracy')
    # Resize the graph to fit all possible values
    plt.ylim(-1, 101)

    # Save the graph
    if save:
        plt.savefig(filepath)
    # Display the graph
    plt.show()

if __name__ == '__main__':
    # neural_network(label_column='expert', remake_data=False)

    # for col in ['revenue', 'audience', 'expert']:
    #     regressive_model(col, error=10)

    tests = []
    trains = []
    col = 'expert_rating'
    runs = 5
    for i in range(runs):
        test, train = neural_network(label_column=col, remake_data=False)
        tests.append(test)
        trains.append(train)
    print(f'Average  testing difference over {runs} runs for column {col} = {round(np.mean(tests), 2)}')
    print(f'Average training difference over {runs} runs for column {col} = {round(np.mean(trains), 2)}')
