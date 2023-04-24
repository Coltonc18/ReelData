from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def regressive_model(label_column, error=0.2):
    master_df = pd.read_csv('data/master_dataset.csv')
    master_df = master_df[['adult', 'genres', 'budget', 'original_language', 'production_companies', 'production_countries', 'runtime', 'user_rating', 'revenue', 'expert_rating']]

    if 'revenue' in label_column:
        filtered_df = master_df[master_df['revenue'] != 0.0]
    elif 'expert' in label_column:
        filtered_df = master_df[master_df['expert_rating'].notna()]
        label_column = 'expert_rating'
    elif 'user' in label_column:
        filtered_df = master_df[master_df['user_rating'].notna()]
        label_column = 'user_rating'
    else:
        return f'{label_column} is not a valid metric to train on'
    
    filtered_df = pd.get_dummies(filtered_df).dropna()
    
    features = filtered_df.drop(['user_rating', 'expert_rating', 'revenue'], axis='columns')
    labels = filtered_df[label_column]
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

    accuracies = []
    for depth in range(1, 100, 2):

        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(features_train, labels_train)

        # Compute training accuracy
        train_predictions = model.predict(features_train)
        close = 0
        for prediction, actual in zip(train_predictions, labels_train):
            if abs(prediction / actual - 1) <= error:
                close += 1
        train_acc = close/len(train_predictions)*100
        # print('Train Accuracy:', train_acc, '%')

        # Compute test accuracy
        test_predictions = model.predict(features_test)
        close = 0
        for prediction, actual in zip(test_predictions, labels_test):
            # print(f'Prediction: {prediction}, Actual: {actual}')
            if abs(prediction / actual - 1) <= error:
                close += 1
        test_acc = close/len(test_predictions)*100
        # print('Test  Accuracy:', test_acc, '%')

        accuracies.append({'max depth': depth, 'train accuracy': train_acc, 
                       'test accuracy': test_acc})
        
    accuracies = pd.DataFrame(accuracies)
    plot_accuracies(accuracies, 'train accuracy', 'Train', f'accuracy_graphs/{label_column}_RegressorTrain')
    plot_accuracies(accuracies, 'test accuracy', 'Test', f'accuracy_graphs/{label_column}_RegressorTest')

def plot_accuracies(accuracies, column, name, filepath):
    sns.relplot(kind='line', x='max depth', y=column, data=accuracies)
    plt.title(f'{name} Accuracy as Max Depth Changes')
    plt.xlabel('Max Depth')
    plt.ylabel(f'{name} Accuracy')
    plt.ylim(-1, 101)

    plt.savefig(filepath)
    plt.show()  # Display the graph
    

def neural_network():
    pass

if __name__ == '__main__':
    regressive_model('revenue', error=0.1)