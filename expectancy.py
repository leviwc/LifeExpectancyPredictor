import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool, freeze_support

pd.set_option("display.max_rows", 6)
pd.set_option("display.max_columns", 14)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

EMPTY_TOKEN = '..'
DATA_FILE = 'world_indicators_database.csv'
dataset = pd.read_csv(DATA_FILE)
LIFE_EXPECTANCY_COLUMN = 'SP.DYN.LE00.IN'

# define classes of age as (1,2), (3, 4), (5,6) ...
def transform_to_class_range_two(age):
    age_class = None
    age = pd.to_numeric(age)
    # Calculate the class based on age range
    if age >= 0 and age < 100:
        age_class = "Class " + str((int(age) // 2)* 2) + "-" + str((int(age) // 2)* 2 + 1)

    return age_class

# define classes of age as (0, 4), (5, 9), (10, 14) ...
def transform_to_class_range_five(age):
    age_class = None
    age = pd.to_numeric(age)
    # Calculate the class based on age range
    if age >= 0:
        lower_bound = (age // 5) * 5
        upper_bound = lower_bound + 4
        age_class = "Class " + str(lower_bound) + "-" + str(upper_bound)

    return age_class

# pivot dataset and change name
def pivot_dataset(dataset, col_name):
    # Pivot to use the indices as columns, new index will be country Name - year
    dataset_with_of_col = dataset.pivot(index= 'Country Name', columns='Series Code', values=col_name)

    # add col_name in Country Name
    dataset_with_of_col.index = dataset_with_of_col.index + ' - ' + col_name

    return dataset_with_of_col

# merge all years in one column
def merge_all_years_and_pivot(dataset):
    #new dataframe empty with columns series codes and country name as index with no rows
    columns = dataset['Series Code'].unique()
    new_dataset = pd.DataFrame(columns= columns)

    for column in dataset.columns:
        if column not in ['Country Name', 'Series Code']:

            new_dataset = pd.concat([new_dataset, pivot_dataset(dataset, column)], ignore_index=False )

    return new_dataset


# Counting empty values by column
def count_empty_values(dataset):
  empty_amount = {}
  for column in dataset.columns:
    empty_amount[column] = 0
    for value in dataset[column]:
      if value == EMPTY_TOKEN:
        empty_amount[column] += 1
  return empty_amount


# print unique values from column 'Series Name'
def print_unique_values():
  print(len(dataset['Series Name'].unique()))

# go through rows and make a dictionary with Series Code as key and Series Name as value
def make_series_dict():
    series_dict = {}
    for index, row in dataset.iterrows():
        series_dict[row['Series Code']] = row['Series Name']
    return series_dict

# Remove columns Country Code Series Name
def remove_columns(dataset):
  for column in dataset.columns:
    if column in ['Country Code', 'Series Name']:
      dataset.drop(column, axis=1, inplace=True)
  return dataset

# Get training set
def remove_same_country(train, test):
    # Extract the Country Name values from the test dataset
    test_country_names = test.index[0].split(' - ')[0]

    # Use boolean indexing to filter the train dataset
    training_set = train[~train.index.str.startswith(tuple(test_country_names))]

    # Return the training set
    return training_set

# get training set
def get_training_set(train, test):
    #drop indexes with in train that are share a begining of Country Name of test
    train = remove_same_country(train, test)

    x_train, y_train = train.drop(LIFE_EXPECTANCY_COLUMN, axis=1), train[LIFE_EXPECTANCY_COLUMN]
    x_test, y_test = test.drop(LIFE_EXPECTANCY_COLUMN, axis=1), test[LIFE_EXPECTANCY_COLUMN]

    return x_train, y_train, x_test, y_test

# use knn
def use_knn(train, test, k):
    # print(test)
    knn = KNeighborsClassifier(n_neighbors=k)
    x_train, y_train, x_test, y_test = get_training_set(train, test)

    train = knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return y_pred == y_test

# tests the model leaving one out for every country
def leave_one_out_with_knn(normalized_dataset, final_k):
    sum_got_wright = 0
    sum_total = len(normalized_dataset.index)
    # make leave one out testing for every country
    for country in normalized_dataset.index:
        # normalized_dataset without this country
        normalized_dataset_current = normalized_dataset.drop(country, axis=0)
        current_country = normalized_dataset.filter(items=[country], axis=0)
        #create dataset with only this country
        knn_result = use_knn(normalized_dataset_current, current_country, final_k)
        sum_got_wright += knn_result[country]
    res = sum_got_wright / sum_total * 100
    return res

def use_nayve_bayes(train, test):
    # Create a Naive Bayes classifier
    nb_classifier = GaussianNB()

    x_train, y_train, x_test, y_test = get_training_set(train, test)

    # Train the classifier using the training data
    nb_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = nb_classifier.predict(x_test)
    return y_pred == y_test



# tests the model leaving one out for every country
def leave_one_out_with_nayve_bayes(normalized_dataset):
    sum_got_wright = 0
    sum_total = len(normalized_dataset.index)
    # make leave one out testing for every country
    for country in normalized_dataset.index:
        # normalized_dataset without this country
        normalized_dataset_current = normalized_dataset.drop(country, axis=0)
        #create dataset with only this country
        current_country = normalized_dataset.filter(items=[country], axis=0)
        nayve_bayes_result = use_nayve_bayes(normalized_dataset_current, current_country)
        sum_got_wright += nayve_bayes_result[country]
    res = sum_got_wright / sum_total * 100
    return res

def use_mlp(args):
    train, test = args
    # Create a Naive Bayes classifier
    nb_classifier = MLPClassifier(hidden_layer_sizes=(14,), max_iter=3000, activation='relu')

    x_train, y_train, x_test, y_test = get_training_set(train, test)

    # Train the classifier using the training data
    nb_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = nb_classifier.predict(x_test)
    return y_pred == y_test

# Function to perform leave one out testing in parallel
def leave_one_out_with_mlp(normalized_dataset):
    sum_got_wright = 0
    sum_total = len(normalized_dataset.index)

    # Prepare arguments for parallel processing
    args = [(normalized_dataset.drop(country, axis=0), normalized_dataset.filter(items=[country], axis=0))
            for country in normalized_dataset.index]

    # Perform parallel processing using multiprocessing.Pool
    with Pool() as pool:
        results = pool.map(use_mlp, args)

    # Calculate the accuracy
    for res in results:
        sum_got_wright += res.values[0]

    res = sum_got_wright / sum_total * 100
    return res

def get_max_k_value(dataset):
    max_k = 0
    max_value = 0
    for i in range(1, 100):
        res = leave_one_out_with_knn(dataset, i)
        if res > max_value:
            max_value = res
            max_k = i
    return max_k

def reorder_class(dataset):
    #reordering so the class is the last
    dataset_without_column = dataset.loc[:, dataset.columns != LIFE_EXPECTANCY_COLUMN]

    # Create a new dataframe with only the column to be moved
    column_to_move = dataset[LIFE_EXPECTANCY_COLUMN]
    return pd.concat([dataset_without_column, column_to_move], axis=1)

# make a dictionary with Series Code as key and Series Name as value for visualization
code_to_name_of_features = make_series_dict()

def process_data(dataset):


    # Remove columns from dataset that aren't Country Name, Series Code, or any year
    dataset = remove_columns(dataset)
    dataset.to_csv('prints/after_drop.csv')

    # Pivot to use the indices as columns, new index will be country Name - year
    dataset = merge_all_years_and_pivot(dataset)

    ## Only 2019
    # dataset = dataset.pivot(index='Country Name', columns='Series Code', values='2019 [YR2019]')

    # remove rows with '..' values in LIFE_EXPECTANCY_COLUMN rows
    dataset = dataset[dataset[LIFE_EXPECTANCY_COLUMN] != EMPTY_TOKEN]
    dataset.to_csv('prints/after_pivot.csv')

    # substitute empty values with null
    dataset.replace(EMPTY_TOKEN, np.nan, inplace=True)

    # transform the target class
    dataset[LIFE_EXPECTANCY_COLUMN] = dataset[LIFE_EXPECTANCY_COLUMN].apply(transform_to_class_range_five)

    # Concatenate the two dataframes
    dataset = reorder_class(dataset)
    dataset.to_csv('prints/after_class_transformation.csv')

    # Filter rows with more than 14 nulls
    dataset = dataset[dataset.isnull().sum(axis=1) <= 14]

    # Drop columns with any null values
    dataset = dataset.dropna(how='any', axis=1)
    dataset.to_csv('prints/after_class_removing_rows_and_columns_for_null.csv')

    columns_to_normalize = dataset.columns.drop(LIFE_EXPECTANCY_COLUMN)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the selected columns
    dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

    # Create a new dataframe with normalized values
    normalized_dataset = pd.DataFrame(dataset, columns=dataset.columns)

    normalized_dataset.to_csv('prints/after_normalizing.csv')

    return normalized_dataset

if __name__ == '__main__':
    freeze_support()
    dataset = process_data(dataset)

    final_k = 9
    print('Final K: ' + str(final_k))

    res = leave_one_out_with_knn(dataset, final_k)
    print('KNN: ' + str(res) + '%')

    res = leave_one_out_with_nayve_bayes(dataset)
    print('Naive Bayes: ' + str(res) + '%')

    res = leave_one_out_with_mlp(dataset)
    print('MLP: ' + str(res) + '%')