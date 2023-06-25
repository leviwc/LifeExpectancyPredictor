import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option("display.max_rows", 20000)
pd.set_option("display.max_columns", 6)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

EMPTY_TOKEN = '..'

dataset = pd.read_csv('full_size.csv')
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

# Remove columns from dataset that aren't Country Name, Series Code, 2019 [YR2019]
def remove_columns(dataset):
  for column in dataset.columns:
    if column not in ['Country Name', 'Series Code', '2019 [YR2019]']:
      dataset.drop(column, axis=1, inplace=True)
  return dataset

# get training set
def get_training_set(train, test):
    x_train, y_train = train.drop(LIFE_EXPECTANCY_COLUMN, axis=1), train[LIFE_EXPECTANCY_COLUMN]
    x_test, y_test = test.drop(LIFE_EXPECTANCY_COLUMN, axis=1), test[LIFE_EXPECTANCY_COLUMN]

    return x_train, y_train, x_test, y_test

# use knn
def use_knn(train, test, k):

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
        #create dataset with only this country
        current_country = normalized_dataset.filter(items=[country], axis=0)
        knn_result = use_knn(normalized_dataset_current, current_country, final_k)
        sum_got_wright += knn_result[country]
    res = sum_got_wright / sum_total * 100
    return res

def use_nayve_bayes(train, test, k):
    # Create a Naive Bayes classifier
    nb_classifier = GaussianNB()

    # Define the bin edges for the three classes
    bin_edges = [-0.1, 0.3, 0.6, 1.1]

    # Define the labels for the three classes
    class_labels = [0, 1, 2]

    for column in train.columns:
        if column == LIFE_EXPECTANCY_COLUMN:
            continue
        # Use the cut() function to transform the values into classes
        train[column] = pd.cut(train[column], bins=bin_edges, labels=class_labels, include_lowest=True)


    x_train, y_train, x_test, y_test = get_training_set(train, test)

    # Train the classifier using the training data
    nb_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = nb_classifier.predict(x_test)
    return y_pred == y_test

# tests the model leaving one out for every country
def leave_one_out_with_nayve_bayes(normalized_dataset, final_k):
    sum_got_wright = 0
    sum_total = len(normalized_dataset.index)
    # make leave one out testing for every country
    for country in normalized_dataset.index:
        # normalized_dataset without this country
        normalized_dataset_current = normalized_dataset.drop(country, axis=0)
        #create dataset with only this country
        current_country = normalized_dataset.filter(items=[country], axis=0)
        nayve_bayes_result = use_nayve_bayes(normalized_dataset_current, current_country, final_k)
        sum_got_wright += nayve_bayes_result[country]
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

    # Remove columns from dataset that aren't Country Name, Series Code, 2019 [YR2019]
    dataset = remove_columns(dataset)

    # Pivot to use the indices as columns
    dataset = dataset.pivot(index='Country Name', columns='Series Code', values='2019 [YR2019]')

    # remove rows with '..' values in LIFE_EXPECTANCY_COLUMN rows
    dataset = dataset[dataset[LIFE_EXPECTANCY_COLUMN] != EMPTY_TOKEN]

    # substitute empty values with null
    dataset.replace(EMPTY_TOKEN, np.nan, inplace=True)

    # transform the target class
    dataset[LIFE_EXPECTANCY_COLUMN] = dataset[LIFE_EXPECTANCY_COLUMN].apply(transform_to_class_range_five)

    # Concatenate the two dataframes
    dataset = reorder_class(dataset)

    # Filter rows with more than 15 nulls
    dataset = dataset[dataset.isnull().sum(axis=1) <= 15]

    # # Fill empty columns that have null values
    # dataset = dataset.fillna(0.5, axis=1)


    # Drop columns with any null values
    dataset = dataset.dropna(how='any', axis=1)

    columns_to_normalize = dataset.columns.drop(LIFE_EXPECTANCY_COLUMN)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the selected columns
    dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

    # Create a new dataframe with normalized values
    normalized_dataset = pd.DataFrame(dataset, columns=dataset.columns)

    # print(normalized_dataset)

    return normalized_dataset

dataset = process_data(dataset)

final_k = 1
print(final_k)

res = leave_one_out_with_knn(dataset, final_k)
print(str(res) + '%')

res = leave_one_out_with_nayve_bayes(dataset, final_k)
print(str(res) + '%')