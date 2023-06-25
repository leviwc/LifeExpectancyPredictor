import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


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
def remove_columns():
  for column in dataset.columns:
    if column not in ['Country Name', 'Series Code', '2019 [YR2019]']:
      dataset.drop(column, axis=1, inplace=True)

# iterate through Brazil's columns and print them
def print_brazil(dataset):
    for column in dataset.loc['Brazil']:
        print(column)

# use knn
def use_knn(train, test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    x_train, y_train = train.drop(LIFE_EXPECTANCY_COLUMN, axis=1), train[LIFE_EXPECTANCY_COLUMN]
    x_test, y_test = test.drop(LIFE_EXPECTANCY_COLUMN, axis=1), test[LIFE_EXPECTANCY_COLUMN]

    train = knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    return y_pred == y_test

# tests the model leaving one out for every country
def leave_one_out_with_knn(normalized_dataset, final_k):
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

# Remove columns from dataset that aren't Country Name, Series Code, 2019 [YR2019]
remove_columns()

# Pivot to use the indices as columns
dataset = dataset.pivot(index='Country Name', columns='Series Code', values='2019 [YR2019]')


# remove rows with '..' values in LIFE_EXPECTANCY_COLUMN rows
dataset = dataset[dataset[LIFE_EXPECTANCY_COLUMN] != EMPTY_TOKEN]

# substitute empty values with null
dataset.replace(EMPTY_TOKEN, np.nan, inplace=True)

# transform the target class
dataset[LIFE_EXPECTANCY_COLUMN] = dataset[LIFE_EXPECTANCY_COLUMN].apply(transform_to_class_range_five)

# Drop columns with any null values
dataset = dataset.dropna(how='any', axis=1)

columns_to_normalize = dataset.columns.drop(LIFE_EXPECTANCY_COLUMN)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the selected columns
dataset[columns_to_normalize] = scaler.fit_transform(dataset[columns_to_normalize])

# Create a new dataframe with normalized values
normalized_dataset = pd.DataFrame(dataset, columns=dataset.columns)

final_k = 43
sum_got_wright = 0
sum_total = len(normalized_dataset.index)

res = leave_one_out_with_knn(normalized_dataset, final_k)

print(str(res) + '%')