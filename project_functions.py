# import packages

#!pip install fuzzywuzzy
import fuzzywuzzy
from fuzzywuzzy import process, fuzz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# load datasets
def load_data(filename):
    result = pd.read_csv(filename)
    return result

# plot state distribution pie charts
def state_pie(df, left_or_right):
    # Pie chart of state distribution in left data
    state_counts = df['state'].value_counts()
    plt.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
    plt.title(f'Distribution of State: {left_or_right}')
    plt.show()

# top 15 city distribution
def city_bar(df, left_or_right):
    city_counts = df['city'].value_counts().nlargest(15)
    plt.barh(city_counts.index, city_counts.values)
    plt.xlabel('Count')
    plt.ylabel('City')
    plt.title(f'Top 15 City Distribution: {left_or_right}')
    plt.show()

# zip code distribution by states
def zip_bar(df, left_or_right):
    # Top 5 zip codes in different states in left data
    state_zip_counts = df.groupby(['state', 'zip_code']).size().reset_index(name='count')
    state_zip_counts = state_zip_counts.sort_values(['state', 'count'], ascending=[True, False]).groupby('state').head(5)

    # Create a dictionary of state to zip codes mapping in left data
    state_zip_dict = {}
    for state, zip_code, count in state_zip_counts.values:
        if state in state_zip_dict:
            state_zip_dict[state].append((zip_code, count))
        else:
            state_zip_dict[state] = [(zip_code, count)]

    # plot a horizontal grouped bar chart for left data
    n_groups = 5
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8
    colors = ['r', 'g', 'b', 'c', 'm']

    for i, (state, zip_counts) in enumerate(state_zip_dict.items()):
        zip_codes = [x[0] for x in zip_counts]
        counts = [x[1] for x in zip_counts]
        rects = ax.barh(index + i * bar_width, counts, bar_width, alpha=opacity, color=colors[i % len(colors)], label=state)
        for j, rect in enumerate(rects):
            ax.text(rect.get_width() + 5, rect.get_y() + rect.get_height()/2, str(zip_codes[j]), ha='left', va='center')

    ax.set_xlabel('Count')
    ax.set_ylabel('Zip Code')
    ax.set_title(f'Top 5 Zip Codes in Different States: {left_or_right}')
    ax.set_yticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_yticklabels([str(i+1) for i in range(n_groups)])
    ax.legend()

    plt.tight_layout()
    plt.show()


# function: lowercase, remove punctuation, trailing strip
def standarlize(df, column):
    # lower case
    df[column] = df[column].str.lower()
    # remove punctuation
    df[column] = df[column].str.replace(r'[^\w\s]+', '')
    # remove trailing whitespace
    df[column] = df[column].str.strip()

# replace abbreviations in address to full names
def abb_to_full(df, column):
    # create a dictionary of abbreviations and corresponding full name
    abbreviations = {
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bcir\b': 'circle',
        r'\bct\b': 'court',
        r'\bexpy\b': 'expressway',
        r'\bfwy\b': 'freeway',
        r'\bln\b': 'lane',
        r'\bpky\b': 'parkway',
        r'\brd\b': 'road',
        r'\bsq\b': 'square',
        r'\bst\b': 'street',
        r'\bste\b': 'suite',
        r'\btpke\b': 'turnpike',
        r'\bn\b': 'north',
        r'\be\b': 'east',
        r'\bs\b': 'south',
        r'\bw\b': 'west',
        r'\bne\b': 'northeast',
        r'\bse\b': 'southeast',
        r'\bsw\b': 'southwest',
        r'\bnw\b': 'northwest'
    }
    df[column] = df[column].replace(abbreviations, regex=True)

# check unique cities
def unique_city(df):
    res = df['city'].unique()
    res.sort()
    return res

# get top 10 closest matches of a chosen city
def get_top10_match(city, city_list):
    res = fuzzywuzzy.process.extract(city, city_list, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    return res


# replace inconsistent data entry
def replace_matches_in_column(df, column, string_to_match, min_ratio = 80):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 80
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match

# check null values 
def sum_null(df, column):
    res = df[column].isnull().sum()
    return res

# drop null values
def drop_null(df, column):
    res = df.dropna(subset=[column], inplace=True)
    return res

# get state subset
def state(df, state):
    return df[df['state'] == state]

# get common zip codes in left and right datasets and number of counts
def zip_match(left, right):
    left_zip_counts = left['zip_code'].value_counts().reset_index()
    left_zip_counts.columns = ['zip_code', 'count']
    right_zip_counts = right['zip_code'].value_counts().reset_index()
    right_zip_counts.columns = ['zip_code', 'count']
    zip_merged_inner = pd.merge(left_zip_counts, right_zip_counts, 
                           on=['zip_code'],
                           how='inner')
    return zip_merged_inner

# fuzzy match
def find_matches1(left, right,threshold=0.80):
    results = []

    for index1, row1 in left.iterrows():
        for index2, row2 in right.iterrows():

            # Calculate name and address similarity
            name_similarity = fuzz.token_set_ratio(row1["name"], row2["name"])
            address_similarity = fuzz.token_set_ratio(row1["address"], row2["address"])
            city_similarity = fuzz.token_set_ratio(row1["city"], row2["city"])
  
            # Calculate confidence score
            confidence_score = (name_similarity * 0.4 + address_similarity * 0.4 + city_similarity * 0.2)  / 100

            if confidence_score > threshold:
                results.append((row1["business_id"], row2["entity_id"], confidence_score))

    matches = pd.DataFrame(results, columns=["left", "right", "confidence_score"])
    return matches

# find match between same zip code pairs
def find_match_byzip(zip_match_data, left_data, right_data, filename):
    # create an empty dataframe
    all_results = pd.DataFrame()
    for index,row in zip_match_data.iterrows():
        left = left_data[left_data['zip_code'] == row['zip_code']]
        right = right_data[right_data['zip_code'] == row['zip_code']]
        result = find_matches1(left, right,threshold=0.80)
        # concatenate the result to the empty DataFrame
        all_results = pd.concat([all_results, result])
    all_results.to_csv(filename, index=False)
    print(all_results)
    return all_results

# visualize matching results
def visualize_match(df):
    plt.hist(df['confidence_score'], bins=20, color='blue', edgecolor='black', alpha=0.5)

    plt.title('Histogram of Confidence Score')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')

    plt.show()