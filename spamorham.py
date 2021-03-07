import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from utils import fetch_and_cache_gdrive
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker

sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)

fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()

null_vals = original_training_data.isnull()
null_vals.sum(axis=0)
original_training_data['subject'] = original_training_data['subject'].fillna('')
null_vals = original_training_data.isnull()
null_vals.sum(axis=0)

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)

# Find frequency of words in Spam vs. Ham emails
my_words = ['www', '#', '%', '!', 'html']
t = train[['spam']]
found_words = words_in_texts(my_words, train['email'])
t['www'] = found_words[:, 0]
t['#'] = found_words[:, 1]
t['%'] = found_words[:, 2]
t['!'] = found_words[:, 3]
t['html'] = found_words[:, 4]
t = t.melt('spam')
t['spam'] = t['spam'].replace({0:'Ham', 1:'Spam'})
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=t['variable'], y=t['value'], hue=t['spam'], data=t, ci=None)
legend = ax.legend()
plt.title('Frequency of Words in Spam/Ham Emails')
plt.xlabel('Words')
plt.ylabel('Proportion of Emails')
plt.ylim(0.0, 1.0)

def words_in_texts(words, texts):
    '''
    Args:
        words (list): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array = []
    for i in words:
        indicator_array.append(texts.str.contains(i).values.astype(int))
    indicator_array = np.asarray(indicator_array)
    indicator_array = np.transpose(indicator_array)
    return indicator_array

train = train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts

spell = SpellChecker()

def count_misspellings(words):
    words = re.sub(r'[^a-zA-Z]', ' ', words)
    words = words.lower().split()
    misspellings = spell.unknown(words)
    count = 0
    for i in words:
        if i in list(misspellings):
            count += 1
    return count/len(words)

final_train = train.copy()
final_train['subject'] = final_train['subject'].fillna('')
final_train['full'] = final_train['subject'] + ' ' +  final_train['email']

# Common words in spam emails
final_words = ['please', 'html', 'free', 'urgent', 'help', 'loan', 'quick', 'money', 'drug', 'sex', 'win', 'act now', 'bonus', 'order', 'limited','special', 'shipping', 'url', '!', '#', '@', '%', '\$']

X_train_f = words_in_texts(final_words, final_train['full'])

Y_train_f = final_train['spam']

X = pd.DataFrame(X_train_f)
X['no_of_symbols'] = final_train['full'].apply(lambda x: sum(not i.isalpha() and not i.isdigit() for i in x))
X['no_of_digits'] = final_train['full'].apply(lambda x: sum(i.isdigit() for i in x))
X['no_of_cap_letters'] = final_train['full'].apply(lambda x: sum(i.isupper() for i in x))
X['is_re/fwd'] = final_train['full'].str.contains(r'Re:|Fw:').astype(int)
X['misspelling_ratio'] = final_train['full'].apply(lambda x: count_misspellings(x))

X_train_f = X.to_numpy()

final_model = LogisticRegression(max_iter=5000)
final_model.fit(X_train_f, Y_train_f)
final_model.predict(X_train_f)

final_model_training_accuracy = final_model.score(X_train_f, Y_train_f)
print("Training Accuracy: ", final_model_training_accuracy)

def prepare(x):
    x['subject'] = x['subject'].fillna('')
    x['full'] = x['subject'] + ' ' + x['email']
    
    x_train = words_in_texts(final_words, x['full'])
    
    X = pd.DataFrame(x_train)
    X['no_of_symbols'] = x['full'].apply(lambda x: sum(not i.isalpha() and not i.isdigit() for i in x))
    X['no_of_digits'] = x['full'].apply(lambda x: sum(i.isdigit() for i in x))
    X['no_of_cap_letters'] = x['full'].apply(lambda x: sum(i.isupper() for i in x))
    X['is_re/fwd'] = x['full'].str.contains(r'Re:|Fw:').astype(int)
    X['no_misspellings'] = x['full'].apply(lambda x: count_misspellings(x))
    
    x_train = X.to_numpy()
    return x_train

test_set = prepare(test)
test_predictions = final_model.predict(test_set)
