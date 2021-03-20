import enchant
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker

sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)

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

def count_misspellings(words):
    '''
    Args:
        words (list): words to determine if misspelled

    Returns:
        the ratio of misspelled words to total words in email
    '''
    words = re.sub(r'[^a-zA-Z]', ' ', words)
    words = words.lower().split()
    count = 0
    misspellings = spell.unknown(words)
    for word in words:
        if word in misspellings:
            count += 1
    return count/len(words)

def prepare_set(x):
    x['subject'] = x['subject'].fillna('')
    x['full'] = x['subject'] + ' ' + x['email']
    
    x_train = words_in_texts(final_words, x['full'])
    
    X = pd.DataFrame(x_train)
    X['no_of_symbols'] = x['full'].apply(lambda x: sum(not i.isalpha() and not i.isdigit() for i in x))
    X['no_of_digits'] = x['full'].apply(lambda x: sum(i.isdigit() for i in x))
    X['no_of_cap_letters'] = x['full'].apply(lambda x: sum(i.isupper() for i in x))
    X['is_re/fwd'] = x['full'].str.contains(r'Re:|Fw:').astype(int)
    X['ratio_misspellings'] = x['full'].apply(lambda x: count_misspellings(x))
    
    x_train = X.to_numpy()
    return x_train

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()

# Handle null values in data
original_training_data['subject'] = original_training_data['subject'].fillna('')

# Split the data into training and test sets
train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)

# Find frequency of some example words in Spam emails
my_words = ['@', '#', '%', '!', 'html']
t = train[['spam']]
found_words = words_in_texts(my_words, train['email'])
t['@'] = found_words[:, 0]
t['#'] = found_words[:, 1]
t['%'] = found_words[:, 2]
t['!'] = found_words[:, 3]
t['html'] = found_words[:, 4]
t = t.melt('spam')
t['spam'] = t['spam'].replace({0:'Not Spam', 1:'Spam'})
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=t['variable'], y=t['value'], hue=t['spam'], data=t, ci=None)
legend = ax.legend()
plt.title('Frequency of Words in Spam Emails')
plt.xlabel('Words')
plt.ylabel('Proportion of Emails')
plt.ylim(0.0, 1.0)
plt.show()

# We must do this in order to preserve the ordering of emails to labels for words_in_texts
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

# Common words in spam emails
final_words = ['please', 'html', 'free', 'urgent', 'help', 'loan', 'quick', 'money', 'drug', 'sex', 'win', 'act now',
 'bonus', 'order', 'limited','special', 'shipping', 'url', '!', '#', '@', '%', '\$']

spell = SpellChecker()

X_train_f = prepare_set(train.copy())
Y_train_f = train['spam']

X_val_f = prepare_set(val.copy())
Y_val_f = val['spam']

final_model = LogisticRegression(max_iter=5000)
final_model.fit(X_train_f, Y_train_f)
final_model.predict(X_train_f)

final_model_training_accuracy = final_model.score(X_train_f, Y_train_f)
final_model_val_accuracy = final_model.score(X_val_f, Y_val_f)
print("Training Accuracy: ", final_model_training_accuracy)
print("Validation Accuracy: ", final_model_val_accuracy)