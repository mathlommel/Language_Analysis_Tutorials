# Importation of the libraries

import re
import sklearn
import os
import scipy
import numpy as np
import matplotlib
from wordcloud import WordCloud
import string
import sklearn
import math
import string
import sys
#import textmining
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


########################### Creation of the corpus ############################

# Function that open the file, create the corpus, and clean the data
def generate_corpus(file_name):
    
    print("Name of the file : ", file_name)
    # Reading of the file
    with open("tweets.txt", "r") as file:
        # We create the corpus
        original_corpus = file.readlines()
    
    cleaned_corpus = []
    
    k=0
    # We clean each document of the corpus
    for tweet in original_corpus:
        # Change to lower case
        tweet = tweet.lower()
        
        # Remove the head of each tweet 
        tweet = tweet.replace('b\'','')
        tweet = tweet.replace('"b','')
        tweet = tweet.replace('b"','')
        
        # Remove URLs (http and https)
        tweet = re.sub("http?:\/\/.*[\r\n]*", "", tweet)
        tweet = re.sub("https?:\/\/.*[\r\n]*", "", tweet)
        
        # Remove emails
        tweet= re.sub(r'\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','',tweet)
        
        # Remove mentions
        tweet = re.sub("@\S+", "", tweet)
        
        # Remove punctuations, commas and special characters
        punctuation = string.punctuation
        translation_table = str.maketrans('', '', punctuation)
        
        tweet = tweet.translate(translation_table)
        
        # Remove numbers
        tweet = re.sub(r'\d+', '', tweet)
        
        # Remove the \n
        tweet = re.sub('\\n', '', tweet)
        
        if (not tweet.isspace()) and tweet != '':
            # If the tweet is still interesting
            cleaned_corpus.append(tweet)
        else:
            # If not, we delete it from the original corpus
            original_corpus.pop(k)
        
        k += 1
    
    print("Pre-processing succesfully computed.")
    print("Length of the initial data : ",len(original_corpus))
    print("Length of the cleaned data : ",len(cleaned_corpus))
    
    return (original_corpus, cleaned_corpus)

######################## Mathematical Representation ##########################

# Get the frequencies of each word
def get_freq_words(corpus):
    # We initialize the vectors
    corpus_frequencies = []
    corpus_words = []
    
    # For each tweet, we calculate the frequencies of its words
    for tweet in corpus:
        dic = {}
        
        for word in tweet.split():
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
            
            # If the word is not yet in corpus_words, we add it
            if word not in corpus_words:
                corpus_words.append(word)
    
        for word,freq in dic.items():
            dic[word] = freq / len(tweet.split())
        
        corpus_frequencies.append(dic)
        
    print("Word's frequency for each document successfully computed.")
    print("List of words seen in the corpus succesfully computed.")
        
    return (corpus_frequencies, corpus_words)

# Get the Document Term Matrix
def get_dtm(corpus):
    vec = CountVectorizer()
    X = vec.fit_transform(corpus)
    # Creation of the matrix
    term_matrix = pd.DataFrame(X.toarray(), columns = vec.get_feature_names_out())
    
    print("Document Term Matrix successfully computed.")
    return term_matrix


############################ Text Visualization ###############################

# Get the key words of each tweet 
def get_key_words(term_matrix):
    # We search for the most frequent words of each document
    key_words = term_matrix.apply(lambda x : x[x==x.max()].index.tolist(), axis=1)
    
    return key_words

def get_most_relevant(term_matrix, n_words):
    # We search for the most frequent words in the global corpus
    # We sum all the rows of the term_matrix
    words_occurences = term_matrix.sum(axis = 1)
    # We keep the n_words most used words
    most_relevant = words_occurences.nlargest(n_words)
    
    words = term_matrix.columns
    res = pd.DataFrame({'Words': words[most_relevant.index], 'Occurences':most_relevant.tolist()})
    return res

# Print the wordcloud of a given document
def get_wordCloud(corpus, id_document, save_name=''):
    
    document = corpus[id_document]
    
    # Create and generate a word cloud image
    wordcloud = WordCloud().generate(document)
    
    # Visualisation of the wordcloud
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(document)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    print("WordCloud successfully created.")
    # Save the image if needed
    if save_name != '':
        wordcloud.to_file(save_name)
        print("Save option : YES")
        print("File created : ",save_name)
    else: 
        print("Save option : NO")

############################# Document Similarity #############################

# Compute the angle between 2 documents
def dot(D1, D2): 
    res = 0.0
    for key in D1:
        if key in D2:
            res += (D1[key] * D2[key])
            
    return res
  
# Get the similarity between 2 documents
def document_similarity(frequencies, id_1, id_2):
    # We get the frequencies of each document
    D1 = frequencies[id_1]
    D2 = frequencies[id_2]
    
    # We compute and return the distance
    numerator = dot(D1,D2)
    D1_norm = math.sqrt(dot(D1,D1))
    D2_norm = math.sqrt(dot(D2,D2))
    denominator = D1_norm * D2_norm
    
    
    return  numerator / denominator
      

# Get the 2 most similar documents of the corpus
def get_most_similar(frequencies):
    # Number of documents
    nb_doc = len(frequencies)                    
    # We initialize the result
    res = document_similarity(frequencies, 0, 1)
    doc1 = 0
    doc2 = 1
    
    # We browse each couple of document
    for i in range(0,nb_doc-1):
        for j in range(i+1,nb_doc):
            # We determine the similarity of the documents
            value = document_similarity(frequencies, i, j)
                
            # If they are more similar, we change our solution
            if value > res:
                res = value
                doc1 = i
                doc2 = j
            
    return (doc1,doc2)
            
######################## Application of our functions #########################

# We read and clean the corpus
print("*********************** Pre-processing *************************") 
file_name = "tweets.txt"
original, corpus = generate_corpus(file_name)
print("****************************************************************\n")


# Get the Document Term Matrix
print("******************** Document Term Matrix **********************") 
term_matrix = get_dtm(corpus)
print("Document Term Matrix :")
print(term_matrix)
print("****************************************************************\n")

# Get the most relevant words from all documents
print("******************** Most Relevant words ***********************") 
print("Most relevant words for each document :")
key_words = get_key_words(term_matrix)
print(key_words)
print("\nMost relevant words from all documents :")
n_words = 10
most_relevant = get_most_relevant(term_matrix, n_words)
print(most_relevant)
print("****************************************************************\n")

# Get the frequencies of each word in each document
print("******************* Frequencies & Words ***********************") 
corpus_freq, corpus_words = get_freq_words(corpus)
print("Words seen in all the corpus : ",len(corpus_words))
print("****************************************************************\n")

# Get the wordCloud of a document
print("************************ WordCloud *****************************") 
id_document = 10
print("Let's show the wordcloud of the tweet n°{} of the corpus.".format(id_document))
get_wordCloud(corpus, id_document, 'tweet_{}.png'.format(id_document))

print("\n Comments :")
print("We can see that it is really difficult to find usefull wordclouds.")
print("In fact, a tweet is quite short, and contain fiew words only.") 
print("As a consequence, in some cases, the wordcloud shows quasi")
print("completely the tweet : we obtain many words, with similar frequencies.")
print("In other cases, we can find only a fiew keywords, which doesn't really")
print("help to understand what the tweet deals with... \n")
print("--> It is often difficult to extract any useful information from this illustration.")
print("****************************************************************\n")

# We can get the similarity between 2 documents
print("************************ Similarities *****************************")
id_doc1 = 0
id_doc2 = 1
print("Tweet n°1 : {}".format(id_doc1)) 
print("Tweet n°2 : {}".format(id_doc2)) 
similarity = document_similarity(corpus_freq, id_doc1, id_doc2)
print("Similarity : {}".format(similarity)) 
print("****************************************************************\n")

# We can also try to find the 2 most similar documents in the corpus
print("******************* Most Similar tweets ************************")
id1, id2 = get_most_similar(corpus_freq)
print("One of the most similar tweets :\n")
print("   - Tweet n°{}".format(id1)," : ", original[id1])
print("   - Tweet n°{}".format(id2)," : ", original[id2])
print("As we can see those 2 tweets are quite the same (the message conveyed is the same).")
print("****************************************************************\n")
