
###############################################################################
###################### DELIVERY 1 : DATA PRE-PROCESSING #######################
###############################################################################

# Importation of the libraries
import pandas as pd
import sklearn
import re
import string

# Pre-processing function
def preprocess_file(file, stop_words, numbers=0, freq_threshold=None):
    """
    This function will be used to clean a corpus, written on a .txt file
    
    Inputs : 
        file           : string - name of the file
        stop_words     : list<string> - list of the stop words
        numbers        : boolean - 1 if we remove the numbers
        freq_threshold : list<integer> - thresholds for useful words
        
    Outputs : 
        content        : string - preprocessed data

    """
    # Reading of the file
    with open(file, "r") as file:
        content = file.read()
        print("----------------------------------------------------------------")
        print("---------------- Corpus read in the document -------------------")
        print("----------------------------------------------------------------")
        print(content)
        print("----------------------------------------------------------------\n")
        
    # Change to lower case
    content = content.lower()
    
    # Remove URLs (http or https)
    content = re.sub("http??:\/\/.*[\r\n]*", "", content)
    
    # Remove mentions
    content = re.sub("@\S+", "", content)
    
    # Remove emails
    content= re.sub(r'\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b','',content)
    
    # Remove punctuations, commas and special characters
    punctuation = string.punctuation
    translation_table = str.maketrans('', '', punctuation)

    content = content.translate(translation_table)
    
    # Remove numbers
    if numbers:
        content = re.sub(r'\d+', '', content)
    
    # Remove stop words
    for word in stop_words:
        content = re.sub(word,'',content)
    
    # Remove NaN
    content = re.sub('nan','',content)
    
    # Remove frequent/rare words
    if freq_threshold != None:
        # dictionary that gives the number of occurrences of each word
        occurences={}
        # We create a list of all the words of the corpus
        data=content.split()
        
        # We read each word of the content
        for word in data:
            if word in occurences:
                # We increase the number of occurences
                occurences[word]+=1
            else:
                # We initialize to 1 the number of occurences
                occurences[word]=1
            
            if occurences[word]>freq_threshold[1]:
                # Remove frequent words
                content=re.sub(word,'',content)
                data=content.split()
        
        for word in occurences:
            # Remove rare words
            if occurences[word]<freq_threshold[0]:
                content = re.sub(word,'',content)
    
    # We return the cleaned content
    return content


# Tests of the function (with the document "Data_1.txt")

## First try without any particular parameter
file_name = "data.txt"
stop_words = []

print("*************** First test : no special parameter ****************") 
content_1 = preprocess_file(file_name, stop_words)
print(content_1)
print("****************************************************************\n") 
# We can see that all the special characters have been removed
## Also, the text is fully in lower case


## Second try, with special parameters
### We choose somme stop_words ("arbitrarily")
stop_words = ["in","is","of"]
### We remove the numbers
numbers=1

print("*********** Second test : use of special parameter *************")
content_2 = preprocess_file(file_name, stop_words, numbers=numbers) 
print(content_2)
print("****************************************************************\n") 
### We can easily observe that the stop words, and the numbers have been removed


## Third try, with frequency thresholds
### We remove the words that appear only one time, and those which appear more than 7 times
freq=[1,7]

print("************* Third test : frequency thresholds ****************") 
content_3 = preprocess_file(file_name, stop_words, numbers=numbers, freq_threshold=freq)
print(content_3)
print("****************************************************************\n") 
### We can easily observe that some words disappeared ("mathematics","applied" for example)


## Then, we can observe that this cleaning action can have a not negligible effect 
##    on the size of the dataset.
with open(file_name, "r") as file:
    content = file.read()
print("----------- Comparison of the size of each dataset -------------")
print("Initial length      : ",len(content))
print("Length for 1st test : ",len(content_1))
print("Length for 2nd test : ",len(content_2))
print("Length for 3rd test : ",len(content_3))

# In fact, with this example, we have reduced the size of the data by 25%.

