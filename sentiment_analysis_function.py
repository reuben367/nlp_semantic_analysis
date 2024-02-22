import spacy
import pandas as pd
import matplotlib.pyplot as plt
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.tokens import Doc

# function to take review file and produce pandas df with analysis of trends
def sentiment_analysis(file_path, delimiter):
    '''
    Function to take file path and perform 
    sentiment analysis of reviews in file
    '''
    # import relevant libraries
     
    # load spacy english model and textblob component
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('spacytextblob')

    # load review file into df
    df = pd.read_csv(file_path, sep = delimiter)
    # load reviews column 
    review_unclean_df = df["reviews.text"]
    # remove any missing values
    review_df = review_unclean_df.dropna()

    # turn reviews into spacy doc objects
    doc_list = []
    # only processing first 1000 values to reduce processing time, my laptop is sloooow
    for index, sentence in enumerate(review_df[0:1000]): 
        sentence = sentence.strip()
        sentence = sentence.lower()
        doc = nlp(sentence)
        doc_list.append(doc)
    
    # tokenize each review
    token_list = []
    for doc in doc_list:
        tokenized_sentence = [token.text for token in doc if not token.is_punct | token.is_space | token.is_stop ]
        sentence_text = ' '.join(tokenized_sentence)
        tokenized_sentence_text = nlp(sentence_text)
        token_list.append(tokenized_sentence_text)
    

    # get polarity and subjectivity for all reviews
    polarity_list = []
    subjectivity_list = []
    for sentence in token_list:
        sentiment = sentence._.blob.sentiment
        polarity = round(sentiment.polarity,2)
        subjectivity = round(sentiment.subjectivity,2)
        polarity_list.append(polarity)
        subjectivity_list.append(subjectivity)

    # store review tokens and scores in pandas df
    data = pd.DataFrame( {'Review tokens' : token_list, 'Polarity': polarity_list, 
                          'Subjectivity': subjectivity_list, })

    # print histogram of polarity
    plt.hist(data["Polarity"], bins = 100)
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.title('Polarity of reviews')
    plt.show()

    # print histrogram of subjectivity
    plt.hist(data["Subjectivity"], bins = 100)
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')
    plt.title('Subjectivity of reviews')
    plt.show()

    # print histogram of star ratings
    plt.hist(df["reviews.rating"], bins = 5)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Star rating of reviews')
    plt.show()

    return


sentiment_analysis('amazon_product_reviews.csv', ',')
