import logging,datetime
import argparse
import os
import traceback
from tqdm import tqdm
from datetime import datetime
# import the necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

ap = argparse.ArgumentParser(description="Script to find topic in text")
ap.add_argument("-d", "--debug", action="store_true", help="Set debug mode")
ap.add_argument("InputFile", type=str, help="Text file with dump of interactions")
args = ap.parse_args()

is_debug = args.debug
loglevel = logging.DEBUG if is_debug else logging.INFO

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) + os.sep)
BASE_DATA_DIR = BASE_DIR + "data"+os.sep
logname,ext=os.path.splitext(BASE_DIR+os.path.basename(__file__))
logname=logname+".log"
logging.basicConfig(format='%(asctime)s::%(levelname)s::%(message)s', filename=logname, filemode="w", level=loglevel, encoding = 'utf-8')

input_file = args.InputFile

logging.info("Start")

try:
    # Keywords to focus on
    keywords_to_focus_on = ['issue', 'problem', 'error', 'bug', 'defect', 'trouble','need']
    
    # Define number of topics
    num_topics = 20
    
    documents = []
    # define the text data
    #text_data = ["This is some sample text about topic A",
    #             "This is some more text about topic B",
    #             "And this is some text about topic C"]

    logging.info("Loading text")  
    total_lines = sum(1 for _ in open(input_file, 'r',encoding='utf-8'))           
    with open(input_file,'r',encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc='Processing lines'):
            documents.append(line.strip())
    logging.info(str(total_lines)+" lines loaded")
    logging.info("Analyze text to find topics")

    logging.info("Load stopwords")
      # Download NLTK stop words (if not already downloaded)
    nltk.data.path.append(r"C:\Users\gheno\AppData\Roaming\nltk_data")
    if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords')):
        nltk.download('stopwords')

    # English stop words from NLTK
    english_stop_words = stopwords.words('english')

    # French stop words from NLTK
    french_stop_words = stopwords.words('french')

    # Combine both stop word lists
    stop_words = english_stop_words + french_stop_words

    logging.info("*** Non-Negative Matrix Factorization (NMF) for topic modeling")
    logging.info("NMF Vectorize")

    # Create a TF-IDF vectorizer with combined stop words
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)

    logging.info("NMF Fit and Transform")
    # Fit and transform the documents
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

      # Run NMF
    logging.info("Run NMF")
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

    # Display top words for each topic
    logging.info("Display topics, top "+str(num_topics))
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf_model.components_):
        logging.info("Topic %d:" % (topic_idx + 1)+" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))

    logging.info("NMF Fit and Transform with keywords to focus on")
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Adjust term frequencies to focus on specified keywords
    for keyword in keywords_to_focus_on:
        if keyword in tfidf_vectorizer.vocabulary_:
            # Increase term frequency for specified keywords
            tfidf_matrix[:, tfidf_vectorizer.vocabulary_[keyword]] *= 10

    # Run NMF
    logging.info("Run NMF with keywords to focus on")
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

    # Display top words for each topic
    logging.info("Display topics, top "+str(num_topics))
    feature_names = tfidf_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf_model.components_):
        logging.info("Topic %d:" % (topic_idx + 1)+" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]]))

    """ logging.info("*** Latent Dirichlet Allocation (LDA) for topic modeling")
    logging.info("LDA Vectorizing text")
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
    tf = count_vectorizer.fit_transform(documents)

    logging.info("Running LDA")
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=1)
    lda_matrix = lda_model.fit_transform(tf)

    # Display top words for each topic
    logging.info("Displaying topics, top %d" % num_topics)
    feature_names = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        logging.info("Topic %d:" % (topic_idx + 1)+" ".join([feature_names[i] for i in topic.argsort()[:-6:-1]])) """

    logging.info("End")

except Exception as err:
    logging.error(err)
    print(traceback.format_exc())
