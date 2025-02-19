import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import string
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
import gensim.downloader as api
import warnings
warnings.filterwarnings('ignore')
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import LsiModel
from gensim import corpora


def clean(doc):
    # remove stopwords, punctuation, and normalize the corpus
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def create_corpus(reviews):
    clean_corpus=[clean(review).split() for review in reviews]
    return clean_corpus
    
def create_dictionary_matrix(clean_corpus):

    # Creating document-term matrix 
    dictionary = corpora.Dictionary(clean_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_corpus]

    return dictionary,doc_term_matrix

def ModelTopics(type,tune,doc_term_matrix,dictionary,clean_corpus):

    if(type=="LSI"):    
        # LSA model
        lsa = LsiModel(doc_term_matrix, num_topics=3, id2word = dictionary)
        # LSA model
        print(lsa.print_topics(num_topics=3, num_words=3))
        if(tune=="Yes"):
            coherence_score=[]
            n_topics=[]
            for topics in range(2,20):
                n_topics.append(topics)
                # LSA model
                lsa = LsiModel(doc_term_matrix, num_topics=topics, id2word = dictionary)
                cm = CoherenceModel(model=lsa,texts=clean_corpus)
                coherence = cm.get_coherence()  # get coherence value
                coherence_score.append(coherence)

            plt.plot(n_topics,coherence_score)
            plt.show()
            
    if(type=="LDA"):
        # LDA model
        lda = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary)
        # Results
        print(lda.print_topics(num_topics=3, num_words=3))
        if(tune=="Yes"):
            coherence_score=[]
            n_topics=[]
            for topics in range(2,20):
                n_topics.append(topics)
                # LSA model
                lda = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary)
                cm = CoherenceModel(model=lda,texts=clean_corpus)
                coherence = cm.get_coherence()  # get coherence value
                coherence_score.append(coherence)

            plt.plot(n_topics,coherence_score)
            plt.show()

