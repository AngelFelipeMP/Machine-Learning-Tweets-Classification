import re
import string
import numpy as np

### add NLP dependences


import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer

### add ML dependences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB, CategoricalNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix,  classification_report
from numpy import mean


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer



def print_test():
  print('OK')


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = SnowballStemmer("spanish")
    stopwords_spanish = stopwords.words('spanish')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    #change sumbers to <number>
    tweet = re.sub(r'\w*\d\w*', "numtag ", tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_spanish and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


  
  
  
class tools():
  def __init__(self, data, vectorization, ngrams, pipeline, labels, models, metrics, results, mean_results):
    self.data = data
    self.vectorization = vectorization
    self.ngrams = ngrams
    self.pipeline = pipeline
    self.labels = labels
    self.models = models
    self.metrics = metrics
    self.results = results
    self.mean_results = mean_results

  def preprocess_data(self):
    ### Preprocess data and so add it to a new column 'clean_comments_list'
    self.data["clean_comments_list"] = self.data["comment"].apply(lambda x : process_tweet(x))

    ### Removed preprocess data from a list and add it to a new collum called 'clean_comments_string'
    self.data["clean_comments_string"] = self.data["clean_comments_list"].apply(lambda x: ' '.join(x))

    return self.data

  def feature_extraction(self):
    for type_vec, value in self.pipeline.items():
      for ngr in value.keys():

        if type_vec == 'tfidf':
          self.pipeline[type_vec][ngr]['vect'] = TfidfVectorizer(ngram_range=ngr)
          self.pipeline[type_vec][ngr]['X'] = self.pipeline[type_vec][ngr]['vect'].fit_transform(self.data.clean_comments_string)
          self.pipeline[type_vec][ngr]['skf'] = StratifiedKFold(n_splits=10)
          self.pipeline[type_vec][ngr]['skf'].get_n_splits(self.pipeline[type_vec][ngr]['X'], self.labels)

        else:
          self.pipeline[type_vec][ngr]['vect'] = CountVectorizer(ngram_range=ngr)
          self.pipeline[type_vec][ngr]['X'] = self.pipeline[type_vec][ngr]['vect'].fit_transform(self.data.clean_comments_string)
          self.pipeline[type_vec][ngr]['skf'] = StratifiedKFold(n_splits=10)
          self.pipeline[type_vec][ngr]['skf'].get_n_splits(self.pipeline[type_vec][ngr]['X'], self.labels)

  #### Handcraft metric for multiclass evaluation
  def cem_metric(self, conf_metrix):
    cem_metrix = np.zeros(conf_metrix.shape)

    for column in range(conf_metrix.shape[1]):
      for row in range(conf_metrix.shape[0]):
      
        if row == column :
          cem_metrix[row,column] = (conf_metrix.sum(axis=0)[column]/2)/conf_metrix.sum()
                                            
        elif row < column:
          cem_metrix[row,column] = (conf_metrix.sum(axis=0)[column]/2 + conf_metrix.sum(axis=0)[row:column].sum())/conf_metrix.sum()

        elif row > column:
          cem_metrix[row,column] = (conf_metrix.sum(axis=0)[column]/2 + conf_metrix.sum(axis=0)[column+1:row+1].sum())/conf_metrix.sum()

    cem_metrix= - np.log2( np.where(cem_metrix !=0, cem_metrix, cem_metrix+0000000.1 ))

    return np.sum(cem_metrix * conf_metrix.T) / np.sum( np.diag(cem_metrix) * conf_metrix.sum(axis=0))


      #### criate function to train the models
  def train_models(self, average_recall='binary', average_precision='binary', multiclass='False'):

    ### 10-Folds cross-validation
    for clf_name, clf in self.models.items():
      for nam_feature, values in self.pipeline.items():
        for n_gram, skf_tools in values.items():

        ## StratifiedKFold
          for train_index, test_index in skf_tools['skf'].split(skf_tools['X'], self.labels):

            X_train, X_test = skf_tools['X'][train_index], skf_tools['X'][test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            if clf_name == 'gauNB_c':
              X_train = X_train.todense()
              X_test = X_test.todense()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            self.results[clf_name][nam_feature][n_gram]['accuracy'].append(accuracy_score(y_test, y_pred))
            self.results[clf_name][nam_feature][n_gram]['recall'].append(recall_score(y_test, y_pred, average=average_recall))
            self.results[clf_name][nam_feature][n_gram]['precision'].append(precision_score(y_test, y_pred, average=average_precision))
            
            if multiclass == 'True':
            # if 'cem' in self.results[clf_name][nam_feature][n_gram]:
              self.results[clf_name][nam_feature][n_gram]['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
              self.results[clf_name][nam_feature][n_gram]['f1_weighted'].append(f1_score(y_test, y_pred, average='weighted'))
              self.results[clf_name][nam_feature][n_gram]['cem'].append(self.cem_metric(confusion_matrix(y_test, y_pred)))
            else:
              self.results[clf_name][nam_feature][n_gram]['f1'].append(f1_score(y_test, y_pred))

  def average_results(self, average_dict):
    ### Average model results
    for model_name, value in self.results.items():
      for feature_vector, ngra_metric in value.items():
        for ngram, metrics_results in ngra_metric.items():
          for metric, values_results in metrics_results.items():

            average_dict[model_name][feature_vector][ngram][metric] = mean(values_results)

    self.average_dict = average_dict

  ### create dataframe for our results
  def create_Data_Frame(self):

    ### Criate a pandas da Frame with all results
    df_results = pd.DataFrame.from_dict({(i,j,k): self.average_dict[i][j][k]
                              for i in self.average_dict.keys()
                              for j in self.average_dict[i].keys()
                              for k in self.average_dict[i][j].keys()},
                          orient='index')
    return df_results