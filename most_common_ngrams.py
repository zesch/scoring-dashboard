import nltk
import streamlit as st
import pandas as pd
from nltk import FreqDist

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

#@st.cache(suppress_st_warning=True)
def most_freq_ngrams_CountVec(ndf, labels, n, start, stop):
    
    for label in labels:
        #st.write(label)
        #st.write(ndf.head())
        df_sel= ndf[ndf['label']==str(label)] #TODO cast over mappen für bessere Performance!
        if df_sel.empty:
            df_sel= ndf[ndf['label']==label]

        #st.write(df_sel)        
        freq_per_label = {}
        text_content = df_sel['text'].values
        

        for i in range (start, stop+1):
            count_vec = CountVectorizer(analyzer='word', ngram_range=(i,i))        
            vectorized_content = count_vec.fit_transform(text_content)
        # vectorized_content = each entry as sparse vector (0,1)
        # count_vec.vocabulary_ = dict of vocabulary:index
        # Warning: get_feature_names is deprecated > get_feature_names_out()~ might imply code changes
            counts = pd.DataFrame(vectorized_content.toarray(),columns=count_vec.get_feature_names())

            df_counts = counts.T.sort_values(by=0,ascending=False) # transpose word - count matrix, create DataFrame
            n_gram_counts = df_counts.sum(axis=1) # sum of all occurences per word/ngram
            n_gram_counts = n_gram_counts.sort_values(ascending=False) # sort values in descending order
            # create column from current indexes = ngrams, set new indexes + relabel columns
            n_gram_counts = n_gram_counts.reset_index().rename(columns={'index': str(label)+ ' - ' + str(i) + '-gram  ngram', 0: str(label)+ ' - ' + str(i) + '-gram count'})
            freq_per_label[str(label)+"-"+str(i)+"-gram"] = n_gram_counts

        most_commons = pd.DataFrame()
        for key in freq_per_label.keys():
            #st.write(str(key)+" - debug most_commons: ", fdist_per_label[str(key)])
        
            to_add = freq_per_label[str(key)].head(n)
            #st.write('to add: ', to_add)
            most_commons = pd.concat([most_commons, to_add], axis=1)

        st.write('Label: ', str(label),  most_commons)
        # wenn @cache muss das st.write ausgelagert werden..
    return freq_per_label   
    
