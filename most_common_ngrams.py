import nltk
import streamlit as st
import pandas as pd
from nltk import FreqDist

def most_freq_n_grams(ndf, labels, n):
            for label in labels:
                df_sel= ndf[ndf['label']==label]
                
                fdist_per_label = {}
                text_content = ""
                
                for entry in df_sel['text']:
                    text_content += ". " + entry
                text_content = text_content.replace(". .",". ")
                text_content = text_content.replace("..",".")
                
                text_content = text_content.split()

                #text_content = nltk.word_tokenize(text_content) #Alternative to split()
                for num in range(1,5):
                    n_grams = nltk.ngrams(text_content,num)
                    #TODO get rid of n_grams spanning over sentence borders 
                    n_gram_list = [' '.join(grams) for grams in n_grams]
                    fdist_gram = FreqDist(n_gram_list)
                    fdist_per_label[str(label)+"_"+str(num)+"-gram"] = fdist_gram

                #st.write(fdist_per_label.keys())
                
                most_commons = pd.DataFrame()
                for key in fdist_per_label.keys():
                    ser_k = []
                    ser_v = []
                    to_add = fdist_per_label[str(key)].most_common(n)
                    for k,v in to_add:
                        ser_k.append(k)
                        ser_v.append(v)
                    ser_k = pd.Series(ser_k)
                    ser_v = pd.Series(ser_v)
                    most_commons[str(key)+' - word'] = ser_k
                    most_commons[str(key)+' - counts'] = ser_v 

                st.write('Label: ', str(label),  most_commons)
            return fdist_per_label

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def most_freq_ngrams_CountVec(ndf, labels, n, start, stop):
    
    for label in labels:
        df_sel= ndf[ndf['label']==label]
                
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
    return freq_per_label   
    
