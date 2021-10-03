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