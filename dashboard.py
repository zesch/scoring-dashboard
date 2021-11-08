from os import error, sep
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from ml import cross_val_test
from most_common_ngrams import most_freq_ngrams_CountVec

import re
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import plotly.express as px
 
def print_data_stats(label_type):
    with data_stats:
        st.markdown('## **Dataset Statistics**')
        nrof_instances = len(df)
        c1, c2, c3, c4 = st.columns((1, 1, 1.5, 1.5))

        with c1:
            st.metric(label = 'Number of instances: ', value=nrof_instances)

        if nrof_instances < 10:
            st.error("The amount of data is too low to train a machine learning model.")
            st.error("Please upload a dataset with more answers.")
            st.stop()
        elif nrof_instances < 100:
            st.warning("Data amount low. Results might be skewed.")
        else:
            st.success("Sufficient amount of data detected. Proceedings with machine learning part")

        #TODO connect with label type selection, auto-select or user select priority?
        labels = list(set(df[col_label]))
        label_type_idx = 0
        if type(labels[0]) == int or type(labels[0])==float:
            labels.sort()
            if type(labels[0]) == int:
                label_type_idx = 1
            elif type(labels[1]) == float:
                label_type_idx = 2

        with c2:    
            st.metric(label = "Number of labels: ", value=len(labels))
        with c3:
            show_labels = str(labels)
            st.metric("Labels: ", show_labels)
        
        label_freq = df[col_label].value_counts(normalize=True)
            
        if len(labels) >= 9:
            num_labels = True
        else:
            num_labels = False
        st.markdown("### Label distribution")
        st.bar_chart(label_freq, width=110*len(labels), use_container_width=num_labels)
        st.info("*Hover over bars to see exact values")

        #fig3 = sns.catplot(x='label', kind='count',height=5, aspect=3, data=ndf)#,points='all')
        #st.pyplot(fig3)
        
        # TODO Warning if imbalanced
        # Darstellung der Label Verteilung: Bar Chart mit 110pixel*len(labels), Knackpunkt: 9 Daten-Kolumnen

    with data_stats:
        # Ratio
        #label_dist = label_counts.min()/label_counts.max()
        #st.write(label_dist)
        label_dist = label_freq.min()/label_freq.max()
        #st.write(label_dist)
        if label_dist >= 0.9:
            st.success("The labels are evenly distributed.")
        elif label_dist > 0.8:
            st.warning("Distribution is suboptimal but ok to work with..")
        elif label_dist > 0.5:
            st.error("Warning: Imbalanced label distribution...!!")
        else:
            st.error("Warning: The labels' distribution is skewed! This will most likely negatively impact training!")
        # Ausgabe mit Text-Hinterlegung UND Text, Barriere-Freiheit(?)
        # >> Headings, Barriere-Freiheit, Screen-Reader

        # Average Length mit Plot
        st.markdown("### Average length statistics")
        # Achtung: bug in streamlit DataFrame Anzeigebreite issue#371
        with st.expander("Show single data statistics"):
            c1, c2 = st.columns((1,2))
            av_df = pd.DataFrame()
            st.write("\n")
            with c1:
                st.write("Original data")
                df.index = df.index +1
                st.dataframe(df[col_text])
            # average no tokens
            #av_df['test'] = ndf['text'].apply(lambda x: "Hallo."+ str(x))    # debug Test
            av_df['#chars/entry'] = df[col_text].apply(lambda x: len(''.join(str(x).split())))
            av_df['#words/entry'] = df[col_text].apply(lambda x: len(str(x).split()))
            #av_df['#sentences/entry'] = ndf['text'].apply(lambda x: len(str(x).split('.')))
            av_df['#chars/sentence'] = df[col_text].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split('.')))
            av_df['#words/sentence'] = df[col_text].apply(lambda x: len(str(x).split())/len(str(x).split('.')))
            av_df['label'] = df[col_label]
            #av_df['#chars/word'] = ndf['text'].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split()))
            
            # vocabulary = set(text.split())
            
            with c2:
                st.write("Average stats: counts + lengths")
                st.write(av_df)
        if label_type == 'Numeric - discrete' or label_type == 'Numeric continuous':
            sorted_av_df = av_df.sort_values(by=col_label)
        else:
            sorted_av_df = av_df

        st.markdown('### Plot Test SNS displot')
        with st.expander(label="Show/Hide Distribution plots",expanded=True):
            #fig1a = sns.displot(sorted_av_df, x="#chars/entry", hue='label', kde = True, height=5, aspect=3)
            #fig1a.set_xlabels("Counts - #chars/entry")
            #fig1a.set_ylabels("Instances")
            #st.pyplot(fig1a)

            #fig1b = sns.displot(sorted_av_df, x="#chars/sentence", hue='label', kde = True, height=5, aspect=3)
            #fig1b.set_xlabels("Counts - #chars/sentence")
            #fig1b.set_ylabels("Instances")
            #st.pyplot(fig1b)

            fig1c = sns.displot(sorted_av_df, x="#words/entry", hue='label', kind='kde', height=5, aspect=3)
            fig1c.set_xlabels("Counts - #words/entry")
            fig1c.set_ylabels("Proportion")
            st.pyplot(fig1c)

            fig1d = sns.displot(sorted_av_df, x="#words/entry", hue='label', kde = True, height=5, aspect=3)
            fig1d.set_xlabels("Counts - #words/entry")
            fig1d.set_ylabels("Instances")
            st.pyplot(fig1d)

        st.markdown('### Plot Test - Plotly Boxplot')

        av_df_char = pd.DataFrame()
        av_df_word = pd.DataFrame()
        av_df_char['#chars/entry'] = sorted_av_df['#chars/entry']
        #av_df_char['#chars/sentence'] = sorted_av_df['#chars/sentence']
        av_df_char['label'] = sorted_av_df['label']
        av_df_word['#words/entry'] = sorted_av_df['#words/entry']
        #av_df_word['#words/sentence'] = sorted_av_df['#words/sentence']
        av_df_word['label'] = sorted_av_df['label']

        #fig0a = px.box(av_df, color='label')#,points='all')
        #fig0a.update_layout()#(width=400, height=500)
        #st.plotly_chart(fig0a)

        c1, word_c = st.columns((0.1,2))
        #with char_c:
        #    fig0b = px.box(av_df_char, color='label', color_discrete_sequence=px.colors.qualitative.Vivid)#,points='all')
        #    fig0b.update_layout(width=600, 
        #                        height=500,
        #                        xaxis_title="",
        #                        yaxis_title="Counts")
        #    st.plotly_chart(fig0b)
        with word_c:
            fig0c = px.box(av_df_word, color='label',color_discrete_sequence=px.colors.qualitative.Vivid)#,points='all')
            fig0c.update_layout(width=225*len(labels), 
                                height=500,
                                xaxis_title="",
                                yaxis_title="Counts")
            st.plotly_chart(fig0c)
        st.info("*hover over plot to see more detailed stats")

    
        st.subheader("Average values on whole data set selection")
        df_mean = av_df.mean()
        df_mean = df_mean.to_frame(name="average value")

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric(label = 'Average #chars/entry: ', value=round(df_mean.iat[0,0],4))
        with c2:
            st.metric(label = 'Average #chars/sentence: ', value=round(df_mean.iat[2,0],4))
        with c3:
            st.metric(label = 'Average #words/entry: ', value=round(df_mean.iat[1,0],4))
        with c4:
            st.metric(label = 'Average #words/sentence: ', value=round(df_mean.iat[3,0],4))
        

        df_group_mean = av_df.groupby('label').mean()
        st.markdown("### Average values grouped by label")
        st.write(df_group_mean)
        #TODO build a nice table for display!
        #Veto against metric display: works good for two labels but not necessarily for more..

        st.header("Most frequent word n-grams")
        text_content = ""
        for entry in df[col_text]:
            text_content+= " "+ entry
        #text_content = text_content.replace("."," .") 
        # #TODO dealing with the full stops, punctuation, special characters..
        text_content = text_content.split()
        fdist = FreqDist(text_content)
        
        N = st.slider("Number of most common n-grams to display", value=20)  # get by user input?
        ### Function based on nltk ngrams
        #some_data = most_freq_n_grams(ndf, labels, N)
        #st.write(some_data)
        ### Function based on ContVectorizer algorithm
        # last two parameters: start/stop of n-gram range for CountVectorizer
        n_gram_start = 1
        n_gram_stop = 3

#        some_more_data = most_freq_ngrams_CountVec(df, labels, col_text, col_label, N, n_gram_start, n_gram_stop)

        # Type-Token Ratio
        st.subheader("Type-Token-Ratio")

        c5, c6, c7, c_space = st.columns((1,1,1,2))

        ttr_df = pd.DataFrame()
        ttr_df['TTR'] = df[col_text].apply(lambda x: len(set(str(x).split()))/len(str(x).split()))
        ttr_df['label'] = df[col_label]
        ttr_group_mean = ttr_df.groupby('label').mean()
        ttr_group_mean = ttr_group_mean.rename(columns={'TTR':"average TTR"})
        with c5:
            st.write("", ttr_df)

        with c6:
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write("\n")
            st.write(ttr_group_mean)
            st.info("*left-most column: labels")
            
        st.write("\n")
        st.bar_chart(ttr_group_mean, width=len(labels)*115, use_container_width=num_labels)

        #st.line_chart(ttr_df['TTR'])
        #st.bar_chart(ttr_df['TTR'])

def infer_label_type():
    # TODO
    return 0

st.set_page_config("Scoring Dashboard", None, "wide", "auto")
body, stats = st.columns((4, 1))

data_stats = st.container()
machine_learning = st.container()
ml_stats = st.container()

body.write(
"""
# Free-text Scoring Dashboard

Analyze how well your free-text data can be automatically scored
"""
)

df = None
with st.sidebar:
    st.header('Data Upload')
    uploaded_file = st.file_uploader("Upload dataset (csv/tsv format)")  
if uploaded_file:      
    df = pd.read_csv(uploaded_file, encoding = "utf-8")

    st.subheader("Preview of your data")
    # TODO why that?
    df.index = df.index + 1
    st.write(df.head())

    with st.sidebar.expander("Column selection", expanded=True):

        st.markdown("#### Confirm the preselections \n **or** select the corresponding columns: ")

        column_names = df.columns.tolist()
        
        with st.form('Chose Columns'):
            col_text = st.selectbox("Select text column", column_names)
            col_label = st.selectbox("Select label column", column_names)
            
            columns_selected = st.form_submit_button('Confirm')
        
# some pre-code for the auto-detection of column-content           
#        for elem in first_row:
#            try:
#                int(df[elem][0])
#                st.write("is Int")     
#            except ValueError:
#                try:
#                    float(df[elem][0])
#                    st.write("is float")
#                except ValueError:
#                    str(df[elem][0])
#                    st.write("convertible to string")
#            finally:
#                st.write()

# ----------------------------------------Data Analysis Part-----------------------------------------------
    

    # ndf = ndf.dropna(subset=['text']).reset_index()
    # ndf = ndf.dropna(subset=['label']).reset_index()
    
    label_type = 'Categorical'
    if columns_selected:
        print_data_stats(label_type)
        
# ---------------------------------------------- Sidebar ---------------------------------------------------
    
    st.sidebar.header('Configuration')

    # Language
    option02 = st.sidebar.selectbox(
        'Which language?',
        ['German','English'])

    # Numerical Data?
    label_type = st.sidebar.radio(
        "Label type?",
        ('Categorical', 'Numeric - discrete', 'Numeric continuous'), index = infer_label_type())
        # categorical: String, dicrete: int, continuous: float
    st.write(label_type)

    #Algorithm?
    algorithm = st.sidebar.radio(
        "Algorithm?",
        ('SVM', 'Regression', 'Decision Tree', 'Random Forest'))
# ------------------------------------------------------------------------------------------------------

    
    # ML -------------------------------------------------------------------------------------
    st.markdown('# Machine Learning Stats')
    c1, c2 = st.columns((0.8,1))
    with c1:
        st.markdown('## Classifier Performance Preview')
    with c2:
        st.info("*hover over graphs for more details")

    #stops_en = set(stopwords.words('english'))
    #stops_de = set(stopwords.words('german'))

    # workaround - für import nltk error

    stops_en = {'nor', "couldn't", 'why', 'before', "haven't", 'in', 's', 'once', 'been', 'as', 'their', "don't", 'while', 'which', "should've", 'from', 'him', 'if', "you're", 'through', 'or', 'ourselves', 'such', 'for', 'down', 't', 'up', 'few', 'same', 'he', 'themselves', 'until', 'himself', 'doesn', 'have', 'other', 'has', 'ours', 'weren', "doesn't", 'his', 'off', 'shan', 'into', 'couldn', 'those', 'under', 'needn', "isn't", 'her', 'so', 'too', 'me', 'you', 'with', 'again', 'own', "shouldn't", "wasn't", 'whom', 'aren', 'above', 'itself', 'll', "needn't", 'now', 'hadn', 'will', 'yours', 're', 'than', 'these', 'how', 'when', "that'll", 'against', 'more', 'what', 'about', 'does', 'we', 'can', 'hers', 'didn', 'ma', 'its', 'be', 'hasn', 'some', 'don', 'very', 'being', 'over', 'my', "she's", 'at', 'did', 'won', 'it', 'all', 'y', "weren't", "shan't", 'who', 'both', 'during', 'our', 'having', 'am', 'ain', "mustn't", "you'd", 'myself', 'were', 'shouldn', 'they', 'between', 'isn', 'out', "hasn't", 'and', 'on', 'd', 'to', 'not', 've', 'theirs', 'them', 'here', 'do', 'i', "you've", 'the', 'just', 'a', 'wouldn', 'further', 'no', "wouldn't", 'is', "hadn't", 'mustn', "it's", 'after', 'should', 'yourselves', 'an', 'only', 'there', 'had', 'wasn', 'doing', "aren't", 'most', 'this', 'herself', 'o', "won't", 'each', 'of', "you'll", "mightn't", 'she', "didn't", 'where', 'then', 'by', 'any', 'below', 'haven', 'because', 'are', 'that', 'your', 'yourself', 'but', 'm', 'was', 'mightn'}
    stops_de = {'diesem', 'als', 'jenes', 'aller', 'ohne', 'einmal', 'denselben', 'in', 'werde', 'weil', 'eurem', 'welche', 'nicht', 'zwischen', 'denn', 'unsere', 'daß', 'was', 'sein', 'des', 'deinen', 'viel', 'diesen', 'nun', 'einigen', 'anderer', 'das', 'würden', 'keinen', 'es', 'ander', 'zur', 'jetzt', 'einiger', 'manches', 'solches', 'ein', 'ihr', 'dich', 'euren', 'einig', 'dazu', 'etwas', 'welches', 'aber', 'einige', 'habe', 'weiter', 'dein', 'dies', 'sind', 'dasselbe', 'um', 'unter', 'über', 'vom', 'da', 'hatten', 'hier', 'machen', 'eine', 'dieselbe', 'jede', 'dieses', 'dir', 'musste', 'jedes', 'sollte', 'so', 'anderen', 'können', 'dessen', 'keinem', 'meinem', 'derselben', 'soll', 'wirst', 'im', 'uns', 'selbst', 'unserem', 'also', 'deines', 'will', 'hinter', 'dem', 'und', 'kann', 'sehr', 'euer', 'gewesen', 'anderr', 'manchen', 'auf', 'für', 'nur', 'solche', 'jenen', 'seinem', 'einem', 'anderm', 'keines', 'nach', 'meinen', 'dann', 'wie', 'ihre', 'ob', 'dass', 'ihm', 'manchem', 'auch', 'wir', 'ihren', 'keine', 'welcher', 'wollte', 'andere', 'durch', 'bin', 'derselbe', 'dieser', 'seinen', 'alle', 'anderes', 'unseren', 'ist', 'hatte', 'war', 'mich', 'meine', 'wo', 'der', 'den', 'hab', 'würde', 'mit', 'hin', 'am', 'seines', 'eurer', 'zum', 'wollen', 'vor', 'zu', 'wieder', 'andern', 'meines', 'ihn', 'du', 'demselben', 'doch', 'desselben', 'ins', 'sonst', 'eures', 'aus', 'allen', 'deinem', 'könnte', 'eines', 'er', 'welchen', 'meiner', 'wird', 'solchem', 'einer', 'allem', 'solcher', 'einen', 'warst', 'welchem', 'anderem', 'anders', 'mein', 'ihrer', 'jener', 'dieselben', 'noch', 'eure', 'oder', 'gegen', 'wenn', 'mir', 'unseres', 'zwar', 'an', 'jenem', 'keiner', 'man', 'jeden', 'jene', 'einiges', 'unser', 'von', 'sie', 'sondern', 'hat', 'diese', 'manche', 'haben', 'bist', 'alles', 'ihres', 'seine', 'deine', 'seiner', 'deiner', 'jeder', 'ihnen', 'weg', 'während', 'ihrem', 'die', 'kein', 'mancher', 'derer', 'euch', 'sich', 'waren', 'indem', 'nichts', 'werden', 'ich', 'jedem', 'bei', 'bis', 'damit', 'dort', 'solchen', 'einigem', 'muss'}

    if option02 == 'German':
        stops = stops_de
    elif option02 == 'English':
        stops = stops_en

    # cv value fest an Funktion übergeben oder als variablen Input vom User?
    
    overview_res = cross_val_test(df, 10, col_text, col_label)
    overview_res_ = overview_res.reset_index().rename(columns={'index': 'fold', 'score': 'score'})
    overview_res_['fold']=overview_res_['fold'].values + 1
    overview_stats = []
    
    for elem in overview_res.columns:
        overview_stats.append((str(elem[:len(elem)-6]),(round(overview_res_[elem].mean(),4)),(round(overview_res_[elem].std(),4))))
    
    c_name, c_score, c_std, c_plot = st.columns((0.7,0.7,0.7,3))
    with c_name:
        max_score = overview_stats[0][1]
        
        # also for std? max std and trends?
        for (name, mean, std) in overview_stats:
            st.write(" ")
            st.write(name)
            st.write(" ")
            st.write(" ")
            st.write(" ")
            if mean >= max_score:
                max_score = mean

    with c_score:
        for (name, mean,std) in overview_stats:
            if mean == max_score:
                st.metric(label = 'Mean Score: ', value=mean)
            else:
                st.metric(label = 'Mean Score: ', value=mean, delta=round(mean-max_score,2))

    with c_std:
        for (name, mean,std) in overview_stats:
            st.metric(label = 'Mean Std: ', value=std)
    with c_plot:
        fig4a = px.box(overview_res, points='all')
        #fig4a.update_yaxes(range=[0.6,1.0])
        fig4a.update_layout(height = 450,
                            width=600, 
                            xaxis_title="Classifier",
                            yaxis_title="Score",
                            margin=dict(l=20,r=20,t=20,b=0))
            
        st.plotly_chart(fig4a)
        
#----------------------------------------------- ML self implement construction site----------------------------------------
 
    from ml import test, get_clf
    classifier = algorithm
    ml_results = {}
    ml_results = test(get_clf(classifier), df, 10, col_text, col_label)
    #st.write(ml_results.keys())
    ml_res_df = pd.DataFrame()
    ml_res_df['accuracy'] = ml_results['accuracy']
    ml_res_df['precision'] = ml_results['precision']
    ml_res_df['recall'] = ml_results['recall']
    ml_res_df['f1_score'] = ml_results['f1_score']
    ml_res_df_ = ml_res_df.reset_index().rename(columns={'index': 'fold'})
    ml_res_df_['fold']=ml_res_df_['fold'].values + 1

    #st.write(np.mean(ml_results['accuracy']))
    #st.write(np.mean(ml_results['f1_score']))
    #st.metric(label='mean acc', value=mean_acc)

    #if algorithm == 'SVM':
    #    results = cross_val_mat_SVM(ndf, 10)
    #elif algorithm == 'Regression':
    #    results = cross_val_reg(ndf, 10)
    #elif algorithm == 'Decision Tree':
    #    results = cross_val_dtree(ndf, 10, stops)

    #scores = results['scores']

    #df_scores = pd.DataFrame(data=scores, columns=['score'])
    #scores_ = df_scores.reset_index().rename(columns={'index': 'fold', 'score': 'score'})
    #scores_['fold']=scores_['fold'].values + 1
    
    st.markdown('## Chosen Classifier Performance')
    st.metric(label='Chosen algorithm: ',value = str(algorithm))
    c7a, c7, c7b, c8 = st.columns((0.15,2,0.15,2)) 

    with c7:
        st.write("Results per fold")
        ml_res_df_.index = ml_res_df_.index + 1
        st.write(ml_res_df_) # wollen wir das ganz anzeigen? Oder nur den mean + std deviation?
    
    with c8:
    # Gesamtgröße auf Seite anpassen
    # > requires Work-around, streamlit always scales to max container width..
        fig4b = px.box(ml_res_df, points='all')
        #fig4a.update_yaxes(range=[0.6,1.0])
        fig4b.update_layout(height = 400,
                            width=533, 
                            xaxis_title=" ",
                            yaxis_title="Score",
                            margin=dict(l=20,r=20,t=20,b=10))
            
        st.plotly_chart(fig4b)
        
    c1, c2, c3, c4, c5 = st.columns((1,1,1,1,3))
    with c1:
        st.metric(label='Mean Accuracy', value=round(np.mean(ml_results['accuracy']),4))
    with c2:
        st.metric(label='Mean Precision', value=round(np.mean(ml_res_df['precision']),4))
    with c3:
        st.metric(label='Mean Recall', value=round(np.mean(ml_res_df['recall']),4))
    with c4:
        st.metric(label='Mean F1 score', value=round(np.mean(ml_res_df['f1_score']),4))
    with c5:
        st.info("*hover over graphs for more details")


    c9a, c9b,c10 = st.columns((0.25,6.5,6.5))
    with c9b:
        #st.markdown('## **Confusion matrix**')
        #conf_mat = confusion_matrix(results['y'],results['y_pred'])
        #fig1 = plt.figure(figsize=(4,3))
        #sns.heatmap(conf_mat, annot=True, cmap=plt.cm.Blues)
        #plt.tight_layout()
        #plt.ylabel('Gold label')
        #plt.xlabel('Predicted label')
        #st.pyplot(fig1)
 
        st.markdown('## **Confusion matrix **') # matrix 2.0
        conf_mat2 = confusion_matrix(ml_results['y_true'],ml_results['y_pred'])
        fig1b = plt.figure(figsize=(4,3))
        sns.heatmap(conf_mat2, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('Gold label')
        plt.xlabel('Predicted label')
        st.pyplot(fig1b)
  
    #with c10:
        #st.markdown('## **Classification report**')
        #stats = classification_report(results['y'],results['y_pred'], output_dict=True)
        #fig2 = plt.figure(figsize=(5.4,4))
        ## sns.heatmap(pd.DataFrame(stats).iloc[:-1,:].T, annot=True, cmap=plt.cm.Blues) # to exclude support
        ## standard cmap "Blues", yellow-green-purple alternative: "viridism" red-blue: "coolwarm", red-black-ish: "magma"
        ## dezent bunt: "cubehelix", "Spectral"
        #sns.heatmap(pd.DataFrame(stats).iloc[:-1,:].T, annot=True, cmap=plt.cm.coolwarm)
        #st.pyplot(fig2)

        st.markdown('## **Classification report **') # 2.0
        stats2 = classification_report(ml_results['y_true'],ml_results['y_pred'], output_dict=True)
        fig2b = plt.figure(figsize=(5.4,4))

        sns.heatmap(pd.DataFrame(stats2).iloc[:-1,:].T, annot=True, cmap=plt.cm.coolwarm)
        st.pyplot(fig2b)