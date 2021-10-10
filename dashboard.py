import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from ml import cross_validate, cross_val_dtree, cross_val_reg, cross_val_mat_SVM, cross_val_test
from most_common_ngrams import most_freq_n_grams, most_freq_ngrams_CountVec

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#nltk.download()

import plotly.express as px

# set page dimension, title and icon
st.set_page_config("Scoring Dashboard", None, "wide", "auto")

body, stats = st.columns((4, 1))

# Aufteilung des Codes in Container-Bereiche entsprechend den (inhaltlichen) Teilbereichen
load_data = st.container()
data_stats = st.container()
machine_learning = st.container()
ml_stats = st.container()

body.write(
"""
# Free-text Scoring Dashboard

Analyze how well your free-text data can be automatically scored
"""
)

# files could have header or not - done
# TODO we should not rely on specific column names, but auto-detect what is probably what
# TODO take care of encodings!

uploaded_file = body.file_uploader("Upload dataset (csv format)")
df = None
ndf = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter="\t", encoding = "utf-8") # add encoding?
    n_columns = len(df.columns)
    
#---------------------------------------Sidebar------------------------------------------------
    st.sidebar.header('Configuration')

    with st.sidebar.expander("Data Upload", expanded=True):
        st.markdown("#### Assign the content of the first row of your data")
        option01 = st.radio("First row contains :", ["Header", "Data"])

        st.markdown("#### Confirm the preselections \n **or** select the corresponding columns: ")
        
        with st.form('Chose Columns'):
            col_choices = [None]
            for i in range(n_columns):
                col_choices.append(str(i))
        
            first_row = []
            for col in df.columns:
                first_row.append(col)

            if "id" in first_row:
                ind_id = first_row.index("id")
                col_id = st.selectbox("Select ID column", col_choices, index=ind_id+1)
            else:
                col_id = st.selectbox("Select ID column", col_choices)
            if "text" in first_row:
                ind_text = first_row.index("text")
                col_text = st.selectbox("Select text column", col_choices, index=ind_text+1)
            else:
                col_text = st.selectbox("Select text column", col_choices)

            if "label" in first_row:
                ind_label = first_row.index("label")
                col_label = st.selectbox("Select label column", col_choices, index=ind_label+1)
            else:
                col_label = st.selectbox("Select label column", col_choices)

            submitted01 = st.form_submit_button('Submit')
            #if submitted01:
            #    st.write(col_id, col_text, col_label)
#---------------------------------Sidebar - break ---------------------------------------------------

    with load_data:

        # attention: lines that are outcommented in multiple lines will be shown as code in the app !

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

        # ggf noch Denkfehler drin, soll für jedes Paar der Spalten von Columns und der ersten Datenspalte prüfen,
        # ob bei den gleichen Datentyp enthalten ()= Columns enthält evtl. Daten) oder nicht (= enthält Header)
 
        hasNoHeader = [] # implementiert über list of booleans und anschließende Ver-Undung
        for col in df.columns:
            #st.write(type(col))
            #st.write(type(df[col][0]))
            try:
                header_item = int(col)
                data_item = int(df[col][0])
                if type(header_item) == type(data_item):
                    hasNoHeader.append(True)
                else:
                    hasNoHeader.append(False)

            except ValueError:
                try:
                    header_item = float(col)
                    data_item = float(df[col][0])
                    if type(header_item) == type(data_item):
                        hasNoHeader.append(True)
                    else:
                        hasNoHeader.append(False)

                except ValueError:
                    str(df[col][0])

        # hasNoHeader enthält die Progrose, ob die erste Zeile der csv Datei Daten enthält oder nicht
        if len(hasNoHeader) >= 1:
            hasNoHeader = all(hasNoHeader)
        else: 
            hasNoHeader = False

        # Text (Vor-) Verarbeitung, alles in lower case setzen
        for elem in first_row:
            if type(elem) == str:
               elem.lower()

        st.subheader("Preview of your data")
        st.write(df.head())
        
        # check if content is convertible to int
        def tryInt(item):
            try:
                item = int(item)
                return item
            except ValueError:
                return item

        # check if content is convertible to float
        def tryFloat(item):
            try:
                if not type(item) == int:
                    item = float(item)
                return item
            except ValueError:
                return item
        # string to int of '5.0' throws exception while string to float does not -> order 

        # add first line of file (now as header in columns) back as first line of data to dataframe
        # currently implemented for int data type
        # TODO implement also for float data type - done
        # ..other data types necessary?
        if option01 == "Data" and hasNoHeader:
            header = range(n_columns)
            data_to_add = {}
            new_columns = {}
            for i in header:
                converted_a = tryInt(df.columns[i])
                converted_b = tryFloat(converted_a)
                data_to_add.__setitem__(str(i),converted_b)
            
            new_frame = pd.DataFrame(columns=header)
            new_frame.loc[len(new_frame.index)] = df.columns
            
            for i in range(n_columns):
                new_columns.__setitem__(df.columns[i],str(i))

            df.rename(columns = new_columns, inplace=True)

            df.loc[-1] = data_to_add
            df.index = df.index + 1
            df = df.sort_index()

            st.write(df.head())

        # catch case when first line data types are not compatible to DataFrame data types (saved in hasNoHeader)    
        elif option01 == "Data" and not hasNoHeader:
            st.error("There are type mismatches between the first row and the second row of your data. Therefore we assume the presence of a header. Please check your file for inconsistencies. ")

        # do not proceed computing while text + label have not been selected yet!
        # TODO check if two distinct(!) columns have been selected at least..(data, label; id optional)

        # Submitted button for column choice
        if submitted01:
            st.write(col_id, col_text, col_label)
        
        if col_id is not None and col_text is not None and col_label is not None:
            col_id = int(col_id)
            col_text = int(col_text)
            col_label = int(col_label)
            new_df = pd.DataFrame(columns=["id","text","label"])
            new_df["id"] = df[df.columns[col_id]]
            new_df["text"] = df[df.columns[col_text]]
            new_df["label"] = df[df.columns[col_label]]
            ndf = new_df

        elif col_text is not None and col_label is not None:
            col_text = int(col_text)
            col_label = int(col_label)
            new_df = pd.DataFrame(columns=["text","label"])
            new_df["text"] = df[df.columns[col_text]]
            new_df["label"] = df[df.columns[col_label]]
            ndf = new_df
        else:
            st.error("You have not yet selected the mandatory columns text and/or label.")

        if ndf is not None and submitted01: 
            st.write("Your selection:", ndf.head())

# ab hier bei Zugriffen auf ID - prüfen, ob ID in df gesetzt ist!
# df ab hier umbenannt zu ndf (dataframe mit der Columnen-Auswahl, falls data hochgeladen wird, die mehr als nur die 2-3 Spalten enthält)

# ----------------------------------------Data Analysis Part-----------------------------------------------

if ndf is not None:

    zero_values = ndf[ndf['id'].isna()|ndf['text'].isna()|ndf['label'].isna()]
    ndf = ndf.dropna(subset=['text'])

    with data_stats:
        st.markdown('## **Dataset Statistics**')
        instances = len(ndf)
        c0, c01,c02,c_space = st.columns((1,1,1.5,1.5))

        with c0:
            st.metric(label = 'Number of instances: ', value=instances, delta=str(len(zero_values)*(-1))+" *")
            st.markdown("##### * Number of NaN instances")
            st.write(" ")

        if instances < 10:
            st.error("The amount of data is too low for correct analysis.")
            st.error("Programm will be terminated. Please try again with a larger data set.")
            st.stop()
            # 'st.stop' stops further processing
            # noch einen Regulator einfügen? > 'stop'
        elif instances < 100:
            st.warning("Data amount low. Results might be skewed.")
        else:
            st.success("Data amount is sufficient.")
            
        #@TODO Warning if too low
            labels = set(ndf["label"])
            
        with c01:    
            st.metric(label = "Number of labels: ", value=len(labels))
        with c02:
            show_labels = str(labels)
            st.metric("Labels: ", show_labels )
        
        label_counts = ndf["label"].value_counts()
        label_freq = ndf["label"].value_counts(normalize=True)
          
        if len(labels) >= 9:
            num_labels = True
        else:
            num_labels = False
        st.markdown("### Label distribution")
        st.bar_chart(label_freq,width=110*len(labels), use_container_width=num_labels)
        st.info("*Hover over bars to see exact values")

    
        #fig3 = sns.catplot(x='label', kind='count',height=5, aspect=3, data=ndf)#,points='all')
        #st.pyplot(fig3)
        
        # TODO Warning if imbalanced
        # Darstellung der Label Verteilung: Bar Chart mit 110pixel*len(labels), Knackpunkt: 9 Daten-Kolumnen

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
                st.dataframe(ndf['text'])
            # average no tokens
            #av_df['test'] = ndf['text'].apply(lambda x: "Hallo."+ str(x))    # debug Test
            av_df['#chars/entry'] = ndf['text'].apply(lambda x: len(''.join(str(x).split())))
            av_df['#words/entry'] = ndf['text'].apply(lambda x: len(str(x).split()))
            #av_df['#sentences/entry'] = ndf['text'].apply(lambda x: len(str(x).split('.')))
            av_df['#chars/sentence'] = ndf['text'].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split('.')))
            av_df['#words/sentence'] = ndf['text'].apply(lambda x: len(str(x).split())/len(str(x).split('.')))
            av_df['label'] = ndf['label']
            #av_df['#chars/word'] = ndf['text'].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split()))
            
            # vocabulary = set(text.split())
            
            #@TODO text in die Tabelle hinzu, scrollbar seitlich? done: gelöst als 2 df nebeneinander
            #@TODO means + plot
            with c2:
                st.write("Average stats: counts + lengths")
                st.write(av_df)

        #TODO automated plotting for x labels..

        st.markdown('### Plot Test SNS displot')
        with st.expander(label="Show/Hide Distribution plots",expanded=True):
            fig1a = sns.displot(av_df, x="#chars/entry", hue='label', kde = True, height=5, aspect=3)
            fig1a.set_xlabels("Counts - #chars/entry")
            fig1a.set_ylabels("Instances")
            st.pyplot(fig1a)

            fig1b = sns.displot(av_df, x="#chars/sentence", hue='label', kde = True, height=5, aspect=3)
            fig1b.set_xlabels("Counts - #chars/sentence")
            fig1b.set_ylabels("Instances")
            st.pyplot(fig1b)

            fig1c = sns.displot(av_df, x="#words/entry", hue='label', kde = True, height=5, aspect=3)
            fig1c.set_xlabels("Counts - #words/entry")
            fig1c.set_ylabels("Instances")
            st.pyplot(fig1c)

            fig1d = sns.displot(av_df, x="#words/sentence", hue='label', kde = True, height=5, aspect=3)
            fig1d.set_xlabels("Counts - #words/sentence")
            fig1d.set_ylabels("Instances")
            st.pyplot(fig1d)

        st.markdown('### Plot Test - Plotly Boxplot')

        av_df_char = pd.DataFrame()
        av_df_word = pd.DataFrame()
        av_df_char['#chars/entry'] = av_df['#chars/entry']
        av_df_char['#chars/sentence'] = av_df['#chars/sentence']
        av_df_char['label'] = av_df['label']
        av_df_word['#words/entry'] = av_df['#words/entry']
        av_df_word['#words/sentence'] = av_df['#words/sentence']
        av_df_word['label'] = av_df['label']

        #fig0a = px.box(av_df, color='label')#,points='all')
        #fig0a.update_layout()#(width=400, height=500)
        #st.plotly_chart(fig0a)

        word_c, char_c = st.columns(2)
        with char_c:
            fig0b = px.box(av_df_char, color='label', color_discrete_sequence=px.colors.qualitative.Vivid)#,points='all')
            fig0b.update_layout(width=600, 
                                height=500,
                                xaxis_title="",
                                yaxis_title="Counts")
            st.plotly_chart(fig0b)
        with word_c:
            fig0c = px.box(av_df_word, color='label',color_discrete_sequence=px.colors.qualitative.Vivid)#,points='all')
            fig0c.update_layout(width=600, 
                                height=500,
                                xaxis_title="",
                                yaxis_title="Counts")
            st.plotly_chart(fig0c)
        st.info("*hover over plot to see more detailed stats")
        
        #TODO check out plotly distplot self build...
    
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
        for entry in ndf['text']:
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
        # last two parameters: start/stop of n-gram range fpr CountVectorizer
        n_gram_start = 1
        n_gram_stop = 3
        some_more_data = most_freq_ngrams_CountVec(ndf, labels, N, n_gram_start, n_gram_stop)

        # Type-Token Ratio
        st.subheader("Type-Token-Ratio")

        c5, c6, c7, c_space = st.columns((1,1,1,2))

        ttr_df = pd.DataFrame()
        ttr_df['TTR'] = ndf['text'].apply(lambda x: len(set(str(x).split()))/len(str(x).split()))
        ttr_df['label'] = ndf['label']
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

# Sidebar ----------------------------------------------------------------------------
    # Language
    option02 = st.sidebar.selectbox(
        'Which language?',
        ['German','English'])

    # Numerical Data?
    labels = st.sidebar.radio(
        "Label type?",
        ('Categorical', 'Numeric - discrete', 'Numeric continuous'))
        # categorical: String, dicrete: int, continuous: float

    #Algorithm?
    algorithm = st.sidebar.radio(
        "Algorithm?",
        ('SVM', 'Regression', 'Decision Tree'))
    

    # ML -------------------------------------------------------------------------------------
    st.markdown('# Machine Learning Stats')
    st.markdown('## Classifier Performance Preview')

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
    
    overview_res = cross_val_test(ndf, 10) # 
    ## moved following lines to function, so it returns a data frame
    #overview_scores = pd.DataFrame()
    #for key in overview_res.keys():
    #    overview_scores[str(key)+ " score"] = overview_res[key]
    overview_res_ = overview_res.reset_index().rename(columns={'index': 'fold', 'score': 'score'})
    overview_res_['fold']=overview_res_['fold'].values + 1
    overview_stats = [] # columns require multiple iterations, calculate only once
    for elem in overview_res.columns:
        overview_stats.append((str(elem[:len(elem)-6]),(round(overview_res_[elem].mean(),4)),(round(overview_res_[elem].std(),4))))
    
    c_name, c_score, c_std, c_plot = st.columns((0.7,0.7,0.7,3))
    with c_name:
        max_score = overview_stats[0][1]
        # also for std? max std and trends?
        for (name, mean,std) in overview_stats:
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
                            margin=dict(l=20,r=20,t=20,b=10))
            
        st.plotly_chart(fig4a)
        st.info("*hover over graphs for more details")

    if algorithm == 'SVM':
        results = cross_val_mat_SVM(ndf, 10)
    elif algorithm == 'Regression':
        results = cross_val_reg(ndf, 10)
    elif algorithm == 'Decision Tree':
        results = cross_val_dtree(ndf, 10, stops)

    scores = results['scores']

    df_scores = pd.DataFrame(data=scores, columns=['score'])
    scores_ = df_scores.reset_index().rename(columns={'index': 'fold', 'score': 'score'})
    scores_['fold']=scores_['fold'].values + 1
    
    st.markdown('## Chosen Classifier Performance')
    c7b, c7c, c8 = st.columns((1.15,0.9, 3))
    with c7b:
        st.metric(label="", value="")
        st.metric(label='Chosen algorithm: ',value = str(algorithm))
        st.metric(label = 'Mean Score: ', value=round(scores.mean(),4))#, delta=(len(zero_values)*(-1)))
        st.metric(label = 'Mean Std: ', value=round(scores.std(),4))
    with c7c:
        st.metric(label="", value="")
        st.write(df_scores) # wollen wir das ganz anzeigen? Oder nur den mean + std deviation?
    
    with c8:
    # Gesamtgröße auf Seite anpassen
    # > requires Work-around, streamlit always scales to max container width..
        #fig4b = px.box(scores_,x='fold', y='score', points='all')
        fig4b = px.box(scores, points='all')
        #fig4b.update_yaxes(range=[0.6,1.0])
        fig4b.update_layout(height = 450,
                            width=250, 
                            xaxis_title=" ",
                            yaxis_title="Score",
                            margin=dict(l=20,r=20,t=20,b=10))
            
        st.plotly_chart(fig4b)

    c9a, c9b,c10 = st.columns((0.25,6.5,6.5))
    with c9b:
        st.markdown('## **Confusion matrix**')
        conf_mat = confusion_matrix(results['y'],results['y_pred'])
        fig1 = plt.figure(figsize=(4,3))
        sns.heatmap(conf_mat, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('Gold label')
        plt.xlabel('Predicted label')
        st.pyplot(fig1)
  
    #with c10:
        st.markdown('## **Classification report**')
        stats = classification_report(results['y'],results['y_pred'], output_dict=True)
        fig2 = plt.figure(figsize=(5.4,4))
        # sns.heatmap(pd.DataFrame(stats).iloc[:-1,:].T, annot=True, cmap=plt.cm.Blues) # to exclude support
        # standard cmap "Blues", yellow-green-purple alternative: "viridism" red-blue: "coolwarm", red-black-ish: "magma"
        # dezent bunt: "cubehelix", "Spectral"
        sns.heatmap(pd.DataFrame(stats).iloc[:-1,:].T, annot=True, cmap=plt.cm.coolwarm)
        st.pyplot(fig2)

