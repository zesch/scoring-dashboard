import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import base64
from ml import cross_validate

# set page dimension, title and icon
st.set_page_config("Scoring Dashboard", None, "wide", "auto")

body, stats = st.columns([4, 1])
#body, stats = st.columns((4, 1))

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

# TODO files could have header or not
# TODO we should not rely on specific column names, but auto-detect what is probably what
# TODO take care of encodings!

uploaded_file = body.file_uploader("Upload dataset (csv format)")
df = None
ndf = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter="\t")
    n_columns = len(df.columns)

    with load_data:
        first_row = []
        for col in df.columns:
            first_row.append(col)

        # attention: outcommenting multiple lines is shown in the app !
        # some pre-code for auto-detecting columns content
           
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

        # ggf noch Denkfehler drin, soll für jede Spalte von Columns und der ersten Datenspalte prüfen,
        # ob der gleiche Datentyp enthalten ist (= Columns enthält evtl Daten) oder nicht (= enthält Header)
        hasNoHeader = []
        for col in df.columns:
            #st.write(type(col))
            #st.write(type(df[col][0]))
            try:
                header_item = int(col)
                data_item = int(df[col][0])
                #st.write("is Int")
                if type(header_item) == type(data_item):
                    hasNoHeader.append(True)
                    #st.write("1. match found")
                else:
                    hasNoHeader.append(False)

            except ValueError:
                try:
                    header_item = float(col)
                    data_item = float(df[col][0])
                    #st.write("is float")
                    if type(header_item) == type(data_item):
                        hasNoHeader.append(True)
                        #st.write("2. match found")
                    else:
                        hasNoHeader.append(False)

                except ValueError:
                    str(df[col][0])
                    #st.write("convertible to string")
            #finally:
                #st.write(hasNoHeader)
        # hasNoHeader enthält die Progrose, ob die erste Zeile der csv Datei Daten enthält oder nicht
        if len(hasNoHeader) >= 1:
            hasNoHeader = all(hasNoHeader)
        else: 
            hasNoHeader = False
        #st.write(hasNoHeader)  

        for elem in first_row:
            if type(elem) == str:
               elem.lower()

        st.subheader("Preview of your data")
        st.write(df.head())
        
        st.subheader("Assign the content of the first row of your data")

        option = st.radio("First row contains :", ["Header", "Data"])

        def tryInt(item):
            try:
                item = int(item)
                return item
            except ValueError:
                return item

        if option == "Data" and hasNoHeader:
            header = range(n_columns)
            data_to_add = {}
            new_columns = {}
            for i in header:
                converted = tryInt(df.columns[i])
                data_to_add.__setitem__(str(i),converted)
            
            new_frame = pd.DataFrame(columns=header)
            new_frame.loc[len(new_frame.index)] = df.columns
            
            for i in range(n_columns):
                new_columns.__setitem__(df.columns[i],str(i))

            df.rename(columns = new_columns, inplace=True)

            df.loc[-1] = data_to_add
            df.index = df.index + 1
            df = df.sort_index()

            st.write(df.head())
        
        elif option == "Data" and not hasNoHeader:
            st.write("There are type mismatches between the first row and the second row of your data. Please check your file for inconsistencies. ")

        # while not selected text + label, stop!
        # check ob jeweils unterschiedliche Columns gewählt wurden..
        st.subheader("Confirm - if applicable - the preselections or select the corresponding columns: ")
        
        with st.form('Chose Columns'):
            col_choices = [None]
            for i in range(n_columns):
                col_choices.append(str(i))

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
                col_label = st.selectbox("Select/confirm label column", col_choices)

            submitted01 = st.form_submit_button('Submit')
            if submitted01:
                st.write(col_id, col_text, col_label)

        #st.write(col_id, col_text, col_label)
        #st.write(df.columns[col_id])
        #st.write(df[first_row[col_id]])
        
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
            st.write(Warning, "You have not selected the mandatory columns text and/or label.")

        if ndf is not None: 
            st.write("Your selection:", ndf.head())
            st.write("Press button to proceed with your selection")
            button1 = st.button("Continue")

# ab hier bei Zugriffen auf ID - prüfen, ob ID in df gesetzt ist!
# df ab hier umbenannt zu ndf (Columnen-Auswahl)

# ----------------------------------------Data Analysis -----------------------------------------------

if ndf is not None and button1:
    with data_stats:
        st.header('Dataset Statistics')
        c0, c01,c02 = st.columns((1,1,1))
        with c0:
            st.write(" ", "\n")
            st.write('Number of instances: ', str(len(df)))
        #@TODO Warning if too low

            labels = set(ndf["label"])
        # label ggf. sorten vor Anzeige?
            st.write("Number of labels: ", str(len(labels)))
            st.write("Labels: ", str(labels))
            
        with c01:
            label_counts = ndf["label"].value_counts()
            label_freq = ndf["label"].value_counts(normalize=True)
            st.write("Frequency of labels: ",label_freq)
        with c02:
            label_perc = label_freq * 100
            st.write("Percentage of labels: ",label_perc)
        
        if len(labels) <= 9:
            num_labels = True
        else:
            num_labels = False
        st.bar_chart(label_freq,width=110*len(labels), use_container_width=num_labels)
        
        #@TODO Warning if imbalanced
        #110*len(labels), if else Knackpunkt 9 Daten-Kolumnen
        #@TODO Container, und Container hier in 2 Columns splitten und dann den Plot am Container ausrichten
        # 
        #st.write(label_counts.min())
        #st.write(label_counts.max())

        #Ratio
        label_dist = label_counts.min()/label_counts.max()
        if label_dist >= 0.9:
            st.success("The labels are evenly distributed.")
        elif label_dist > 0.8:
            st.warning("Distribution is marginal/suboptimal but ok to work with..")
        elif label_dist > 0.5:
            st.error("Warning: Imbalanced label distribution...!!")
        else:
            st.error("Warning: The labels' distribution is skewed! This will most likely have a negative impact on your results!!")
        # Ausgabe mit Text-Hinterlegung UND Text, Barriere-Freiheit(?)
        # Headings, Barriere-Freiheit, Screen-Reader

        # Average Length mit Plot
        st.subheader("Average length data")
        # bug in streamlit DataFrame Anzeigebreite issue#371
        c1, c2 = st.columns((1,2))
        av_df = pd.DataFrame()
        st.write("\n")
        with c1:
            st.write("Data")
            st.dataframe(ndf['text'])
        # average no tokens
        #av_df['test###///'] = ndf['text'].apply(lambda x: "Hallo."+ str(x))    # debug Test
        av_df['#chars/entry'] = ndf['text'].apply(lambda x: len(''.join(str(x).split())))
        av_df['#words/entry'] = ndf['text'].apply(lambda x: len(str(x).split()))
        #av_df['#sentences/entry'] = df['text'].apply(lambda x: len(str(x).split('.')))
        av_df['#chars/sentence'] = ndf['text'].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split('.')))
        av_df['#words/sentence'] = ndf['text'].apply(lambda x: len(str(x).split())/len(str(x).split('.')))
        av_df['label'] = ndf['label']
        #av_df['#chars/word'] = df['text'].apply(lambda x: len(''.join(str(x).split()))/len(str(x).split()))
        # vocabulary = set(text.split())
        #@TODO text in die Tabelle hinzu, scrollbar seitlich?
        #@TODO means + plot
        with c2:
            st.write("Average stats: counts + lengths")
            st.write(av_df)
        #st.area_chart(av_df['#chars/entry'])
        st.line_chart(av_df)
        #st.bar_chart(av_df)
        
        st.subheader("Average values on whole data set selection")
        df_mean = av_df.mean()
        df_mean = df_mean.to_frame(name="average value")
        
        c3, c4 = st.columns((1,2))
        with c3:
            st.write("\n")
            st.write(df_mean)

        df_group_mean = av_df.groupby('label').mean()
        with c4:
            st.write("Grouped by label",df_group_mean)

        #st.area_chart(df_group_mean)
        st.line_chart(df_group_mean)
        st.bar_chart(df_group_mean)
        #@TODO get Scatterplot instead of line_chart!! seaborn, plotly...

        # Type-Token Ratio
        st.subheader("Type-Token-Ratio")

        c5, c6 = st.columns((2))

        ttr_df = pd.DataFrame()
        ttr_df['TTR'] = ndf['text'].apply(lambda x: len(set(str(x).split()))/len(str(x).split()))
        ttr_df['label'] = ndf['label']
        ttr_group_mean = ttr_df.groupby('label').mean()
        ttr_group_mean = ttr_group_mean.rename(columns={'TTR':"average TTR"})
        with c5:
            st.write("", ttr_df)
        with c6:
            st.write("left most column represents labels",ttr_group_mean)

        #st.area_chart(ttr_df['TTR'])
        st.line_chart(ttr_df['TTR'])
        #st.bar_chart(ttr_df['TTR'])

        #st.area_chart(ttr_group_mean)
        #st.line_chart(ttr_group_mean)
        st.bar_chart(ttr_group_mean)


    st.sidebar.header('Configuration')

    # Language
    option = st.sidebar.selectbox(
        'Which language?',
        ['German','English'])

    # Numerical Data?
    labels = st.sidebar.radio(
        "Label type?",
        ('Categorical', 'Numeric - discrete', 'Numeric continuous'))

    #Algorithm?
    algorithm = st.sidebar.radio(
        "Algorithm?",
        ('SVM', 'Regression', 'Decision Tree'))

    
    # ML
    st.write(ndf)
    
    # scores = cross_validate(df, 5)
    # st.write(scores)