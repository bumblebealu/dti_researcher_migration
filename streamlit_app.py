import pandas as pd
import streamlit as st
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Arxiv Dashboard")


df = pd.read_csv("reduced.csv")
if st.toggle("Paper Search"):
    st.write("Search Papers")
    # search for papers by title
    search = st.text_input("Search Papers by Title")
    # display the papers that match the search
    st.write(df[df["title"].str.contains(search,case=False)].head(20))

if st.toggle("Add Filters"):
    search = st.text_input("Must Include")
    df = df[df["title"].str.contains(search,case=False)]
    
df["year"] = df["versions"].str[-20:-16]
df["papers"] = 1

add_sidebar = st.sidebar.selectbox("Select Display",("Barchart","Exponential Fit","Top Authors","Abstract Lengths", "Paper Recommender"))
# make a barplot of the year column based on freqeuncy of each year sorted by year
if add_sidebar == "Barchart":
    tab1, tab2 = st.tabs(["Default", "Grouped By Category"])
    #  barchart of paper frequency by year grouped by category
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x='year:T',
        y='papers',
        color='categories:N',
    )
    with tab1:
        st.write("Paper Frequency by Year")
        st.bar_chart(df["year"].value_counts().sort_index())
    with tab2:
        st.write("Paper Frequncy by Year")
        st.altair_chart(chart, theme=None, use_container_width=True)
    
    
# display the above plot in the streamlit app


# fit papers per year to a function of form 2^(a(x-1992)) from 2000 to 2006
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def func(x, a,b):
    return b*2**(a*(x-1992))

# print the fitted function


# display the distribution of years in the year column in df
papers_per_year = df['year'].value_counts().sort_index()
# change index of papers_per_year to integers
papers_per_year.index = papers_per_year.index.astype(int)

popt, pcov = curve_fit(func, papers_per_year.index[:-2], papers_per_year[:-2])

if add_sidebar == "Exponential Fit":
    st.write("Doubling Time")
    # plot papers per year and the fitted function
    plt.scatter(papers_per_year.index[:-2], papers_per_year[:-2], label="y = b*2^(a(x-1992))"+"\na = " + str(popt[0]) + "\nb = " + str(popt[1]))
    plt.plot(papers_per_year.index[:-2], func(papers_per_year.index[:-2], *popt), 'r-', label='Fit')
    plt.scatter(papers_per_year.index[:-2],papers_per_year[:-2],color = "b", label = "Data")
    # add text to the legend of the graph
    
    print("y = b*2^(a(x-1992))")
    print("a = " + str(popt[0]))
    print("b = " + str(popt[1]))
    
    # add appropriate labels
    plt.xlabel('Year')
    plt.ylabel("Number of Papers Per Year")
    plt.title("Doubling Time is "+ str(1/popt[0]) + "years")
    # html figure
    plt.legend()
    st.pyplot()
    #plt.savefig('doubling.png')

if add_sidebar == "Top Authors":
    st.write("Top Authors")
    top_authors = df["submitter"].value_counts().sort_values(ascending=False).head(20)
    # sort top_authors by number of papers submitted
    top_authors_index = sorted(top_authors)
    # st.write(top_authors)
    
    st.bar_chart(top_authors)

if add_sidebar == "Abstract Lengths":
    # get the length of each abstract
    df["abstract_length"] = df["abstract"].str.len()
    # plot the distribution of abstract lengths
    st.write("Distribution of Abstract Lengths")
    plt.hist(df["abstract_length"],bins = 100)
    st.pyplot()
    #st.write(df["abstract_length"].describe())

if add_sidebar == "Paper Recommender":
        if st.toggle("FIRE UP THE RECOMMENDER"):
            trainer= TfidfVectorizer(stop_words='english') 
            trainer.fit(df.iloc[:200,11])
            tester = trainer.transform(df.iloc[:200,11])
            TFIDF = pd.DataFrame(tester.toarray(), columns=trainer.get_feature_names_out())
            # search for papers by title
            power_search = st.text_input("Search Papers by Title")
            power_vector = trainer.transform([power_search])
            word_list = power_search.split(" ")
            small_df = TFIDF[word_list]
            matching_index = small_df.sum(axis=1).sort_values(ascending=False).head(10).index[0]
            matching_title = df.iloc[matching_index,4]
            st.write(matching_title)
            st.write(matching_index)

        if st.toggle("FIRE UP THE RECOMMENDER 2.0"):
            trainer= TfidfVectorizer(stop_words='english') 
            trainer.fit(df.iloc[:200,11])
            tester = trainer.transform(df.iloc[:200,11])

            TFIDF = pd.DataFrame(tester.toarray(), columns=trainer.get_feature_names_out())
            # search for papers by title
            power_search = st.text_input("Search Papers by Title")
            power_vector = trainer.transform([power_search])
            word_list = power_search.split(" ")
            word_vector = trainer.fit(word_list)
            st.write(word_vector)