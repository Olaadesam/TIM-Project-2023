# -*- coding: utf-8 -*-
# simple end-to-end process to create a topic co-occurrence matrix

import streamlit as st
import pandas as pd

import wordcloud

# the gensim library is used for topic modeling
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# model

@st.cache(allow_output_mutation=True)
def load_corpus(file):
	documents = pd.read_csv(file)
	return documents

def read_stopwords(file):
	file = open(file, 'r')
	return [w.strip() for w in file.read().split('\n')]

def remove_hyphens(document):
	return document.replace('- ', '')

def tokenize(document):
	return [w.lower() for w in document.split()]

def corpus_to_tokens(corpus, user_stopwords):
	stopwords_en = read_stopwords("data/stopwords-en.txt")
	user_stopwords = [word.strip() for word in user_stopwords.split('\n')]
	stopwords = stopwords_en + user_stopwords
	# remove stopwords and numbers
	return [[w for w in tokenize(document) if w not in stopwords and w.isalnum()]
		for document in corpus['content']]

def tokens_to_bow(tokens, dictionary):
	return [dictionary.doc2bow(document) for document in tokens]

def fit_lda(tokens, number_of_topics, dictionary):
	lda = get_saved_topic_model(number_of_topics)
	if lda:
		return lda
	lda = LdaModel(tokens_to_bow(tokens, dictionary), number_of_topics, dictionary)
	set_saved_topic_model(number_of_topics, lda)
	return lda

# Get saved state from st.session_state
def get_saved_topic_model(number_of_topics):
	return st.session_state.get(f'topic_model_{number_of_topics}')

# Save state using st.session_state
def set_saved_topic_model(number_of_topics, lda):
	st.session_state[f'topic_model_{number_of_topics}'] = lda

def ids_to_words(bow, dictionary):
	return [(dictionary.id2token[w], f) for w, f in bow]

def document_topics_matrix(tokens, dictionary):
	return [lda.get_document_topics(bow) for bow in tokens_to_bow(tokens, dictionary)]

def topics_sparse_to_full(topics, number_of_topics):
	topics_full = [0] * number_of_topics
	for topic, score in topics:
		topics_full[topic] = score
	return topics_full

def dtm_sparse_to_full(dtm, number_of_topics):
	return [topics_sparse_to_full(document_topics, number_of_topics) 
		for document_topics in dtm]

def topic_weights(dtm, number_of_topics):
	dtm = dtm_sparse_to_full(dtm, number_of_topics)
	return [sum([row[k] for row in dtm])/len(dtm) for k in range(number_of_topics)]

def documents_with_topic(dtm, topic, min_weight=0.1):
	return [i for i, topics in enumerate(dtm) 
		if topic in [t for t, w in topics if w >= min_weight]]

def topic_co_occurrence_matrix(dtm, min_weight=0.1):
	return [[t for t, w in topics if w >= min_weight] for topics in dtm]

def tcom_to_sentences(tcom):
	for tco in tcom:
		tco = ["T{}".format(t) for t in tco]
		tco.append('.')
	return "\n".join(tco)

# view

def show_corpus(corpus):
	st.dataframe(corpus)

def show_topics(lda, number_of_topics, sort_by_weight=False):
	# show top 10 keywords for each topic
	topics_df = pd.DataFrame([[" ".join([tw[0] for tw in lda.show_topic(t, 10)])] 
		for t in range(number_of_topics)], columns=['keywords'])
	# compute topic frequencies (weights)
	dtm = document_topics_matrix(tokens, dictionary)
	# insert topic weights before keywords
	topics_df.insert(0, 'weight', topic_weights(dtm, number_of_topics))
	if sort_by_weight:
		topics_df = topics_df.sort_values(by='weight', ascending=False)
	st.table(topics_df)

def show_topic_word_cloud(lda, topic):
	# show word cloud for a topic
	words = lda.show_topic(topic, 100)
	words = {w: f for w, f in words}
	wc = wordcloud.WordCloud(background_color="white", max_words=100)
	wc.generate_from_frequencies(words)
	st.image(wc.to_array())
	st.write(f"Word cloud for topic {topic}")

def show_document_topics_matrix(dtm, number_of_topics, corpus, selected_topic=None):
	dtm_df = pd.DataFrame(dtm_sparse_to_full(dtm, number_of_topics))
	dtm_df['content'] = corpus['content']
	# If a topic is selected, show sort dataframe by topic weight
	if selected_topic is not None:
		dtm_df = dtm_df.sort_values(by=selected_topic, ascending=False)
		# Highlight selected topic
		dtm_df = dtm_df.style.apply(lambda x: ['background-color: lightblue' 
			if x.name == selected_topic else '' for i in x], axis=0)
	st.dataframe(dtm_df, height=300)

def show_topic_co_occurrences(tcom):
	# Create dataframe with one column per document
	# Each row contains a list of topics
	tcom_df = pd.DataFrame([[tco] for tco in tcom])
	tcom_df.columns = ['topics']
	st.dataframe(tcom_df, width=200)

def show_topic_trends(dtm, number_of_topics, corpus):
	# Create dataframe with one column per topic
	# Each row contains a list of topic weights
	trends_df = pd.DataFrame(dtm_sparse_to_full(dtm, number_of_topics))
	if 'year' in corpus.columns:
		trends_df['year'] = corpus['year']
		# drop rows with missing year
		trends_df = trends_df.dropna(subset=['year'])
		# convert year to integer (if it is float)
		trends_df['year'] = trends_df['year'].astype(int)
		trends_df = trends_df.groupby('year').sum()
		st.bar_chart(trends_df)
	else:
		st.write("No year column")

# main

st.sidebar.title("TME")

corpus_file = st.sidebar.file_uploader("Corpus", type="csv",
	on_change=lambda: st.session_state.clear())
user_stopwords = st.sidebar.text_area("Stopwords (one per line)",
	on_change=lambda: st.session_state.clear())

st.header("Corpus")
if corpus_file is not None:
	corpus = load_corpus(corpus_file)
	show_corpus(corpus)

	# Remove hyphens from the text
	# This is just an example of the kind of preprocessing you may want to do
	corpus['content'] = corpus['content'].apply(remove_hyphens)

	st.header("Preprocessed corpus")
	tokens = corpus_to_tokens(corpus, user_stopwords)
	st.write(pd.DataFrame(tokens))

	st.header("Topics")
	number_of_topics = st.sidebar.slider("Number of topics", min_value=1, max_value=50, value=10)
	sort_by_weight = st.sidebar.checkbox("Sort topics by weight", value=False)
	dictionary = Dictionary(tokens)
	lda = fit_lda(tokens, number_of_topics, dictionary)
	show_topics(lda, number_of_topics, sort_by_weight)

	st.header("Topic world cloud")
	selected_topic = st.sidebar.number_input("Choose topic for word cloud and document-topics matrix", 0, number_of_topics-1)
	show_topic_word_cloud(lda, selected_topic)

	st.header("Document-topics matrix")
	dtm = document_topics_matrix(tokens, dictionary)
	show_document_topics_matrix(dtm, number_of_topics, corpus, selected_topic)

	st.header("Topic co-occurrences")	
	tcom = topic_co_occurrence_matrix(dtm, 0.1)
	show_topic_co_occurrences(tcom)

	st.header("Topic trends")
	show_topic_trends(dtm, number_of_topics, corpus)

	# tcom_to_sentences(tcom)
else:
	st.markdown("Please upload a corpus. The csv file should contain at least a 'name' and a 'content' column.")