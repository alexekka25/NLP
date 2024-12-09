# Text Summarization Project 

## Overview

This project implements Text Summarization using the Extractive Approach. It allows users to generate summaries of articles by dynamically selecting articles from a dataset. The project also includes powerful visualizations such as:

* Sentence Rankings (via TF-IDF scores)
* Word Cloud of key terms
The summarization is built on Python and leverages NLP techniques and machine learning models to identify the most important sentences and terms.

# Approaches to Text Summarization

1. Extractive Approach
The extractive summarization method selects and ranks sentences directly from the text based on their importance. This project uses:

* TF-IDF (Term Frequency-Inverse Document Frequency): Scores sentences based on term importance.

* TextRank (Graph-Based Ranking): Uses cosine similarity and a graph-based algorithm to rank sentences.


2. Abstractive Approach

Abstractive summarization generates new sentences to summarize the main ideas of the text. It mimics how humans summarize by paraphrasing and restructuring.

Key Characteristics:

* Uses language models to rewrite content.
* Produces summaries that are more natural and concise.
* Computationally intensive.

In this Project, I will be using the extractive approach to summarize text using Machine Learning and Python


# Features

* Dynamic Article Selection: Choose article_id to summarize specific articles.

* Sentence Rankings (TF-IDF): Visualize sentence importance.

* Word Cloud: Generate key term visualizations for each article.

* Multiple Summarization Methods: Supports TF-IDF and TextRank.

# Visualizations
1. Sentence Rankings

A bar chart showing the importance of sentences based on TF-IDF scores.

2. Word Cloud

A visual representation of the most frequent and important terms.

# Implementation with GloVe

Sentences are vectorized using GloVe embeddings.

TextRank with GloVe-based similarity ensures better semantic rankings.
