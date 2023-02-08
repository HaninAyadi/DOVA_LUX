# Folder Overview

## data_collection.ipynb
Collection of posts (without the comments) from the different subreddits using the Pushshift API

## duplicates_deletion.ipynb 
Deletion of the posts with duplicate content

## personal_long_covid_classifier_training.ipynb (run in Google Colbaoratory environment)
Training of BERTweet model using manually labeled text

## personal_long_covid_classifier_application.ipynb (run in Google Colbaoratory environment)
Application of the classifier on the collected Reddit posts

## reddit_posts_preprocessing.ipynb
Text preprocessing applied on the remaining Reddit posts after duplicates deletion and personal/non-personal classfier application

## lexicon_symptoms_categories_mapping.ipynb
Mapping the different synonyms of symptoms to their original symptom name and the different symptoms to their category

## initial_insights_from_reddit_posts.ipynb
Statistical insights on the symptoms / categories of symptoms found in our input data

## kmeans_clustering.ipynb
Application of kmeans clustering on the preprocessed Reddit posts