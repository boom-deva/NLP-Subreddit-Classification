# Project 3: Sub-Reddit NLP Classification (The_Donald vs. Politics)

### Contents:
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)

### Problem Statement 

Reddit is a social news aggregation, content rating, and discussion website that features over 11,400 different "sub-Reddits" or content groups. Users are able to add or interact with content called "posts" (which are titled, contain a main body of either text or media attachments, etc.) through comments and voting up/down. But what if a bug in the system or simple human error ends up corrupting sub-Reddit tags? Then the tedious responsibility of fixing it befalls someone to read, understand, and classify the posts. However, content across sub-Reddits are not necessarily disjoint. This makes it difficult to simply glance and guess which sub-Reddit a given post belongs to.

When Phaedrus famously said *"Things are not always what they seem; the first appearance deceives many; the intelligence of a few perceives what has been carefully hidden"*, surely he didn't mean artificial intelligence. Yet those words of wisdom ring true in this data-driven age - perhaps more so now than ever, which leads us to our problem: **How accurately can we use machine learning to classify similar content from two different sources?**

To answer that question, we scrape 500 posts for each of the two sub-Reddits using Reddit’s API, clean our raw data by getting rid of irrelevant content, turn our messy soup of post-level data into a processable form, then train and test out 3 different classification models (Random Forest, Logistic Regression, and Support Vector Machine) which will be evaluated based on accuracy scores, as we are only concerned about whether or not a post is correctly classified so false positives and false negatives are considered equally undesirable.

### Executive Summary

The main **Project-3.ipynb** notebook begins by looping requests to pull data using Reddit's API from /r/Politics (a nonpartisan sub-Reddit with 4,380,649 subscribers for news about general U.S. politics) and /r/The_Donald (“a never-ending rally dedicated to” Donald Trump with 690,286 subscribers). The extracted data is in the form of a dictionary with key-value pairs corresponding to different features of a post (e.g. title, main body, score, number of comments, etc.). While interesting, the numeric features (number of comments and score) do not serve as a distinguishing factor across different sub-Reddits, but partially make up a good filter to eliminate irrelevant posts ("trolls"). Thus, we focus on the *title* of the posts to conduct natural language processing on since the actual body of many of these posts consist of memes in the form of images, which unfortunately is beyond our capacity to parse at this stage.

In the data cleaning process, we get rid of duplicate posts, apply the troll filter to remove any posts that have been flagged at least once and/or posts with a net negative score (i.e. number of down votes > number of up votes). We then convert all post dictionaries into a dataframe of the title of and sub-Reddit source, but binarize the sub-Reddit source (our target variable). The dataset is then partitioned into a 70-30 train/test split before going through a pipeline consisting of the TF-IDF Vectorizer (which simultaneously tokenizes the words, and eliminates punctuation and stop words) and fit against one of 3 models:
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine

Each stage of the pipeline is optimized using a Grid Search of all major parameters of the vectorizer/model in an attempt to ensure the best fit, then compared against the Baseline Accuracy of 50.17% resulting from the case when we predict the majority class (Politics sub-Reddit) for all observations. We ignore the coefficients of the Logistic regression as interpretation is not relevant to answer our data science problem here. 


### Conclusions and Recommendations

All three models (Random Forest, Logistic Regression, Support Vector Machine) exceed the baseline accuracy of 50.17% by at least 1.5x as all models performed with an accuracy greater than 75%, demonstrating strong ability to classify at least 3 in 4 posts into the correct sub-Reddit. However, the clear winner was the **Support Vector Machine (SVM) model, which was able to correctly 4 in 5 posts**. Therefore, this is the model we would recommend Reddit administrators to implement to help them classify buggy posts that have lost their sub-Reddit tags.

Unfortunately, this model is not without limitations. While the model is good in practice, Reddit administrators do need to be aware that they still need to manually review the 1 in 5 posts that are misclassified by the SVM model. The model can likely be further fine-tuned to improve accuracy, however the nature of SVM makes the optimization process highly computationally intensive.

Another limitation of not just the SVM model but any model implemented for this purpose is that the nature of the sub-Reddits may change over time. For example, in the case of /r/Politics vs /r/The_Donald, we saw that the word "Trump" is the most frequently occurring word in both sub-Reddits. However, in a political regime where Donald Trump ceases to be relevant (either because of impeachment or the end of his term in 2020), we would expect to see a divergence in the most common words in each sub-Reddit and key differentiating factors. In today's posts, we saw that colloquialisms (e.g. "like", "just", "billions!") were words that distinguished /r/The_Donald from /r/Politics; however, in a future where Trump is no longer president, words such as the new president's name, his cabinet members' names, and words topical to his/her policies and actions would serve to differentiate /r/Politics from /r/The_Donald.

However for the remainder of the current political regime, we can expect our model to perform consistently at the same level of accuracy on new posts as it has done on our testing set here.

Further improvements for the model include:

- Addressing the limitation of using single words as our splitting criteria: to reduce variance in our model, could we consider grouping synonyms together into a single feature
- Paying more attention to capitalization: while CountVectorizer and TfidfVectorizer made everything lower case to prevent the distinction of identical words like "Wall" vs. "wall", it may be important to consider words that are *all* caps as separate features since all caps, which typically represents shouting, sends an emotional signal. Such a signal is expected to be rather common in /r/The_Donald posts, but not in posts from a more formal sub-Reddit like /r/Politics. 