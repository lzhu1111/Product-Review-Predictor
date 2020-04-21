# Product-Review-Predictor
Performed principal component analysis on 500,000+ lines of product review data and developed a lasso logistic regression model to predict the 5-star rating of Amazon electronics


## Tasks To Complete:

1. Data Cleaning
- Covert all reviewas to lower case
- Remove stop words
```
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
```
- Handle negation
- Stem all words using Porter 1979
```
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
ps = PorterStemmer()
```
2. Create a bag-of-word vector representation for each review in electronics by creating frequency count table for all word stems in all reviews and using the most frequent 500 words to define the word vector. 

3. Establish a method for measuring the distance between different reviews. Print the mutual distance between the first 100 reviews (review IDs) to screen, sorted from closest to furthest.

For this review data where language structures are complicated, we think Bag-of-Words method might not perform well, so we chose to build a Word2Vec model where we pay attention to the context of words in order to capture more of the semantics. 

4. Run a PCA and graph the first two PCs for the first 100 reviews. 

We ran a two-component PCA on the vocabulary vectors of the Word2Vec model for the first 100 reviews. Then we plot the two Primary components.
It reflects our findings that negative reviews like “disappointing”, “not_work”, “not_track” are in one cluster and positive reviews like “like”, “recommend” are in another cluster. So both methods reviews some difference in these reviews.

![Image description](https://github.com/lzhu1111/Product-Review-Predictor/blob/master/PCA%20Graph.png)


5. Perform a lasso logistic regression and measure the out-of-sample accuracy of your method of choice.

I used logistic regression to predict the rating of 5 levels of a product. We may use ordinal logistic regression because we have ordinal target variables (ratings from 1-5 means worst to best). Therefore, we can use 0, 0.25, 0.5, 0.75, 1 as our output for the prediction. It is better to use ordinal logistic regression than MNL because MNL has no intrinsic ordering. Since there is an association between the levels of the ratings (5 levels), it is better to use ordinal logistic regression.

6. Implement a method to aggregate reviews by product. Can you use any of the other columns to help with aggregation? Explain why or why not. Please clearly explain your method.

In this question, we use helpful start and helpful end, review time as well as bag of word matrix score to help with the aggregation of products. Helpful is a good metric showing whether the review is helpful or not. Review time can see whether recent reviews have more influence. Bag of words show the details of the review text. Then, we use groupby method to create a matrix of showing the average value of overall of different products.

7. Establish a method for measuring the distance between different products.

We use Euclidean distance measure to calculate the distance between different products based on the previous matrix of bag of words and products. By creating a square matrix, the numbers on the diagonal are all zero, because each country is identical to itself, and the numbers above and below are mirror images.
