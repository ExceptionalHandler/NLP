# Group similar log-lines together using TF-IDF

Python script to assist log analysis by grouping together similar lines. Uses TF-IDF weight to compute similarity.
### Usage 
```
python3 .\declutter.py -i .\sylog.log
```
This will produce a sorted log file next to input file. 
 
### Blog and explaination [here](https://medium.com/@anadi.bhardwaj/declutter-log-files-using-tf-idf-transformation-358e8b61efa8).
Example syslog.log and syslog.log_sorted.txt in repo. 

The script uses a CountVectorizer with a word stemmer behind it and also ignores non-English numeric tokens like timestamps by considering only alphabetic tokens. 

This is what the word-vectorizer looks like:
```
stemmer = PorterStemmer()
def get_stemmed_tokens(tokens, stemmer):
    stemmed_tokens = []
    for token in tokens:
        if token.isalpha():
            stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def get_tokens(string):
    tokens = word_tokenize(string)
    stemmed_tokens = get_stemmed_tokens(tokens, stemmer)
    return stemmed_tokens
vectorizer = ext.CountVectorizer(tokenizer=get_tokens, stop_words='english')
```
The lines from a document are fed into it as if each line is a document and the whole document is the corpus:
```
with open(inputFile) as file:
    lines = [line.rstrip() for line in file]
lineNos = dict(zip(range(1, len(lines)), lines))
```
The tfidf transform of the resulting sparse matrix:
```
tf_idf_transformer = ext.TfidfTransformer().fit(doc_matrix)
sparse = tf_idf_transformer.transform(doc_matrix).toarray()
```
Divide total weight of each line with number of words in that line. This will assign densest lines the highest score:
```
perLineScore  = []
for row in sparse:
    perLineScore.append(row.sum()/len(row.nonzero()[0]))

lineScores = dict(zip(range(1, len(lines)), perLineScore))
```
lineScores is the dict of line-number and score. Sorting it by the score makes 'densest' lines appear at the top and similar lines appear together. 
