# Declutter log files using TF-IDF transformation
Engineers who sift through hundreds log-lines everyday to look for evidence of potential problems would know that most log-lines in a well-functioning software product are repetitive, non-informative and just noise. It is that single line of ‘timeout’ or ‘closed handle’ that sometimes carry more information than scroll-full of ‘everything is fine’.
While doing log analysis, other than looking for error and exceptions, I have always felt the need to sort log lines based on their value. Rarely-occurring informative lines at the top. Repetitive ‘everything-works’ at the bottom.
NLTK in python solves this problem with TF-IDF, which is a weight assigning technique to words or tokens based on their uniqueness or rarity. As per tfidf.com : “This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus” . 
This means that words occurring rarely in corpus carry more weight than those occurring more frequently. If we consider lines in a log file, the whole log-file as corpus, and sort the lines based on highest per-word weight, we would have sorted the log file in decreasing order of rarity. 
“Rarely-occurring informative lines at the top. Repetitive ‘everything-works’ at the bottom.”
We start with creating a CountVectorizer with a word stemmer behind it. Remember stemming will have the additional treating words from the same stem as the same word. This will make connect and connection count as same word ‘connect’. We also ignore non-English numeric tokens like timestamps by considering only alphabetic tokens. 
This is what the word-vectorizer now looks like:
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
We need to feed the lines from a document into it as if each line is a document and the whole document is the corpus:
```
with open(inputFile) as file:
    lines = [line.rstrip() for line in file]
lineNos = dict(zip(range(1, len(lines)), lines))
```
Now do the tfidf transform 
```
tf_idf_transformer = ext.TfidfTransformer().fit(doc_matrix)
sparse = tf_idf_transformer.transform(doc_matrix).toarray()
```
This produces a line_count x word_count matrix with each element containing the tf-idf weight of the word for that line. The total of all the words of that line would provide relative importance of that line. However, with smaller lines having less words, we may get less score even if they contain rarest of the rare words. We want to get lines with highest weight density. The lines with most rare words. So we divide total weight of each line with number of words in that line. This will assign densest lines the highest score.
```
perLineScore  = []
for row in sparse:
    perLineScore.append(row.sum()/len(row.nonzero()[0]))

lineScores = dict(zip(range(1, len(lines)), perLineScore))
```
lineScores the dict of line number and score. Sort it by the score and you will get densest lines at the top and similar lines will appear together. Neat.
The source code is available as declutter.py and you can run it with the log file path as an -i argument. 
```
python3 .\declutter.py -i sylog.log
```

