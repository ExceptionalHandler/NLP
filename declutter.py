import sklearn.feature_extraction.text as ext
from nltk import corpus
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
import sys, getopt

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

def parseLogs(inputFile, outputFile):
    vectorizer = ext.CountVectorizer(tokenizer=get_tokens, stop_words='english')
    with open(inputFile) as file:
        lines = [line.rstrip() for line in file]
    lineNos = dict(zip(range(1, len(lines)), lines))
    doc_matrix = vectorizer.fit_transform(lines)

    tf_idf_transformer = ext.TfidfTransformer().fit(doc_matrix)
    sparse = tf_idf_transformer.transform(doc_matrix).toarray()

    perLineScore  = []
    for row in sparse:
        perLineScore.append(row.sum()/len(row.nonzero()[0]))

    lineScores = dict(zip(range(1, len(lines)), perLineScore))

    df = pd.DataFrame([lineNos, lineScores]).T
    df.columns = ['d{}'.format(i) for i, col in enumerate(df, 1)]
    df = df.sort_values(by=['d2'], ascending = False)

    with open(outputFile, 'w') as outFile:
        for index, row in df.iterrows():
            line = "{0:0=3d}  {1}\n"
            outFile.write(line.format(index, row['d1']))

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('try.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            outputfile = inputfile + '.sorted.txt'
    parseLogs(inputfile, outputfile)


if __name__ == "__main__":
   main(sys.argv[1:])