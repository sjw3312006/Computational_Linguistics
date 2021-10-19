import os, sys, optparse
import tqdm
import pymagnitude
from pymagnitude import converter
import six
from copy import deepcopy
import numpy as np

class LexSub:

    def __init__(self, wvec_file, ontology_file, topn=10):
        if os.getcwd().split('/')[-1] == 'answer':
            self.path = os.path.dirname(os.getcwd())
        else:
            self.path = os.getcwd()
            
        self.vocabulary = set()
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

        self.Beta = {} # hyperparameter matrix key: (key1, key2); value: number of times the pair appears in ontology
        self.wvecs_dic = {}
        self.Q = {}
        for key, vector in self.wvecs:
            self.vocabulary.add(key)
            self.Q[key] = vector

        # store text data in ontolgy file
        self.ontology_dic = {}
        for line in open(ontology_file, 'r'):
            words = line.lower().strip().split()
            target = words[0]
            contexts = words[1:]
            
            if words[0] not in self.ontology_dic:
                self.ontology_dic[words[0]] = [word for word in words[1:]]
            else:
                self.ontology_dic[words[0]] = self.ontology_dic[words[0]] + [word for word in words[1:]]


    def find_qjs(self, target):
        """
        given target word, find_qjs returns list of key and vector pair
        of contexts words appeared in the ontology
        """
        if target in self.ontology_dic:
            contexts = self.ontology_dic[target]
            return contexts
        else:
            return None
        

    def update_qi(self):
        a = 20
        B = 1
        for index, target in enumerate(self.vocabulary): 
            qjs = self.find_qjs(target) 
            if qjs == None:
                continue
            else:     
                new_qi = np.zeros(self.Q[target].shape)
                count = 0
                for context in qjs: 
                    if context in self.Q:
                        new_qi += B * self.Q[context] 
                        count += 1
                new_qi = (new_qi + a * self.Q[target]) / (count + a)
                self.Q[target] = new_qi


    def create_retrofit_txt(self):
        text_file_name = os.path.join(self.path, 'data', 'glove.6B.100d.retrofit.txt')
        with open(text_file_name, 'w') as f:
            for key, vector in self.Q.items():
                line = key
                for num in vector:
                    line = line + " " + str(num)
                line = line + '\n'
                f.write(line)
        destination_file_name = os.path.join(self.path, 'data', 'glove.6B.100d.retrofit.magnitude')
        converter.convert(text_file_name, destination_file_name)

        return destination_file_name


    def retrofit(self):
        # create set of vocabularies and assign 
        for t in range(self.topn):
            self.update_qi()
        Q_pymag_dest = self.create_retrofit_txt()
        self.Q_pymag = pymagnitude.Magnitude(Q_pymag_dest)
        
        
    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.Q_pymag.most_similar(sentence[index], topn=self.topn))))


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="input file with target word in context")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, "data/lexicons/ppdb-xl.txt", int(opts.topn))
    lexsub.retrofit()

    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split())))
