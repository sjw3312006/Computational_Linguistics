import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import re # added module

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        segmentation = [ w for w in text ] # segment each char into a word
        return segmentation

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)


class iterative_segmenter:
    """ Data Structure of the iterative_segmenter
    input: The input sequence of characters
    chart: The dynamic programming table to store the argmax for 
           every prefix of input indexed by character position in input
    Entry: Each entry in the chart has four components: (word, start-position, log-probability, back-pointer)
           the back-pointer in each entry links to a previous entry that it extends
    heap:  A list or priority queue containing the entries to be expanded, sorted on start-position or log-probability
    """
    def __init__(self, Bi):
        self.Pw  = Bi  # bigram probabiltiy distribution
        self.heap = [] # choosing heap to be list instead of priority queue 
        
    def find_keys_matching_string(self, phrase):
        """phrase is a string which will be searched""" 
        print("phrase:", phrase)
        keys_matching_string = []
        LEN_LIMIT = 4
        for i in range(len(phrase)):   
            if self.Pw.unigram(phrase[0:i+1])[0]: # there is a matching word in Bigram!
                # print("phrase[0:i+1]", phrase[0:i+1])
                keys_matching_string.append(phrase[0:i+1]) # all permutation of input
            elif i <= LEN_LIMIT: # allow some combination of unseen to be initialized to the heap
                keys_matching_string.append(phrase[0:i+1])
            else:
                break
        # print("received target:", target)
        # for key, count in self.data: # from the count1w.txt, find all the word that starts with the c0
        #     if key[0] == target:
        #         keys_starting_with_target.append(key)
        # if not keys_starting_with_target: # no word starting with target character
        #     print("find_key_starting_with_target:: no word starting with target character\n")
        return keys_matching_string

    def initialize_heap(self, input):
        # for i in range(len(input)+1):
        #     print(input[0:i])
        c0 = input[0] # get the first character of the input
        starting_words = self.find_keys_matching_string(input)
        print("starting_words:", starting_words)
        for word in starting_words:
            self.heap.append( (word, 0, log10(self.Pw(("<S>", word))[1]), None) )
        

        # print("words initialized to heap:\n", self.heap)
        # print("\n\n")

    def retrieve_from_nested_entry(self, nested_entry):
        "Assume nested_entry has same structure as Entry"
        if nested_entry[3]: # back_pointer not None 
            # print("back_pointer not none! word:", nested_entry[0])
            receive = self.retrieve_from_nested_entry(nested_entry[3])
            # print("got this from deeper layer:", receive)
            # print(receive + [nested_entry[0]])
            return (receive + [nested_entry[0]])
        else: # back_pointer None
            # print("back_pointer none! word:", [nested_entry[0]])
            return [nested_entry[0]]

    def segment(self, input):
        print(input)
        ## Initilize the heap ##
        self.initialize_heap(input)
        ## Iteratively fill in chart[i] for all i ##
        self.chart = [None] * len(input) # initialize the chart
        iterator = 0
        while self.heap: # using fact the empty sequences are false
            entry = self.heap.pop()
            endindex = entry[1] + len(entry[0]) - 1
            # print("entry: ", entry, "\n")
            if endindex > len(input)-1:
                continue # if the word goes out of input length
            if self.chart[endindex] is not None: # there is preventry at endindex
                # print("self.char[endindex] is not None")
                if entry[2] > self.chart[endindex][2]: # entry has a higher probability than preventry
                    self.chart[endindex] = entry
                    # print("CHART ENTRY SWITCH\n")
                    # print("chart:", self.chart, "\n")
                else: # entry has equal or lower probability than preventry
                    continue # we have already found a good segmentation until endindex
            else:
                self.chart[endindex] = entry
                # print("chart:", self.chart, "\n")

            # find newword which matches the input starting  at position endindex+1
            target_index = endindex+1
            if target_index > len(input)-1: # if target_index is output input range, skip!
                continue
            new_words = self.find_keys_matching_string(entry[0], input[target_index:]) 
            if new_words: # not empty
                # print("words to be added to the heap:", new_words, "\n")
                for newword in new_words:
                    new_entry = (newword, target_index, entry[2]+log10(self.Pw((entry[0], newword))[1]), entry)
                    if new_entry not in self.heap:
                        self.heap.append(new_entry)
            else: # empty, treat the unseen character as a word of length one
                new_entry = (input[target_index:target_index+1], target_index, entry[2]+log10(self.Pw((entry[0], input[target_index:target_index+1]))[1]), entry)
                self.heap.append(new_entry)
            # if iterator == 4:
            #     sys.exit()
            # else:
            #     iterator += 1
                # print("end of iteration\n\n\n")
            # sys.exit()
        ## Get the best segmentation ##
        final_index = len(input)-1
        retrieving_entry = self.chart[final_index]
        ret = self.retrieve_from_nested_entry(retrieving_entry)
        # print("ret:", ret, "\n\n")
    
        # sys.exit()
        return ret
        

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)


#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class uni_Pdist(dict): # for unigram
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None, V=None):
        self.V=0
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
            self.V += 1
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./(100*N**len(k)))
    def __call__(self, key):
        if key in self: return (True, self[key]/self.N)  ## on call this unigram probability is calculated
        else: return (False, self.missingfn(key, self.N))

class bi_Pdist(dict): # for bigram
    "A probability distribution estimated from counts in datafile."
    def __init__(self, unigram={}, data=[], N=None, missingfn=None,):
        number_of_empty_token = 0
        for key1, key2, count in data:
            self[(key1, key2)] = self.get((key1,key2), 0) + int(count)
            # print(key1)
            if key1 == "<S>" or key2 == "<S>":
                number_of_empty_token += 1
                # print(number_of_empty_token)
            # print(self[key])
        # self.u_data = u_data
        self.unigram = unigram
        self.number_of_empty_token = number_of_empty_token
        print(self.number_of_empty_token)
        # for key,value in self.u_data:
        #     print(key, value)
        # print(type(self.u_data.values()))
        # self.N = float(N or sum(int(self.u_data.values())))
        self.missingfn = missingfn or (lambda k, N: 1./(100*N**len(k)))
    def __call__(self, key):
        # if key in self: return (True, self[key]/self.u_data(key[0]))  ## on call this unigram probability is calculated
        # else: return (False, self.missingfn(key, self.N))
        if key[0] == "<S>":
            prior_count = self.number_of_empty_token
            # print("key[0]==<S>, prior_count:", prior_count)
        else:
            prior_count = self.unigram[key]
        return  (True, (self[key]+1)/((prior_count) + self.unigram.V+1) ) # apply add-1 smoothing which takes care of all cases


def datafile1(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def datafile2(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            (key1, key2) = key.split(" ")
            yield (key1, key2, value)
            



if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    # Pw= Pdist(data=datafile(opts.counts1w))
    Uni = uni_Pdist(data=datafile1(opts.counts1w))

    ###
    # print(count(Uni.values()))
    ###

    Bi = bi_Pdist(Uni, data=datafile2(opts.counts2w))
    segmenter = iterative_segmenter(Bi) # segmenter = Segment(Pw)
    # segmenter = Segment(Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
