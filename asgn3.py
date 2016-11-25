from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import load_map as nn
import numpy as np
from numpy import linalg
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure

STEMMER = PorterStemmer()
eliminate_words = ['a'] #should be determined after discussion
test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
def get_word_list(eliminate_w,test_w):

    word_id_list = []
    for i in nn.word2wid.values():
        if nn.wid2word[i] in test_w or nn.wid2word[i] in eliminate_w:
            pass
        else:
            word_id_list.append(i)
    return word_id_list
word_id_list = get_word_list(eliminate_words,test_words);

# def read_o_counts(filename):
#
#   o_counts = {} # Occurence counts
#   fp = open(filename)
#   N = float(fp.next())
#   for line in fp:
#     line = line.strip().split("\t")
#     wid0 = int(line[0])
#     o_counts[wid0] = int(line[1])
#
#   return (o_counts)

# def make_word_list(eliminate_wordss):
#     list2 = []
#     op = read_o_counts("counts")
#     for o in op.values():
#         list2.append(o)
#     list2 = sorted(list2,reverse=True);
#     return list2
#
# list1 = make_word_list(eliminate_words)

# for o in nn.word2wid.values():
#     for i in eliminate_words:
#         if nn.wid2word[o] == i:
#             pass
#         else:
#             list1.append(o)

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[nn.word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a 
  username (starts with @).  If so, returns the word unchanged.

  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word

  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
    '''Compute the pointwise mutual information using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value

    '''
    # prob_xy = c_xy/N
    # prob_x = c_x/N
    # prob_y = c_y/N
    # the_prob = prob_xy/(prob_x * prob_y)
    # lognum = log(the_prob,2)
    return log((c_xy*N)/(c_x*c_y),2); # replace this
# Do a simple error check using value computed by hand

if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print "Warning: PMI is incorrectly defined"
else:
    print "PMI check passed"

def get_vector(v0,v1):
  vector0 = []
  np.array(vector0)
  vector1 = []
  np.array(vector1)
  for i in range(0,len(word_id_list)-1):
      if i in v0.keys() and i in v1.keys():
          vector0.append(v0[i])
          vector1.append(v1[i])
      elif i in v0.keys() and i not in v1.keys():
          vector0.append(v0[i])
          vector1.append(0)
      elif i in v1.keys() and i not in v0.keys():
          vector1.append(v1[i])
          vector0.append(0)
  return vector0,vector1

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.

  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  #You will need to replace with the real function\
  # vector0 = []
  # np.array(vector0)
  # vector1 = []
  # np.array(vector1)
  # for i in range(0,len(word_id_list)-1):
  #     if i in v0.keys() and i in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(v1[i])
  #     elif i in v0.keys() and i not in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(0)
  #     elif i in v1.keys() and i not in v0.keys():
  #         vector1.append(v1[i])
  #         vector0.append(0)
  vector0,vector1 = get_vector(v0,v1);
  print 'changud',len(vector0),len(vector1)

  # for i in range(0,len(word_id_list)-1):
  #     if i in v1.keys():
  #         vector1.append(v1[i])
  #     else:
  #         vector1.append(0)
  print "Warning: cos_sim is incorrectly defined"
  print 'haha',vector0[1:1000]
  cos_result = (np.dot(vector0,vector1))/(linalg.norm(vector0)*linalg.norm(vector1))
  cos_result = round(cos_result,2)
  print cos_result
  # print vector0[0:1000]
  return cos_result

def Jaccard(v0,v1):
  # vector0 = []
  # np.array(vector0)
  # vector1 = []
  # np.array(vector1)
  # for i in range(0,len(word_id_list)-1):
  #     if i in v0.keys() and i in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(v1[i])
  #     elif i in v0.keys() and i not in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(0)
  #     elif i in v1.keys() and i not in v0.keys():
  #         vector1.append(v1[i])
  #         vector0.append(0)
  vector0,vector1 = get_vector(v0,v1);
  print 'changud',len(vector0),len(vector1)
  minn = 0
  maxx = 0
  for i in range(0,len(vector0)-1):
      minn += min(vector0[i],vector1[i]);
      maxx += max(vector0[i],vector1[i]);
  sim_Jaccard = minn/maxx
  # print vector0[0:1000]
  sim_Jaccard = round(sim_Jaccard,2)
  print 'haha',sim_Jaccard
  return sim_Jaccard

def Dice(v0,v1):
  # vector0 = []
  # np.array(vector0)
  # vector1 = []
  # np.array(vector1)
  # for i in range(0,len(word_id_list)-1):
  #     if i in v0.keys() and i in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(v1[i])
  #     elif i in v0.keys() and i not in v1.keys():
  #         vector0.append(v0[i])
  #         vector1.append(0)
  #     elif i in v1.keys() and i not in v0.keys():
  #         vector1.append(v1[i])
  #         vector0.append(0)
  vector0,vector1 = get_vector(v0,v1);
  print 'changud',len(vector0),len(vector1)
  minn = 0
  bot_sum = 0
  for i in range(0,len(vector0)-1):
      minn += min(vector0[i],vector1[i]);
      bot_sum += (vector0[i],vector1[i]);

  Dice_value = (2*minn)/bot_sum;
  Dice_value = round(Dice_value,2)
  return Dice_value

def JS(v0,v1):
    vector0,vector1 = get_vector(v0,v1)
    js_v1 = 0
    js_v2 = 0
    for i in range(0,len(vector0)-1):
        js_v1 += (vector0[i] * log(vector0[i]/((vector0[i]+vector1[i])/2)),2)
        js_v2 += (vector1[i] * log(vector1[i]/((vector0[i]+vector1[i])/2)),2)
    JS_value = js_v1+js_v2
    JS_value = round(JS_value,2)
    return JS_value

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for all words, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        dictt = {}
        for i in range(0,len(word_id_list)-1):
            if word_id_list[i] in co_counts[wid0].keys():
                LLLog = log((tot_count*co_counts[wid0][word_id_list[i]]/o_counts[word_id_list[i]]*o_counts[wid0]),2);
                if LLLog>0:
                    dictt[i] = LLLog;
        # dictt[i] = co_counts[wid0][list1[i]]
        ##you will need to change this
        vectors[wid0] = dictt
    print "Warning: create_ppmi_vectors is incorrectly defined"
    return vectors

def create_ppmi_Laplace_vectors(wids, o_counts, co_counts, tot_count,add_value):
    '''Creates context vectors for all words, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        dictt = {}
        for i in range(0,len(word_id_list)-1):
            if word_id_list[i] in co_counts[wid0].keys():
                LLLog = log(((tot_count+add_value* len(word_id_list) *len(wids)))*(co_counts[wid0][word_id_list[i]]+add_value)/((o_counts[word_id_list[i]]+add_value*len(wids))*(o_counts[wid0]+add_value* len(word_id_list))),2);
                if LLLog>0:
                    dictt[i] = LLLog;
        # dictt[i] = co_counts[wid0][list1[i]]
        ##you will need to change this
        vectors[wid0] = dictt
    print "Warning: create_ppmi_vectors is incorrectly defined"
    return vectors

def create_probability_vectors(wids,o_counts,co_counts,tot_count):
    vectors = {}

    for wid0 in wids:
        dictt = {}
        for i in range(0,len(word_id_list)-1):
            if word_id_list[i] in co_counts[wid0].keys():
                joint_prob = (tot_count*co_counts[wid0][word_id_list[i]])/ (o_counts[word_id_list[i]]*o_counts[wid0]);
                if joint_prob!=0:
                    dictt[i] = joint_prob;
    return vectors

def create_t_test_vector(wids,o_counts,co_counts,tot_count):
    vectors = {}
    for wid0 in wids:
        dictt = {}
        for i in range(0,len(word_id_list)-1):
            if word_id_list[i] in co_counts[wid0].keys():
                if co_counts[wid0][word_id_list[i]] * tot_count > o_counts[word_id_list[i]]*o_counts[wid0]:
                    dictt[i] = (co_counts[wid0][word_id_list[i]] / tot_count - o_counts[word_id_list[i]]*o_counts[wid0] / tot_count**2) / ((o_counts[word_id_list[i]]*o_counts[wid0]) / tot_count**2)**0.5
    vectors[wid0] = dictt
    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.

  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(fp.next())
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)



def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.

  :type similarities: dict 
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (nn.wid2word[pair[0]], nn.wid2word[pair[1]])
    print "%0.2f\t%-30s\t%d\t%d" % (similarities[pair],word_pair,o_counts[pair[0]],o_counts[pair[1]])


def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.

  :type items: list
  :param items: the list to pair up
  :return: list of pairs

  '''

  return [(x, y) for x in items for y in items if x < y]


test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([nn.word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs 
wid_pairs = make_pairs(all_wids)

#read in the count information

(o_counts, co_counts, N) = read_counts("counts", all_wids)

##############################################

#make the word vectors

vectors = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
# vectors = create_ppmi_Laplace_vectors(all_wids, o_counts, co_counts, N,2)
# vectors = create_t_test_vector(all_wids,o_counts,co_counts,N)

##################################

# compute similarity:

c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
# c_Jaccard = {(wid0,wid1): Jaccard(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
# c_Dice = {(wid0,wid1): Dice(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

# vectors = create_probability_vectors(all_wids,o_counts,co_counts,N)
# c_JS = {(wid0,wid1): JS(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

#######################################

print 'print the similarity'

print_sorted_pairs(c_sims, o_counts)
# print_sorted_pairs(c_Jaccard, o_counts)
# print_sorted_pairs(c_Dice, o_counts)
# print_sorted_pairs(c_JS, o_counts)


##########may be we need PCA###################

def princomp(A): #Principle component analysis, use for dimensionality reducion

 M = (A-mean(A.T,axis=1)).T
 [latent,coeff] = linalg.eig(cov(M))
 score = dot(coeff.T,M)
 return score

####################example####################

# A = array([1,2,3,4,5],[2,3,4,5,6],[4,3,2,5,67],[213,4,23,34,])
#
# a = princomp(A)
# print a






























































































