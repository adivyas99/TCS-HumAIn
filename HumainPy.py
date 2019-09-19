import collections
import pickle
import ast
from gensim.models import Word2Vec
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import re
import numpy as np
from ete3 import Tree


'''train = pd.read_csv('Train.csv')
        
i=0
def cleanhtml(raw_html):
    global i
    print(i)
    i+=1
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = cleantext.lower()
    return cleantext

train.Body = train.Body.map(cleanhtml)
train.to_csv('cleaned_train_file.csv')'''

train = pd.read_csv('cleaned_train_file.csv')
train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
train=train.iloc[:,1:]
train = train.drop_duplicates()

k=0
tokenized_sentences=[]
def clean_tokenizing_func(x):
    global k
    k+=1
    print(k)
    y = sent_tokenize(x)
    for i in y:
        i=i.lower()
        i = i.translate(str.maketrans('', '', string.punctuation))
        sen = word_tokenize(i)
        tokenized_sentences.append(sen)

train.Title.map(clean_tokenizing_func)

print('Making Model-->>\n')
### Makign embedding-->>>
# train model
model = Word2Vec(tokenized_sentences, min_count=1)

# summarize the loaded model
#print(model)

# summarize vocabulary
words = list(model.wv.vocab)
#print(words)

# access vector for one word
#print(model['sentence'])
print('saving model')
# save model
model.save('Embedding_model.bin')
vector = model['click']  
# load model
#new_model = Word2Vec.load('Embedding_model.bin')
print('saving sentences -- ')
with open('tokenized_sentences', 'wb') as f:
    pickle.dump(tokenized_sentences, f)

'''with open('tokenized_sentences', 'rb') as f:
    tokenized_sentences = pickle.load(f)'''
    
model.most_similar(all_vectors[2].T, topn=1)

k=0
tags=[]
def get_tags(x):
    x=str(x)
    global k
    k+=1
    print(k)
    a=x.split()
    tags.extend(a)

train.Tags.map(get_tags)

tags=list(set(tags))

not_in_vocab_tag = []
all_vectors = []
for i in tags :
    try:
        #w = lemmatizer(i, u"NOUN")[0]
        vector = model[i] 
        vector = vector.reshape(100,1)
        all_vectors.append(vector)
    except KeyError:
        not_in_vocab_tag.append(i)
        
'''with open('Tags', 'wb') as f:
    pickle.dump(tags, f)'''
    
tags_array=np.array(all_vectors).reshape(22608,100)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(tags_array_copy, 'single')

labelList = range(1, 22609)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
    
##another
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(tags_array)
print(cluster.labels_)
labels = cluster.labels_
np.squeeze(tags_array, axis=0).shape
tags_array_copy = tags_array[:,:2]
#print(new_model)


#another--
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(100, 70))
plt.title("Customer Dendograms")
linkages = shc.linkage(tags_array, method='ward')
dend = shc.dendrogram(linkages)
dir(dend)
dnd2 = dend.copy()
dnd2.keys()
dir(linkages)
dend.get()
linkages[0:10]
dir(dend)
idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(tags_array[:,0], tags_array[:,1])  # plot all points
plt.scatter(tags_array[idxs,0], tags_array[idxs,1], c='r')  # plot interesting points in red again
plt.show()
dir(plt)

##
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    ddata = dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

fancy_dendrogram(
    linkages,
    truncate_mode='lastp',
    p=50,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
    )
plt.figure(figsize=(100, 70))

plt.show()
from scipy.cluster.hierarchy import fcluster
max_d = 200
clusters = fcluster(linkages, max_d, criterion='distance')
clusters
collections.Counter(clusters)

k=0
first_cluster = {1:[],2:[]}
for i in range(len(clusters)):
    k+=1
    print(k)
    if clusters[i]==1:
        first_cluster[1].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    else:
        first_cluster[2].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    
first_cluster[1] = first_cluster[1][1:10]
first_cluster[2] = first_cluster[2][1:10]

(str(first_cluster[1])[1:]+str(first_cluster[1])[:-1]).replace("'","")
#unrooted_tree = Tree( "((C,D),(f,g));" )

unrooted_tree = Tree("(("+str(first_cluster[1])[1:-1].replace("'","")+"),("+ str(first_cluster[2])[1:-1].replace("'","")+"));")
print (unrooted_tree)
#-------
max_d = 166
clusters = fcluster(linkages, max_d, criterion='distance')

k=0
second_cluster = {2:[],3:[]}
for i in range(len(clusters)):
    k+=1
    print(k)
    if clusters[i]==2 and len(second_cluster[2])<11:
        second_cluster[2].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    elif clusters[i]==3 and len(second_cluster[3])<11:
        second_cluster[3].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    
second_cluster[2] = second_cluster[2][1:10]
second_cluster[3] = second_cluster[3][1:10]

(str(first_cluster[1])[1:]+str(first_cluster[1])[:-1]).replace("'","")
#unrooted_tree = Tree( "((C,D),(f,g));" )

unrooted_tree = Tree("(("+str(second_cluster[2])[1:-1].replace("'","")+"),("+ str(second_cluster[3])[1:-1].replace("'","")+"));")
print (unrooted_tree)
#-------
max_d = 160
clusters = fcluster(linkages, max_d, criterion='distance')

k=0
third_cluster = {1:[],2:[]}
for i in range(len(clusters)):
    k+=1
    print(k)
    if clusters[i]==1 and len(third_cluster[1])<11:
        third_cluster[1].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    elif clusters[i]==2 and len(third_cluster[2])<11:
        third_cluster[2].append(model.most_similar(all_vectors[i].T, topn=1)[0][0])
    
third_cluster[1] = third_cluster[1][1:10]
third_cluster[2] = third_cluster[2][1:10]

(str(first_cluster[1])[1:]+str(first_cluster[1])[:-1]).replace("'","")
#unrooted_tree = Tree( "((C,D),(f,g));" )

unrooted_tree = Tree("(("+str(third_cluster[1])[1:-1].replace("'","")+"),("+ str(third_cluster[2])[1:-1].replace("'","")+"));")
print (unrooted_tree)


'''
k=0
for i in range(len(list_of_sentences)):
    i = i.replace('\n','')
list_of_sentences_string = str(list_of_sentences)
file1 = open(r"/Users/anilvyas/Desktop/TCS HumAIn/list_of_sentences_string.txt","w") 
file1.write(list_of_sentences_string)
file1.close() '''
list_of_sentences_series = pd.Series(list_of_sentences)
list_of_sentences_series = list_of_sentences_series.drop_duplicates()
#list_of_sentences_series.to_csv('list_of_sentences_series.csv')
list_of_sentences_series = pd.read_csv('list_of_sentences_series.csv',header = None)
list_of_sentences_series = list_of_sentences_series.loc[:, ~list_of_sentences_series.columns.str.contains('^Unnamed')]
list_of_sentences_series = list_of_sentences_series.rename(columns={0: "id", 1: "Body"})
tokenized_sentences = []
k=0
def clean_tokenizing_func2(x):
    global k
    k+=1
    print(k)
    x = x.translate(str.maketrans('', '', string.punctuation))
    sen = word_tokenize(x)
    tokenized_sentences.append(sen)
    return sen

list_of_sentences_series['Body_tokens'] = list_of_sentences_series.Body.map(clean_tokenizing_func2)

#Saving string-
tokenized_sentences_string = str(tokenized_sentences)
file1 = open("tokenized_sentences_string.txt","w") 
file1.write(tokenized_sentences_string)
file1.close() 
# Getting back string-
file1 = open("Tokenized sentences.txt","r") 

all_of_it = file1.read()
list_str = file1.readlines()
tokenized_sentences_test = ast.literal_eval(list_str)
with open('Tokenized sentences.txt', 'r') as file:
    data = file.read().replace('\n', '')
    
    
    
import pydot


def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.iteritems():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, k+'_'+v)

graph = pydot.Dot(graph_type='graph')
first_cluster[1]  = list(first_cluster[1])
first_cluster[2]  = list(first_cluster[2])
visit(first_cluster)
graph.write_png('example1_graph.png')
'''
with open('Tokenized sentences.txt', 'w') as f:
    for item in tokenized_sentences:
        f.write("%s\n" % item)
text_file = open("Tokenized sentences.txt", "r")
list1 = text_file.readlines()

tokenized_sentences = []

# open file and read the content in a list
with open('Tokenized sentences.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        current_place = ast.literal_eval(current_place)

        # add item to the list
        tokenized_sentences.append(current_place)
'''





