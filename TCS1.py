##TCS HumAIn--

import pandas as pd
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


train.columns
train.Tags.head(10)
train.Tags[0].strip()#ewmoving \n
#removing tags of html

c=0
key = []
def unique_key(x):
    global c
    global key
    print(c)
    c+=1
    y = x.split()
    key = key+y
    key = list(set(key))
    #key.append(y)
    
train['Tags'].map(unique_key)

