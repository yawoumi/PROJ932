import pandas as pd
from sklearn.model_selection import train_test_split

# path to the place where data are stored
path = "./data/{}"

# Read the data
data1 = pd.read_csv(path.format("test.csv"), 
                    sep=",", 
                    names=["target", "id", "date", "flag", "user", "text"])

data2 = pd.read_csv(path.format("training.csv"), 
                    sep=",", 
                    names=["target", "id", "date", "flag", "user", "text"], encoding="latin-1")

data1 = data1[["text", "target"]]
data2 = data2[["text", "target"]]

# Splitting 

def splitter(d, per):
    return train_test_split(d, test_size=per)

def data_rep(train_d, test_d, nb_source=3):
    r = []
    x1, x2 = train_d, test_d
    while nb_source != 1:
        x1, y1 = splitter(x1, 1/nb_source)
        x2, y2 = splitter(x2, 1/nb_source)
        y11, y12 = splitter(y1, 0.5)
        r.append((y11, y12, y2))
        nb_source -= 1
    
    if nb_source == 1:
        x11, x12 = splitter(x1, 0.5)
        r.append((x11, x12, x2))
    
    return r

data = data_rep(data2, data1, 10)

for i in range(len(data)):
    data[i][0].to_csv("./data/sources/source_{}_entr.csv".format(i))
    data[i][1].to_csv("./data/sources/source_{}_val.csv".format(i))
    data[i][2].to_csv("./data/sources/source_{}_test.csv".format(i))

