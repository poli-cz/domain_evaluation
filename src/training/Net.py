
  
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import Database

class Net(nn.Module):
    def __init__(self):         # only structure of network
        super().__init__()
        self.fc1 = nn.Linear(14, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


helper = torch.load('./good_nets/net_0.149_err.pt')
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.000001, )
loss_fn = nn.BCELoss()
optimizer.zero_grad()



d = Database.Database('domains')
bad_data = list()
good_data = list()


good_collection = d.return_collection("bad_dataset")
bad_collection = d.return_collection("good_dataset")


for name in good_collection.find():
    good_data.append(name)


for name in bad_collection.find():
    bad_data.append(name)


print(len(bad_data))
print(len(good_data))



 

label = None
dats = shuffle(good_data + bad_data)

testset = list()
dataset = list()
counter =0

for piece in dats:
    if counter < 50:
        testset.append(piece)
    else:
        dataset.append(piece)

    counter+=1







def train(dataset, validate=None):
    loss_val = list()
    counter=0
    sum=0
    batch_sum=0

    for line in dataset:
        label = float(line["label"])

        helper_output=helper(torch.FloatTensor(line["data"]))

        evaluate = float(helper_output)

        if evaluate > 0.5:
            line["data"] = line["data"] + [20]
        else:
            line["data"] = line["data"] + [-20]
        
        dat = torch.FloatTensor(line["data"])
        output=net(dat)
        #print(dat, output, label)
        #input()


        if counter % 500 == 0:

            print("Loss:", round(sum/500, 3),"Progress:",  round((counter/len(dataset))*100, 3), "%")
            sum=0

        counter+=1

        loss = loss_fn(output, torch.FloatTensor([label]))
        sum+=float(loss)
        batch_sum+=float(loss)

        
        if validate is not None:
            #if counter % 2 == 0:
                #print("output:", float(loss.float()), "label:", label)
            loss_val.append(float(loss.float()))


        if validate is None:
            loss.backward()
            optimizer.step()
            

    if validate is not None:
        print("--------------------------------------------------")
        print("Validate Batch loss:", float(batch_sum)/float(len(dataset)))
        print("--------------------------------------------------")
    else:
        print("--------------------------------------------------")
        print("Batch loss:", float(batch_sum)/float(len(dataset)))
        print("--------------------------------------------------")


for i in range(100):
    train(dataset)
    print("Batch:", i)
    train(testset, True)

    input()


#torch.save(net, "./model_bigram.pt")




