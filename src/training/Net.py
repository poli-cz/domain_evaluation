
  
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules 
import Database






'''		
Class: Net
pytorch definition of neural network structure

'''       
class Net(nn.Module):

    # Network structure definition
    def __init__(self):         
        super().__init__()
        self.fc1 = nn.Linear(13, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)


    # Data flow definition
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) # For binarz 



'''		
Parameters
----------
dataset : list
		  
'''          
def train(dataset, validate=None):
    loss_val = list()
    counter=0
    sum=0
    batch_sum=0

    checkpoint_position = 500

    for line in dataset:
        #print(line)
       # input()
        label = float(line["label"])

        dat = torch.FloatTensor(line["data"])
        output=net(dat)

        if counter % checkpoint_position == 0:

            print("Loss:", round(sum/checkpoint_position, 3),"Progress:",  round((counter/len(dataset))*100, 3), "%")
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





if __name__ == "__main__":

    # Initialize network
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.000001, )
    loss_fn = nn.BCELoss()
    optimizer.zero_grad()


    # Prepare learning dataset
    d = Database.Database('domains')
    bad_data = list()
    good_data = list()


    good_collection = d.return_collection("bad_dataset")
    bad_collection = d.return_collection("good_dataset")


    for name in good_collection.find():
        good_data.append(name)


    for name in bad_collection.find():
        bad_data.append(name)

    # mix dataset
    label = None
    merged = shuffle(good_data + bad_data)

    testset = list()
    dataset = list()
    counter =0


    # split to testset and dataset
    for piece in merged:
        if counter < 5000:
            testset.append(piece)
        else:
            dataset.append(piece)

        counter+=1

    #################
    # Train network #
    #################
    epoch_count = 25
    for i in range(epoch_count):
        train(dataset)
        print("Batch:", i)
        train(testset, True)



    torch.save(net, "./model_bigram.pt")




