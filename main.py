import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_df = pd.read_csv("/content/rice-type-classification/riceClassification.csv")
data_df.head()

data_df.dropna(inplace = True)
data_df.drop(['id'], axis = 1, inplace = True)
print(data_df.shape)

data_df.head()

print(data_df['Class'].unique())

print(data_df["Class"].value_counts())

original_df = data_df.copy()
for column in data_df.columns:
        data_df[column] = data_df[column]/data_df[column].abs().max()
data_df.head()

X = np.array(data_df.iloc[:,:-1])
Y = np.array(data_df.iloc[:,-1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.5)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

class dataset(Dataset):
   def __init__(self, X, Y):
      self.X = torch.tensor(X, dtype = torch.float32).to(device)
      self.Y = torch.tensor(Y, dtype = torch.float32).to(device)

   def __len__(self):
      return len(self.X)

   def __getitem__(self, index):
      return self.X[index], self.Y[index]

training_data = dataset(X_train, Y_train)
validation_data = dataset(X_val, Y_val)
testing_data = dataset(X_test, Y_test)

train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size = 64, shuffle = True)
testing_dataloader = DataLoader(testing_data, batch_size = 64, shuffle = True)

HIDDEN_NEURONS = 20
class Astra(nn.Module):
  def __init__(self):
    super(Astra, self).__init__()
    self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
    self.linear = nn.Linear(HIDDEN_NEURONS, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.input_layer(x)
    x = self.linear(x)
    x = self.sigmoid(x)
    return x
model = Astra().to(device)

summary(model, (X.shape [1],))

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = 0.001)

total_loss_train_plot = []
total_loss_val_plot = []
total_acc_train_plot = []
total_acc_val_plot = []

epochs = 10
for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    ## Training and Validation
    for data in train_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()
        acc = ((prediction).round() == labels).sum().item()
        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
      for data in validation_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_val += batch_loss.item()
        acc = ((prediction).round() == labels).sum().item()
        total_acc_val += acc
    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_val_plot.append(round(total_loss_val/1000, 4))

    total_acc_train_plot.append(round(total_acc_train/training_data.__len__() * 100, 4))
    total_acc_val_plot.append(round(total_acc_val/validation_data.__len__() * 100, 4))

    print(f'''
Epoch no. {epoch + 1}
Train Loss : {round(total_loss_train/1000, 4)}
Train accuracy : {round(total_acc_train/training_data.__len__() * 100, 4)}
Validation Loss : {round(total_loss_val/1000, 4)}
Validation Accuracy : {round(total_acc_val/validation_data.__len__() * 100, 4)}
''')
    print("="*25)

with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for data in testing_dataloader:
        inputs, labels = data

        prediction = model(inputs).squeeze(1)

        batch_loss_test = criterion(prediction, labels).item()
        total_loss_test += batch_loss_test
        acc = ((prediction).round() == labels).sum().item()
        total_acc_test += acc
    print(f" Accuracy {round(total_acc_test/testing_data.__len__() * 100, 4)}")

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
axs[0].plot(total_loss_train_plot, label = "Training Loss")
axs[0].plot(total_loss_val_plot, label = "Validation Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_ylim([0,2])
axs[0].legend()


axs[1].plot(total_acc_train_plot, label = "Training Accuracy")
axs[1].plot(total_acc_val_plot, label = "Validation Accuracy")
axs[1].set_title("Training and Validation Loss over Epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].set_ylim([0,100])
axs[1].legend()

plt.show()

# testing inputs for a user :
area = float(input("Area: "))/original_df['Area'].abs().max()
MajorAxisLength = float(input("Major Axis Length: "))/original_df['MajorAxisLength'].abs().max()
MinorAxisLength = float(input("Minor Axis Length: "))/original_df['MinorAxisLength'].abs().max()
Eccentricity = float(input("Eccentricity: "))/original_df['Eccentricity'].abs().max()
ConvexArea = float(input("Convex Area: "))/original_df['ConvexArea'].abs().max()
EquivDiameter = float(input("EquivDiameter: "))/original_df['EquivDiameter'].abs().max()
Extent = float(input("Extent: "))/original_df['Extent'].abs().max()
Perimeter = float(input("Perimeter: "))/original_df['Perimeter'].abs().max()
Roundness = float(input("Roundness: "))/original_df['Roundness'].abs().max()
AspectRation = float(input("AspectRation: "))/original_df['AspectRation'].abs().max()

my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]

print("="*20)
model_inputs = torch.Tensor(my_inputs).to(device)
prediction = (model(model_inputs))
print(prediction)
print("Class is: ", round(prediction.item()))