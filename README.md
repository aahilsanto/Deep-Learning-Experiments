# Deep-Learning-Experiments

## Exp 1:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
        
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

def train_model(ai_brain,X_train,y_train,criterion,optimizer,epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()
        
        ai_brain.history['loss'].append(loss.item())
        if epoch%200==0:
            print(f"Epoch {epoch}/{epochs} = loss: {loss.item():.6f}")
    
    
```

## Exp 2:

```python
class PeopleClassifier(nn.Module):
    def __init__(self,input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        

def train_model(model,train_loader,criterion, optimizer,epochs=200):
    model.train()
    for epoch in range(epochs):
        for inputs,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
        if (epoch+1)%10==0:
            print(f"Epoch [{epoch+1}/{epochs}], loss: {loss.item()}")
            
            
model=PeopleClassifier(input_size=X_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
train_model(model,train_loader,criterion,optimizer,epochs=200)    
```

## Exp 3:

```python
class CNNClassifier():
    def __init__(self):
        super(CNNClassifier,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model,train_loader,epochs=3):
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        for images,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,inputs)
            loss.backward()
            optimiser.step()
            running_loss+=loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```
