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
