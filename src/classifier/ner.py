import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
                    
        return x


class MLPDual(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLPDual, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim[0], hidden_dim)  # Input layer to hidden layer
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # Hidden layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim*2, output_dim)  # Hidden layer to output layer

        self.fc4 = torch.nn.Linear(input_dim[1], hidden_dim)  # Input layer to hidden layer
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)  # Hidden layer to hidden layer
        self.fc6 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer

        self.split_position = input_dim[0]
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).to(dtype=self.fc1.weight.dtype)
        att, mlp = x[:,:self.split_position], x[:,self.split_position:]
        x = torch.relu(self.fc1(att))
        x = torch.relu(self.fc2(x))

        y = torch.relu(self.fc4(mlp))
        y = torch.relu(self.fc5(y))
        out = self.fc3(torch.cat((x,y),dim=-1))
        return out


class MLPexp05(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, cuda=False):
        super(MLPexp05, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
                    
        return x
