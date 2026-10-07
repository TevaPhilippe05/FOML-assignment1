from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        if pooling == "max":
            self.pool = nn.MaxPool2d(2)
        elif pooling == "average":
            self.pool = nn.AvgPool2d(2)

    def forward(self, X):
        X = self.conv(X)
        X = self.relu(X)
        X = self.pool(X)
        return X

class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling):
        super().__init__()
        self.block1 = Conv(in_channels, out_channels, kernel_size, pooling)
        self.block2 = Conv(out_channels, out_channels * 2, kernel_size, pooling)

    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        return X

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64,), num_classes=100, convolution = None, activation_function="relu", regularization=None, reg_param=0.0):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.regularization = regularization
        self.reg_param = reg_param

        if convolution is not None:
            self.features = convolution
        else:
            self.features = nn.Identity()
            
        self.flatten = nn.Flatten()

        self.lst = nn.ModuleList()
        
        self.lst.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.lst.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.lst.append(nn.Linear(hidden_sizes[-1], num_classes))

        if activation_function == "relu":
            self.activation_function = nn.ReLU()
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
        elif activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()

        if regularization == "dropout":
            self.dropouts = nn.ModuleList()
            for i in range(len(hidden_sizes)):
                self.dropouts.append(nn.Dropout(reg_param))

    def forward(self, X):
        X = self.features(X)
        X = self.flatten(X)
        X = self.lst[0](X)
        for i in range(1, len(self.hidden_sizes) + 1):
            X = self.activation_function(X)
            if self.regularization == "dropout":
                X = self.dropouts[i - 1](X)
            X = self.lst[i](X)
        return X