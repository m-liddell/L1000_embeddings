"""
3-stagenn
https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold//blob/main/final/Best%20LB/Training/3-stagenn-train.ipynb
https://www.kaggle.com/c/lish-moa/discussion/201510
"""

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.15)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x

#tfp.layers.weight_norm.WeightNorm
#tf.keras.layers.LeakyReLU
#tf.keras.layers.AlphaDropout with selu

In each stage a similar setup was used: a Adam optimizer with a learning rate of 5e-3, 
a batch size of 256, a weight decay of 1e-5, 
and a OneCycleLR scheduler with a maximum learning rate of 1e-2.