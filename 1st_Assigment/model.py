import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,       
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 509) 
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 1)    
        
        self.leaky_relu = nn.LeakyReLU()

        if weight_norm:
            nn.utils.weight_norm(self.fc1)
            nn.utils.weight_norm(self.fc2)
            nn.utils.weight_norm(self.fc3)
            nn.utils.weight_norm(self.fc4)
            nn.utils.weight_norm(self.fc5)
            nn.utils.weight_norm(self.fc6)
            nn.utils.weight_norm(self.fc7)

        
        self.dropout_p = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
    
        # Final tanh activation
        self.use_tanh = use_tanh
        self.th = nn.Tanh() if use_tanh else nn.Identity() 
        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):
        
        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        x = input
        #1st layer
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        #2nd layer
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

         #3rd layer
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

         #4th layer
        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        #5th layer
        x = self.fc5(torch.cat([x,input],dim=1))
        x = self.leaky_relu(x)
        x = self.dropout(x)

        #6th layer
        x = self.fc6(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        #7th layer
        x = self.fc7(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        #8th layer and tahn activation
        x = self.fc8(x)
        x = self.th(x)
        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
