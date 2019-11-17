import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 



def create_model( num_classes  = 3):
    model = models.resnet34(pretrained = True)
    
    n_input_features = model.fc.in_features 
    
    
    # freeze model params 
    for p in model.parameters():
        p.requires_grad =False



    layers = [nn.Linear(n_input_features,256),
              nn.ReLU(True),
              nn.BatchNorm1d(256),

              nn.Linear(256,128),
              nn.ReLU(True),
              nn.BatchNorm1d(128),
              nn.Dropout(0.1),

              nn.Linear(64, num_classes)


             ]

    module = nn.Sequential(*layers)
    model.fc = module 



    return model




if __name__ == "__main__":
    model = create_model()
    print(model)