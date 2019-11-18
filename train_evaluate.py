import os
import torch 
from torchvision import datasets, transforms
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.optim as optim

class TrainClassifier:

    def __init__(model,
                n_epochs,
                save_path = 'model.pt',
                batch_size  = 32,
                lr = 0.0001,
                weight_decay = 0,
                num_classes = 3,
                train_data_dir = "./data/train",
                val_data_dir = "./data/valid",
                test_data_dir ="./data/test",
                ):
        
        self.model = model 

        self.n_epochs = n_epochs

        self.save_path = save_path

        self.batch_size = batch_size

        # Melanoma, nevus and seborrheic keratosis
        self.num_classes = num_classes

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir

        # Learning rate
        self.lr = lr 

        self.weigth_decay = weight_decay

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        #Define optimizer
        self.optimizer = self.optim.Adam(self.model.parameters(), lr =self.lr, weight_decay = weight_decay )

        # Check for GPU
        self.use_cuda = torch.cuda.is_available()

        # Dataloaders 
        self.loaders = self.create_dataloaders()

        # Create device. Decide if GPU or CPU training
        self.device = torch.device('cuda' if self.use_cuda  else "cpu")




    
    def create_dataloaders(self,trans_train = None, trans_val_test = None):
        """
        trans_train: torchvision.transform object
        trans_val_test: torchvision.transform object

        batch_size: int
            Training/validation/test batch size
        
        return  dictionary of dataloaders 
        {'train': train_dataloader, 'valid':val_dataloader, 'test':test_dataloader}


        """

        if trans_train is None:
            trans_train = transforms.Compose([transforms.Resize(255),
                                        transforms.RandomRotation(35),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                                    ])

        if trans_val_test is None:
            # Only resize, transform to tensor and normalize validation and test data
            trans_val_test = transforms.Compose([transforms.Resize((224,224),interpolation = Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))
                                    ])

        train_dataset = datasets.ImageFolder(self.train_dir, transform = trans_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)


        val_dataset = datasets.ImageFolder(self.val_data_dir, transform = trans_val_test)
        val_dataloader = torch.utils.DataLoader(val_dataset, batch_size = self.batch_size,shuffle = True )

        test_dataset = datasets.ImageFolder(self.test_data_dir, transform = trans_val_test)
        test_dataloader = torch.utils.DataLoader(test_dataset, batch_size = self.batch_size,shuffle = True )

        loaders = {'train': train_dataloader, 'valid':val_dataloader, 'test':test_dataloader}

        return loaders


    def train(self):
        """returns trained model"""
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf 

        
        
        
        for epoch in range(1, self.n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.loaders['train']):
                
                # move to GPU
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)
                ## find the loss and update the model parameters accordingly
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                
                # Zero the grads
                self.optimizer.zero_grad()
                
                # Get predictions
                logits = model(data)

                # Compute loss
                loss = self.criterion(logits, target)
                
                # Backward pass
                loss.backward()
                
                # Param updating 
                self.optimizer.step()
                
                
                train_loss = train_loss + (1/(batch_idx + 1)) * (loss.data - train_loss)
                
                
                
            ######################    
            # validate the model #
            ######################
        
            with torch.no_grad(): # Deactivate grad computation
                # Model to validation mode
                self.model.eval()
                for batch_idx, (data, target) in enumerate(self.loaders['valid']):
                    
                    # move to GPU if available
                    if self.use_cuda:
                        data, target = data.to(self.device), target.to(self.device)


                    # Get predictions 
                    logits_val = self.model(data)

                    # Compute loss
                    loss = self.criterion(logits_val, target)

                    ## update the average validation loss
                    valid_loss = valid_loss + (1/(batch_idx + 1)) * (loss.data - valid_loss)




            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
            
            ## TODO: save the model if validation loss has decreased
            
            if valid_loss < valid_loss_min:
                print("Saving model...")
                # Save model 
                torch.save(self.model.state_dict(),self.save_path)
                
                # Update minimum 
                valid_loss_min = valid_loss
                
            
                
        # return trained model
        self.model = model_scratch.load_state_dict(torch.load(self.save_path))
        return self.model



    def test(self):
    
        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        model.eval()
        # Deactivate grad computation
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.loaders['test']):
                # move to GPU if available
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # update average test loss 
                test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
                # convert output probabilities to predicted class
                pred = output.data.max(1, keepdim=True)[1]
                # compare predictions to true label
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                total += data.size(0)
                
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))



