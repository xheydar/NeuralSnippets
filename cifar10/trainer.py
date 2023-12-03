import torch 
import torch.optim as optim
from tqdm import tqdm

class trainer :
    def __init__( self ):
        pass

    def train_step( self, train_loader, optimizer ):

        self.model['net'].train()

        running_loss = 0.0
        data_count = 0
        for idx, data in tqdm(enumerate( train_loader, 0 )):
            inputs, labels = data 

            inputs = inputs.to( self.device )
            labels = labels.to( self.device )

            optimizer.zero_grad()

            outputs = self.model['net']( inputs )

            loss = self.model['loss']( outputs, labels )

            loss.backward()
            optimizer.step()

            running_loss += float(loss)
            data_count += len(inputs)

        return running_loss / data_count

    def eval_step( self, test_loader ):

        self.model['net'].eval()

        corrects = 0;
        total = 0

        for idx, data in tqdm(enumerate(test_loader), 0):
            inputs, labels = data 

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            pred = self.model['net']( inputs )
            pred = torch.argmax( pred, dim=1 )

            pred = pred.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            inds = np.where( pred == labels )[0]

            corrects += len(inds)
            total += len(labels)

        return corrects / total

    def train( self, nepoch=10 ):
        train_loader = self.datasets['train'].get_loader( 16, True )
        test_loader = self.datasets['test'].get_loader( 16, False )

        optimizer = optim.SGD( self.model['net'].parameters(), lr=0.001, momentum=0.9)

        for epoch in range( nepoch ):

            print(f'Epoch {epoch+1}')
            ave_loss = self.train_step( train_loader, optimizer )
            acc = self.eval_step( test_loader )
            print(f'Training average loss : {ave_loss} - Test accuracy : {acc}')

