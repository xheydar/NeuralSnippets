

class LossCalculator :
    def __init__( self ):
        pass 

    def __call__( self, model, data, device ):
        inputs, labels = data 

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model['net']( inputs )

        loss = model['loss']( outputs, labels )

        return loss

