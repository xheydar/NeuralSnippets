import torch

class LossCalculator :
    def __init__( self ):
        pass 

    def __call__( self, model, data, device, use_amp=False ):
        inputs, labels = data 

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.autocast( device_type="cuda", dtype=torch.float16, enabled=use_amp ): 
            outputs = model['net']( inputs )
            loss = model['loss']( outputs, labels )

        return loss

