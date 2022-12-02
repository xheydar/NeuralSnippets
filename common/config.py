import os
import platform
import yaml

class config :
    def _make_dir( self, p ):
        if not os.path.isdir(p) :
            os.makedirs(p)

    def __init__( self, cfg_path, tag ):
        basename = os.path.basename( os.getcwd() )

        with open(cfg_path, 'r') as ff :
            cfg = yaml.safe_load( ff )

        data_root = cfg['data_root'][platform.system()]
        project = cfg['project']

        self.data_root = os.path.join( data_root, project, basename, tag )
        self._make_dir( self.data_root )

        self.save_snapshots = False
        if 'snapshots' in cfg['templates'] :
            self._make_dir( os.path.join( self.data_root, 'snapshots')  )
            self.save_snapshots = True

        self.model_path = os.path.join( self.data_root, cfg['templates']['model'] )
        self.ema_model_path = os.path.join( self.data_root, cfg['templates']['ema_model'])
        self.snapshots_tmp = os.path.join( self.data_root, cfg['templates']['snapshots'] )

        self.params = cfg['params']

        self.transform = cfg['transform']
        self.dataset = cfg['dataset']
        self.model = cfg['model']
