import requests
import json
import threading
import copy
import datetime

def send_background( uri, data ):
    response = requests.post( uri, data=json.dumps(data) )

    if response.status_code != 200 :
        print('API Error : %s' % ( response.content ))

class api :
    def __init__( self, uri, key ):

        self.uri = uri 
        self.key = key  

        self.cfg = {}
        self.items = []

    def reset( self ):
        self.cfg = {}
        self.items = []

    def add_cfg( self, name, value ):
        self.cfg[name] = value

    def add_item( self, item ):
        item['timestamp'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')
        self.items.append( item )
        

    def send( self, status ):

        payload = {}
        payload['items'] = self.items 
        payload['cfg'] = self.cfg

        data = {}
        data['key'] = self.key 
        data['status'] = status
        data['payload'] = payload  

        data = copy.deepcopy(data)
        self.items = []

        thread = threading.Thread( target=send_background, args=(self.uri, data))
        thread.start()
        #send_background( self.uri, data )


