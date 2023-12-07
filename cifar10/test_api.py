

from api import api


if __name__=="__main__" :

    a = api('http://localhost:3000/api/logs/update-experiment/',
             'UEVNIYDOITPQGPXZFHPS29959808541554486079')
    a.send( {'heydar':'zohre'} )
