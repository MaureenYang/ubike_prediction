import gzip
import json

import pandas as pd

ubike_f = gzip.open("ubike_data.gz", 'r')
ubike_jdata = ubike_f.read()
ubike_f.close()
ubike_data = json.loads(ubike_jdata)
station_info = []
for key,value in ubike_data['retVal'].items():
    #ubike_dict = [{'sno':value['sno'],'sname':value['sarea'],'slat':float(value['lat']),'slon':float(value['lng'])}]
    #ubike_pos = ubike_pos + ubike_dict
    print('key:',key,'  value:',value)
    value.pop('bemp')
    value.pop('sbi')
    value.pop('act')
    value.pop('mday')
    ubike_dict = value
    ubike_dict['sno'] = key
    station_info = station_info + [ubike_dict]

data = json.dumps(station_info)
with open('station_info.json', 'w') as f:
    json.dump(data, f)

st_info = pd.DataFrame(station_info)
st_info.to_csv("station_info.csv")
