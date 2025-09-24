import os
import time
from calendar import timegm
from datetime import date, datetime

import requests
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

today = date.today()

#base = "http://monipe-central.rnp.br"
base = "https://pmp-archive.geant.org"

def get_response(url, time_range):
    cont = 0
    while True:
        header = {"time-range": time_range}
        response = requests.get(url, params=header, verify=False)
        if response.status_code == 200:
            return response
        else:
            cont = cont + 1 
            print("Codigo do Status recebido {}. Tentando novamente em 1 min...".format(response.status_code))
            print("Tentativa: {}".format(cont))
            time.sleep(60)
    
def get_data(url, time_range):
    response = get_response(url, time_range)
    json_data = response.json()
     
    return json_data

def request_by_metadata_key(url, type):
    response = requests.get(url, verify=False)
    #print('teste2: ', url)
    json_data = response.json()
    #print(response.status_code)
    if (response.status_code == 200):
        for obj in json_data["event-types"]:
            if obj["event-type"] == type:
                print(obj)
                return obj
    else:
        return response.status_code

def calc_mean(val):
    values = []
    for key, value in val.items():
        values.append(float(key))
    
    return round(sum(values)/len(values),2)


#coleta a vazao
def request(folder, name, source, destination, type, time_range, target_bandwidth="9999999999"):
    disable_warnings(InsecureRequestWarning)
    #url = "http://monipe-central.rnp.br/esmond/perfsonar/archive/?"
    url = "https://pmp-archive.geant.org/esmond/perfsonar/archive/?"
    hearder = {"pscheduler-test-type": type, "source": source, "destination": destination}#, "bw-target-bandwidth": target_bandwidth, "time-range": time_range}
    response = requests.get(url, params=hearder, verify=False)
    
    print('endereço', response.url)
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_data = response.json()    
    values = []
    if (response.status_code == 200):
        print("Ok")
        bases = []
        for obj in json_data:
            types_list = obj['event-types']
            for obj_types in types_list:
                if obj_types.get('event-type') == type:
                    bases.append(obj_types.get('base-uri'))
                    break
        with open(folder+" esmond data " + source + ' to ' + destination + ' ' + today.strftime("%m-%d-%Y")+".csv", "w") as f:
            f.write(f"{'Timestamp'},{'Data'},{'Vazao'}\n")
            for link in bases:
                values = get_data(base + link, time_range)
                for value in values:
                    f.write(f"{value['ts']},{datetime.fromtimestamp(int(value['ts'])).strftime('%Y-%m-%d %H:%M:%S')},{str(value['val'])}\n")
        f.close()

def request_traceroute(folder, name, source, destination, type, time_range):
    disable_warnings(InsecureRequestWarning)
    limite = "?limit=26400"
    url = "https://pmp-archive.geant.org/esmond/perfsonar/archive/?"
    header = {"pscheduler-test-type": type, "source": source,
              "destination": destination, "time-range": time_range}
    response = requests.get(url, params=header, verify=False)

    if not os.path.exists(folder):
        os.makedirs(folder)
    json_data = response.json()
    #get_data(url, time_range)
    #response.json()
    values = []
    if (response.status_code == 200):
        print("Ok")
        bases = []
        for obj in json_data:
            types_list = obj['event-types']
            for obj_types in types_list:
                if obj_types.get('event-type') == "packet-trace":
                    bases.append(obj_types.get('base-uri'))
                    break
    
    with open(folder + name +" esmond data " + source + ' to ' + destination + ' ' + today.strftime("%m-%d-%Y")+".csv", "w") as f:
        #f.write(f"{'Timestamp'},{'Data'}, {'xxxxx'}\n")
        
        for link in bases:
            
            values: list = get_data(base + link + limite, time_range)
            #print('values', values)
        
            for obj in values:
                #ip_hostname_list = [item['ip'] for item in value['val']]
                f.write(f"{int(obj['ts'])},{datetime.fromtimestamp(int(obj['ts'])).strftime('%Y-%m-%d %H:%M:%S')},")
                for dado in range(len(obj['val'])):
                    try:
                        #print(value['val'][ip]['ip'], value['val'][ip]['hostname'])
                        if dado != len(obj['val']) - 1:
                            f.write(f"{obj['val'][dado]['hostname']},")
                            #f.write(f"{obj['val'][dado]['ip']}, {obj['val'][dado]['hostname']},")
                        else:
                            f.write(f"{obj['val'][dado]['hostname']}\n")
                            #f.write(f"{obj['val'][dado]['ip']}, {obj['val'][dado]['hostname']}")
                    except BaseException:
                        if dado != len(obj['val']) - 1:
                            f.write("'No Hostname',")
                            #f.write("'No Ip', 'No Hostname',")
                        else:
                            f.write("'No Hostname'\n")
                            #f.write("'No Ip', 'No Hostname'")
                #f.write('\n')
           
    f.close()
    

def request_atraso(folder, name, source, destination, type, time_range, label):
    disable_warnings(InsecureRequestWarning)
    limite1 = "?limit=285000"
    url = "https://pmp-archive.geant.org/esmond/perfsonar/archive/?"
    header = {"pscheduler-test-type": type, "source": source,
              "destination": destination, "time-range": time_range}
    
    response = requests.get(url, params=header, verify=False)
    print(response.url)
    if not os.path.exists(folder):
        os.makedirs(folder)
    json_data = response.json()
    values = []
    if (response.status_code == 200):
        print("Ok")
        bases = []
        for obj in json_data:
            types_list = obj['event-types']
            for obj_types in types_list:
                if obj_types.get('event-type') == label:
                    bases.append(obj_types.get('base-uri'))
                    break
        with open(folder + name +" esmond data " + source + ' to ' + destination + ' ' + today.strftime("%m-%d-%Y")+".csv", "w") as f:
        #with open(folder +name+" esmond data " + source.split('-')[1] + '-' + destination.split('-')[1] + ' ' + today.strftime("%m-%d-%Y")+".csv", "w") as f:
            f.write(f"{'Timestamp'},{'Data'},{'Atraso(ms)'}\n")
            for link in bases:
                values = get_data(base + link + limite1, time_range)
                for value in values:
                    f.write(f"{value['ts']},{datetime.fromtimestamp(int(value['ts'])).strftime('%Y-%m-%d %H:%M:%S')},{calc_mean(value['val'])}\n")
        f.close()




request("./pmp2/bbr/","bbr", "psmp-gn-bw-lis-pt.geant.org", "perfsonar.restena.lu", 
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/bbr/","bbr", "psmp-gn-bw-poz-pl.geant.org","perfsonar-ankara.ulakbim.gov.tr", 
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/bbr/","bbr", "psmp-gn-bw-poz-pl.geant.org","pspmp-anella.csuc.cat", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr", "psmp-gn-bw-lis-pt.geant.org","perfsonar-ankara.ulakbim.gov.tr", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr", "psmall.lut.ac.uk","psmp-gn-bw-vie-at.geant.org", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr", "perfsonar-sonda.rediris.es", "psmp-gn-bw-lis-pt.geant.org", 
        "throughput", "15552000", "10000000000")  # 6 meses

'''request("./pmp2/bbr/","bbr",  "perfsonar.restena.lu", "psmp-gn-bw-lis-pt.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/bbr/","bbr", "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-poz-pl.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/bbr/","bbr", "pspmp-anella.csuc.cat", "psmp-gn-bw-poz-pl.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr", "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-lis-pt.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr", "psmp-gn-bw-vie-at.geant.org", "psmall.lut.ac.uk",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/bbr/","bbr",  "psmp-gn-bw-lis-pt.geant.org", "perfsonar-sonda.rediris.es",
        "throughput", "15552000", "10000000000")  # 6 meses'''


print('protocolo CUBIC: ')

request("./pmp2/cubic/","cubic", "psmp-gn-bw-lis-pt.geant.org", "perfsonar.restena.lu", 
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/cubic/","cubic", "psmp-gn-bw-poz-pl.geant.org","perfsonar-ankara.ulakbim.gov.tr", 
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/cubic/","cubic", "psmp-gn-bw-poz-pl.geant.org","pspmp-anella.csuc.cat", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic", "psmp-gn-bw-lis-pt.geant.org","perfsonar-ankara.ulakbim.gov.tr", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic", "psmall.lut.ac.uk","psmp-gn-bw-vie-at.geant.org", 
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic", "perfsonar-sonda.rediris.es", "psmp-gn-bw-lis-pt.geant.org", 
        "throughput", "15552000", "10000000000")  # 6 meses

'''request("./pmp2/cubic/","cubic",  "perfsonar.restena.lu", "psmp-gn-bw-lis-pt.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/cubic/","cubic", "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-poz-pl.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses

request("./pmp2/cubic/","cubic", "pspmp-anella.csuc.cat", "psmp-gn-bw-poz-pl.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic", "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-lis-pt.geant.org",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic", "psmp-gn-bw-vie-at.geant.org", "psmall.lut.ac.uk",
        "throughput", "15552000", "10000000000")  # 6 meses
request("./pmp2/cubic/","cubic",  "psmp-gn-bw-lis-pt.geant.org", "perfsonar-sonda.rediris.es",
        "throughput", "15552000", "10000000000")  # 6 meses'''

#print('Traceroute0')
#request_traceroute("./pmp2/traceroute/", "traceroute", "psmp-gn-bw-lis-pt.geant.org", "perfsonar.restena.lu",
#                      "trace", "15552000")
print('Traceroute: ')
'''request_traceroute("./pmp2/traceroute/", "traceroute", "psmp-gn-bw-lis-pt.geant.org", "perfsonar.restena.lu",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "psmp-gn-bw-poz-pl.geant.org", "perfsonar-ankara.ulakbim.gov.tr",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "psmp-gn-bw-poz-pl.geant.org", "pspmp-anella.csuc.cat",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "psmp-gn-bw-lis-pt.geant.org", "perfsonar-ankara.ulakbim.gov.tr",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "psmall.lut.ac.uk", "psmp-gn-bw-vie-at.geant.org",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "perfsonar-sonda.rediris.es", "psmp-gn-bw-lis-pt.geant.org",
                      "trace", "15552000")'''

'''request_traceroute("./pmp2/traceroute/", "traceroute", "perfsonar.restena.lu", "psmp-gn-bw-lis-pt.geant.org",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute", "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-poz-pl.geant.org",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute",  "pspmp-anella.csuc.cat", "psmp-gn-bw-poz-pl.geant.org",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute",  "perfsonar-ankara.ulakbim.gov.tr", "psmp-gn-bw-lis-pt.geant.org",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute",  "psmp-gn-bw-vie-at.geant.org", "psmall.lut.ac.uk",
                      "trace", "15552000")

request_traceroute("./pmp2/traceroute/", "traceroute",  "psmp-gn-bw-lis-pt.geant.org", "perfsonar-sonda.rediris.es",
                      "trace", "15552000")'''

'''request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-lis-pt.geant.org","perfsonar-sonda.rediris.es", "latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-vie-at.geant.org","psmall.lut.ac.uk", "latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "perfsonar-ankara.ulakbim.gov.tr","psmp-gn-owd-lis-pt.geant.org", "latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "pspmp-anella.csuc.cat","psmp-gn-owd-poz-pl.geant.org", "latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "perfsonar-ankara.ulakbim.gov.tr","psmp-gn-owd-poz-pl.geant.org", "latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "perfsonar.restena.lu","psmp-gn-owd-lis-pt.geant.org", "latencybg", 
               "15552000", "histogram-owdelay")'''

# request_atraso("./pmp2/atraso/","atraso", "perfsonar-sonda.rediris.es", "psmp-gn-owd-lis-pt.geant.org","latencybg", 
#                "15552000", "histogram-owdelay")

# request_atraso("./pmp2/atraso/","atraso", "psmall.lut.ac.uk", "psmp-gn-owd-vie-at.geant.org","latencybg", 
#                "15552000", "histogram-owdelay")

# request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-lis-pt.geant.org", "perfsonar-ankara.ulakbim.gov.tr","latencybg", 
#                "15552000", "histogram-owdelay")

# request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-poz-pl.geant.org", "pspmp-anella.csuc.cat","latencybg", 
#                "15552000", "histogram-owdelay")

'''request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-poz-pl.geant.org", "perfsonar-ankara.ulakbim.gov.tr","latencybg", 
               "15552000", "histogram-owdelay")

request_atraso("./pmp2/atraso/","atraso", "psmp-gn-owd-lis-pt.geant.org", "perfsonar.restena.lu","latencybg", 
               "15552000", "histogram-owdelay")'''


'''request("datasets vazao/original/cubic/", "cubic", "monipe-se-banda.rnp.br", "monipe-ac-banda.rnp.br",
       "throughput", "15552000")  # 6 meses'''