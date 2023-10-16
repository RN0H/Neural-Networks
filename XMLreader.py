import xml.dom.minidom
from collections import defaultdict as d
d = {}

def ParseXML(document)->None:
    for type in document:
        d[str(type.getAttribute('type'))] = str(type.getElementsByTagName("value")[0].firstChild.data)
    
    for i in d:
        print(i, "----->", d[i])

if __name__ == "__main__":
    pass
