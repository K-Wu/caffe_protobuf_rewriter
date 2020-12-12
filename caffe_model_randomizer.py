import numpy as np
import sys, os
import argparse
import caffe_pb2 as cq
import json
if __name__=="__main__":
    f = open('mylenet/mycaffe.caffemodel', 'rb')
    cq2 = cq.NetParameter()
    cq2.ParseFromString(f.read())
    from google.protobuf import json_format

    with open("json.output", 'w') as fd:
        fd.write(json_format.MessageToJson(cq2))
    f.close()

    with open("json.output") as fd:
        with open("mylenet.json.output", 'w') as f2d:
            json_dict = json.loads(fd.read())
            json_dict['layer'][6]['blobs'][1]['data']=np.random.random(10).tolist()
            json_dict['layer'][4]['blobs'][1]['data']=np.random.random(120).tolist()
            json_dict['layer'][1]['blobs'][1]['data']=np.random.random(16).tolist()
            f2d.write(json.dumps(json_dict))

    cq2 = cq.NetParameter()

    with open("mylenet/mycaffe.caffemodel", 'wb') as fd:
        with open("mylenet.json.output", 'r') as f2d:
            fd.write(json_format.Parse(f2d.read(), cq2).SerializeToString())
    pass