import numpy as np
import sys, os
import argparse
import caffe_pb2 as cq
import json

f = open('lenet/lenet_iter_10000.caffemodel', 'rb')
cq2 = cq.NetParameter()
cq2.ParseFromString(f.read())
from google.protobuf import json_format
with open("json.output",'w') as fd:
    fd.write(json_format.MessageToJson(cq2))
f.close()

with open("json.output") as fd:
    with open("mylenet.json.output",'w') as f2d:
        json_dict=json.loads(fd.read())
        relu1=json_dict['layer'][6]
        before_conv1=json_dict['layer'][0:2]
        conv1_to_conv2=json_dict['layer'][2:4]
        conv2_to_rest=json_dict['layer'][4:]
        json_dict['layer']=before_conv1+[dict(relu1)]+conv1_to_conv2+[dict(relu1)]+conv2_to_rest
        del json_dict['layer'][8:10]
        del json_dict['layer'][6]
        assert(json_dict['layer'][1]['name']=="conv1")
        json_dict['layer'][1]['bottom']=['data']
        json_dict['layer'][1]['convolutionParam']['numOutput']=16
        json_dict['layer'][1]['blobs'][0]['data']=json_dict['layer'][1]['blobs'][0]['data'][:2400]
        json_dict['layer'][1]['blobs'][0]['shape']['dim']=['16','6','5','5']
        json_dict['layer'][1]['blobs'][1]['data'] = json_dict['layer'][1]['blobs'][1]['data'][:16]
        json_dict['layer'][1]['blobs'][1]['shape']['dim'] = ['16']
        #json_dict['layer'][1]['name']='conv1'
        #json_dict['layer'][1]['top'] = ['conv1']
        assert (json_dict['layer'][2]['name'] == "relu1")
        #json_dict['layer'][2]['top'] = ['conv1']
        json_dict['layer'][2]['bottom'] = ['conv1']
        json_dict['layer'][2]['top'] = ['conv1']
        assert (json_dict['layer'][3]['name'] == "pool1")
        #json_dict['layer'][3]['bottom'] = ['conv1']
        #json_dict['layer'][3]['name'] = 'pool1'
        #json_dict['layer'][3]['top'] = ['pool1']
        assert (json_dict['layer'][4]['name'] == "conv2")
        #json_dict['layer'][3]['bottom'] = ['pool1']
        json_dict['layer'][4]['convolutionParam']['numOutput'] = 120
        json_dict['layer'][4]['blobs'][0]['data'] = json_dict['layer'][4]['blobs'][0]['data'][:25000]+json_dict['layer'][1]['blobs'][0]['data'][:23000]
        json_dict['layer'][4]['blobs'][0]['shape']['dim'] = ['120', '16', '5', '5']
        json_dict['layer'][4]['blobs'][1]['data'] = json_dict['layer'][4]['blobs'][1]['data'][:50]+json_dict['layer'][1]['blobs'][1]['data'][:50]+json_dict['layer'][1]['blobs'][1]['data'][:20]
        json_dict['layer'][4]['blobs'][1]['shape']['dim'] = ['120']
        assert (json_dict['layer'][5]['name'] == "relu1")
        json_dict['layer'][5]['name']='relu2'
        json_dict['layer'][5]['top'] = ['conv2']
        json_dict['layer'][5]['bottom'] = ['conv2']
        assert (json_dict['layer'][6]['name'] == "ip1")
        json_dict['layer'][6]['bottom'] = ['conv2']
        json_dict['layer'][6]['innerProductParam']['numOutput']=10
        json_dict['layer'][6]['blobs'][0]['data']=json_dict['layer'][6]['blobs'][0]['data'][:1200]
        json_dict['layer'][6]['blobs'][0]['shape']['dim']=['10','120']
        json_dict['layer'][6]['blobs'][1]['data'] = json_dict['layer'][6]['blobs'][1]['data'][:10]
        json_dict['layer'][6]['blobs'][1]['shape']['dim'] = ['10']
        assert (json_dict['layer'][7]['name']=="loss")
        json_dict['layer'][7]['bottom']=['ip1','label']
        f2d.write(json.dumps(json_dict))

# if 0:
#     with open("json.output") as fd:
#         with open("mylenet.json.output",'w') as f2d:
#             json_dict=json.loads(fd.read())
#             del json_dict['layer'][1:3]
#             assert(json_dict['layer'][1]['name']=="conv2")
#             json_dict['layer'][1]['bottom']=['data']
#             json_dict['layer'][1]['convolutionParam']['numOutput']=16
#             json_dict['layer'][1]['blobs'][0]['data']=json_dict['layer'][1]['blobs'][0]['data']#[:2400]
#             json_dict['layer'][1]['blobs'][0]['shape']['dim']=['16','6','5','5']
#             json_dict['layer'][1]['blobs'][1]['data'] = json_dict['layer'][1]['blobs'][1]['data']#[:16]
#             json_dict['layer'][1]['blobs'][1]['shape']['dim'] = ['16']
#             json_dict['layer'][1]['name']='conv1'
#             json_dict['layer'][1]['top'] = ['conv1']
#             assert (json_dict['layer'][2]['name'] == "pool2")
#             json_dict['layer'][2]['bottom'] = ['conv1']
#             json_dict['layer'][2]['name'] = 'pool1'
#             json_dict['layer'][2]['top'] = ['pool1']
#             assert (json_dict['layer'][3]['name'] == "ip1")
#             json_dict['layer'][3]['bottom'] = ['pool1']
#             json_dict['layer'][3]['innerProductParam']['numOutput']=120
#             json_dict['layer'][3]['blobs'][0]['data']=json_dict['layer'][3]['blobs'][0]['data']#[:1920]
#             json_dict['layer'][3]['blobs'][0]['shape']['dim']=['120','16']
#             json_dict['layer'][3]['blobs'][1]['data'] = json_dict['layer'][3]['blobs'][1]['data']#[:120]
#             json_dict['layer'][3]['blobs'][1]['shape']['dim'] = ['120']
#             assert (json_dict['layer'][4]['name'] == "relu1")
#             assert (json_dict['layer'][5]['name'] == "ip2")
#             f2d.write(json.dumps(json_dict))

print ("name 1st layer: " + cq2.layer[1].name)
cq2 = cq.NetParameter()

with open("mylenet/mycaffe.caffemodel",'wb') as fd:
    with open("mylenet.json.output",'r') as f2d:
        fd.write(json_format.Parse(f2d.read(),cq2).SerializeToString())
pass