import numpy as np
from PIL import Image
import caffe
caffe.set_mode_cpu()
net = caffe.Net('mylenet/lenet.prototxt', caffe.TEST)
net.save('mylenet/mycaffe.caffemodel')
