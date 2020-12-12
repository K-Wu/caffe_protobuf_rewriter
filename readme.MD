## Useful Command
Compile for nv_small

```
./nvdla_compiler --profile basic --prototxt mylenet/caffe-protobuf/mylenet/lenet.prototxt --caffemodel mylenet/mycaffe.caffemodel --configtarget opendla-small
```

Generate test image (4704 Byte)

```
import numpy
numpy.random.random(6*14*14).astype(numpy.float32).tofile("test.img")
```

