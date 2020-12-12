## Useful Command
Compile for nv_small

```
./nvdla_compiler --profile basic --prototxt mylenet/caffe-protobuf/mylenet/lenet.prototxt --caffemodel mylenet/mycaffe.caffemodel --configtarget opendla-small
```

Generate test image (4704 Byte)

```
dd if=/dev/zero of=test.img bs=4 count=0 seek=1176
```
