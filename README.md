这是一个可以在RK3588上运行的yolo26-demo项目，项目自带有量化后的官方模型可以进行测试使用。

转换及python推理代码见：https://github.com/li2390893/yolo26-rk3588-python.git

详细讲解视频，B站搜索：橘子搞AI视觉
【【已开源】YOLO26在RK3588上部署详解】 https://www.bilibili.com/video/BV1gNr4BFEZP

编译
```
cmake -S . -B build
cmake --build build
```

运行
```
./build/yolo26_img ./weights/yolo26s.int.rknn ./images/bus.jpg
```
yolo26_imgs和yolo26_video请自行探索