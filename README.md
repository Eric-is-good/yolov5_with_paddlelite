# PaddleLite for yolov5 开发



[TOC]



## 关于后处理（c++）

pred 直接得到四个数组（tensor 矩阵）

他们长成这样

![](pics\1.jpg)



对于 paddle 而言，就是用 predictor_->GetInput(i) 来取第 i 个数组。

暂未搞懂后面的三个矩阵是什么



python 是用第一个进行预测（0 号矩阵）

所以我决定自己写一个 non_max_suppression 的 c 语言版本



## non_max_suppression

1. 首先尝试能不能把 non_max_suppression 直接加入到 onnx 模型里面。

![](pics\2.jpg)

​       显然不能，变成了这个什么东西

2. c++ 重写

   先看 paddle 数据流向（简化版）

   ```c++
   void Detector::Postprocess(std::vector<Object> *results) {
   	std::map<int, std::vector<Object>> raw_outputs;
   	auto *outptr = output_tensor->data<float>();
   	ExtractBoxes(k-1, outptr, &raw_outputs, shape_out);
   	Nms(raw_outputs, results);
   }
   
   ```

​       应该是最后到 results 里面去了，raw_outputs 只是中间变量。

3. 官方 python 的后处理方法

   ```python
   # 先取出置信度大于 conf_thres 的
   xc = prediction[..., 4] > conf_thres
   x = x[xc[xi]]
   
   # conf = obj_conf * cls_conf  其实没有用，就是单纯展示置信度
   x[:, 5:] *= x[:, 4:5]
   
   # 解析 box
   box = xywh2xyxy(x[:, :4])
   
   # 每条预测一个最大的
   conf, j = x[:, 5:].max(1, keepdim=True)
   
   # 过一个 NMS
   i = torchvision.ops.nms(boxes, scores, iou_thres)  # 返回应该保留的序号
   
   output[xi] = x[i]  # 输出第 xi 张图片
   
   
   # 回复到图片比例
   det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
   
   ```

4. c++ 复现

   







