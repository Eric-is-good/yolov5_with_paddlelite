# PaddleLite for yolov5 开发





## 官方模型的改动

[官方例程](https://github.com/PaddlePaddle/Paddle-Lite-Demo)

后很多人都有 [官方不匹配](https://github.com/PaddlePaddle/Paddle-Lite-Demo) 的问题

如下所示，paddle 例程采用了 yolo 不是主流（？）的三个输出，而现在 yolov5 官方已经改成了在神经网络里面就处理好

![](https://github.com/Eric-is-good/yolov5_with_paddlelite/blob/master/pics/3.jpg)



现在的模型输出长这样

![](https://github.com/Eric-is-good/yolov5_with_paddlelite/blob/master/pics/4.jpg)

yolo 官方给的只用输出 0 预测，不难看出 6300 = 10 * 10 + 20 * 20 + 40 * 40 ，输出 0 直接把原来三个输出合并了，避免了分割图像不一致加 stride 的麻烦。



## 关于后处理（c++）

pred 直接得到四个数组（tensor 矩阵）

他们长成这样

![](https://github.com/Eric-is-good/yolov5_with_paddlelite/blob/master/pics/1.jpg)



对于 paddle 而言，就是用 predictor_->GetInput(i) 来取第 i 个数组。



python 是用第一个进行预测（0 号矩阵）

所以我决定自己写一个 non_max_suppression 的 c++ 语言版本



## non_max_suppression

1. 首先尝试能不能把 non_max_suppression 直接加入到 onnx 模型里面。

   模型就成了这样

![](https://github.com/Eric-is-good/yolov5_with_paddlelite/blob/master/2.jpg)

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

   后处理总概函数
   
   ```c++
   void Detector::Postprocess(std::vector<Object> *results) {
       // 重写后处理，只是用第一个三维矩阵
       std::map<int, std::vector<Object>> raw_outputs;
   
       std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
             std::move(predictor_->GetOutput(0)));
   
       auto *outptr = output_tensor->data<float>();
       auto shape_out = output_tensor->shape();  // [1, 6300, 6]
   
       // 先挑选一波 > 0.25 的，因为后面复杂度为 O(N^2)
       ExtractBoxes(outptr, &raw_outputs, shape_out);
   
       // 过一个 nms 非极大值抑制
       Nms(raw_outputs, results);
   
   }
   ```





​      关于 ExtractBoxes 解析函数

```c++
void Detector::ExtractBoxes(const float *in,    // float[]
                            std::map<int, std::vector<Object>> *outs,
                            const std::vector<int64_t> &shape) {    // [1, 6300, 6]

      int cls_num = shape[2] - 5;

      // 先取出置信度大于 conf_thres 的
      for (int i = 0; i < shape[1]; i++) {
            int offset = i*shape[2];

            if(in[offset+4] < confThresh_)
                  continue;

            Object obj;
            int left = (int)((in[offset] - in[offset+2]/2 - (inputWidth_ - inputW )/ 2.0 ) / ratio_);   // 减去 padding
            int top = (int)((in[offset+1] - in[offset+3]/2)/ratio_);
            int w = (int)(in[offset+2]/ratio_);
            int h = (int)(in[offset+3]/ratio_);

            // 再恢复成原来图片大小一下


            obj.rec = cv::Rect(left, top, w, h);

            // 找到最大的类
            int max_cls_id = 0;
            float max_cls_val = 0;
            for (int i = 0; i < cls_num; i++) {
                if (in[offset + 5 + i] > max_cls_val) {
                    max_cls_val = in[offset + 5 + i];
                    max_cls_id = i;
                }
            }

            obj.class_id = max_cls_id;
            obj.prob = in[offset+4] * in[offset+5+max_cls_id];

            if (outs->count(obj.class_id) == 0)
                 outs->emplace(obj.class_id, std::vector<Object>());
                (*outs)[obj.class_id].emplace_back(obj);
            }


}
```

nms 非极大值抑制

```c++
void Detector::Nms(const std::map<int, std::vector<Object>> &src,
                   std::vector<Object> *res) {
  for (auto it = src.begin(); it != src.end(); it++) {
    auto dets = it->second;
    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
      auto &item = dets[m];
      item.class_name = item.class_id >= 0 && item.class_id < labelList_.size()
                            ? labelList_[item.class_id]
                            : "Unknow";
      item.fill_color = item.class_id >= 0 && item.class_id < colorMap_.size()
                            ? colorMap_[item.class_id]
                            : cv::Scalar(0, 0, 0);
      res->push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n) {
        if (iou_calc(item.rec, dets[n].rec) > nmsThresh_) {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}
```



## 总结

可以直接替换模型。

像这样改动，就可以适应绝大多数的 yolo 模型。
