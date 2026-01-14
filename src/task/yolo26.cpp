#include "yolo26.h"
#include <random>
#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

// define global classes
static std::vector<std::string> g_classes = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
         "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
         "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
         "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
         "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
         "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
         "hair drier", "toothbrush"};


Yolo26::Yolo26()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
}

Yolo26::~Yolo26()
{
    // release input tensor and output tensor
    NN_LOG_DEBUG("release input tensor");
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    NN_LOG_DEBUG("release output tensor");
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr)
        {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolo26::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolo26 load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolo26 input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 6)
    {
        NN_LOG_ERROR("yolo26 output tensor number is not 6, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        want_float_ = true;
        NN_LOG_WARNING("yolo26 output tensor type is float16, want type set to float32");
    }
    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }

    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolo26::Preprocess(const cv::Mat &img, cv::Mat &image_letterbox)
{

    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox
    letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
    cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);



    return NN_SUCCESS;
}

nn_error_e Yolo26::Inference()
{
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input_tensor_);
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolo26::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    void *output_data[6];
    for (int i = 0; i < 6; i++)
    {
        output_data[i] = (void *)output_tensors_[i].data;
    }
    std::vector<float> DetectiontRects;
    if (want_float_)
    {
        // 使用浮点数版本的后处理，他也支持量化的模型
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
        // NN_LOG_INFO("use float version postprocess");
    }
    else
    {
        // 使用量化版本的后处理，只能处理量化的模型
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
        // NN_LOG_INFO("use int8 version postprocess");
    }

    int img_width = img.cols;
    int img_height = img.rows;
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        Detection result;
        result.class_id = classId;
        result.confidence = conf;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = g_classes[result.class_id];
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);

        objects.push_back(result);
    }

    return NN_SUCCESS;
}
void letterbox_decode(std::vector<Detection> &objects, bool hor, int pad)
{
    for (auto &obj : objects)
    {
        if (hor)
        {
            obj.box.x -= pad;
        }
        else
        {
            obj.box.y -= pad;
        }
    }
}

nn_error_e Yolo26::Run(const cv::Mat &img, std::vector<Detection> &objects)
{

    // letterbox后的图像
    cv::Mat image_letterbox;
    std::chrono::steady_clock::time_point tt0 = std::chrono::steady_clock::now();
    Preprocess(img, image_letterbox);
    std::chrono::steady_clock::time_point tt1 = std::chrono::steady_clock::now();
    // 推理
    Inference();
    std::chrono::steady_clock::time_point tt2 = std::chrono::steady_clock::now();
    // 后处理
    // 后处理
    Postprocess(image_letterbox, objects);
    letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad);
    std::chrono::steady_clock::time_point tt3 = std::chrono::steady_clock::now();

    float tmpTime = (std::chrono::duration_cast<std::chrono::duration<double>>(tt1 - tt0)).count()*1000;
    float tmpTime2 = (std::chrono::duration_cast<std::chrono::duration<double>>(tt2 - tt1)).count()*1000;
    float tmpTime3 = (std::chrono::duration_cast<std::chrono::duration<double>>(tt3 - tt2)).count()*1000;
    float tmpTime4 = (std::chrono::duration_cast<std::chrono::duration<double>>(tt3 - tt0)).count()*1000;
    //Preprocess time ms
    printf("%.2f;", tmpTime);
    //Inference time ms
    printf("%.2f;", tmpTime2);
    //Postprocess time ms
    printf("%.2f;", tmpTime3);
    //Total time ms
    printf("%.2f\n", tmpTime4);

    return NN_SUCCESS;
}