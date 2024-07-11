// Made basing on the OpenVINO example: https://github.com/OpenVINO-dev-contest/YOLOv7_OpenVINO_cpp-python/blob/main/cpp/main_preprocessing.cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include "ObjectDetector.h"

ObjectDetector::ObjectDetector(const std::string &model_path, const std::string &device_name)
{
    // Initialize OpenVINO Runtime Core
    core = std::make_shared<ov::Core>();
    model = core->read_model(model_path);

    ov::preprocess::PrePostProcessor prep(model);
    prep.input().tensor().set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    prep.input().model().set_layout("NCHW");
    prep.input().preprocess().convert_color(ov::preprocess::ColorFormat::RGB).scale({255.0, 255.0, 255.0});
    // std::cout << "Preprocessor: " << prep << std::endl;
    model = prep.build();

    compiled_model = core->compile_model(model, device_name);
    input_port = compiled_model.input();
    infer_request = compiled_model.create_infer_request();
}

std::vector<LabeledBox> ObjectDetector::detect(cv::Mat &frame)
{
    // Prepare input
    cv::Mat img;
    padding.clear();
    cv::Mat boxed = letterbox(frame, img_h, img_w);
    boxed.convertTo(boxed, CV_32FC3);

    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), (float *)boxed.data);
    infer_request.set_input_tensor(input_tensor);

    auto start = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> detection_time = end - start;
    // std::cout << "Inference time: " << detection_time.count() << " seconds" << std::endl;

    // Process output
    std::vector<LabeledBox> proposals;
    auto output_num = model->get_output_size();
    if (output_num == 1)
    {
        process_single_output(proposals);
    }
    else
    {
        process_multiple_outputs(proposals);
    }

    std::vector<LabeledBox> detections;
    apply_nms_and_get_detections(proposals, detections);
    for (auto &detection : detections)
    {
        detection.rect = scale_box(detection.rect, frame.rows, frame.cols);
    }

    return detections;
}

cv::Mat ObjectDetector::letterbox(cv::Mat &src, int h, int w)
{
    int in_w = src.cols;
    int in_h = src.rows;
    float r = std::min(float(h) / in_h, float(w) / in_w);
    int inside_w = std::round(in_w * r);
    int inside_h = std::round(in_h * r);
    int padd_w = w - inside_w;
    int padd_h = h - inside_h;
    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padding.push_back(padd_w);
    padding.push_back(padd_h);
    padding.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));

    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect ObjectDetector::scale_box(cv::Rect box, float raw_h, float raw_w)
{

    float x0 = box.x - padding[0];
    float y0 = box.y - padding[1];
    float x1 = x0 + box.width;
    float y1 = y0 + box.height;
    auto ratio = padding[2];

    x0 = x0 / ratio;
    y0 = y0 / ratio;
    x1 = x1 / ratio;
    y1 = y1 / ratio;

    x0 = std::max(std::min(x0, (float)(raw_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(raw_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(raw_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(raw_h - 1)), 0.f);

    cv::Rect scaled_box;
    scaled_box.x = x0;
    scaled_box.y = y0;
    scaled_box.width = x1 - x0;
    scaled_box.height = y1 - y0;

    return scaled_box;
}

void ObjectDetector::draw_box(int classId, float conf, cv::Rect box, cv::Mat &frame)
{
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;

    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    std::string label = cv::format("%.2f", conf);
    if (!class_names.empty())
    {
        CV_Assert(classId < (int)class_names.size());
        label = class_names[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)), cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
}

void ObjectDetector::generate_proposals(int stride, const float *feat, float prob_threshold, std::vector<LabeledBox> &objects)
{
    float anchors[18] = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
    int anchor_num = 3;
    int feat_w = 640 / stride;
    int feat_h = 640 / stride;
    int anchor_group = 0;
    if (stride == 8)
        anchor_group = 0;
    if (stride == 16)
        anchor_group = 1;
    if (stride == 32)
        anchor_group = 2;

    for (int anchor = 0; anchor <= anchor_num - 1; anchor++)
    {
        for (int i = 0; i <= feat_h - 1; i++)
        {
            for (int j = 0; j <= feat_w - 1; j++)
            {
                float box_prob = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + 4];
                box_prob = sigmoid(box_prob);

                if (box_prob < prob_threshold)
                    continue;
                float x = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + 0];
                float y = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + 1];
                float w = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + 2];
                float h = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + 3];

                x = sigmoid(x);
                y = sigmoid(y);
                w = sigmoid(w);
                h = sigmoid(h);

                float max_prob = 0;
                int idx = 0;
                for (int t = 5; t < num_classes + 5; t++)
                {
                    float tp = feat[anchor * feat_h * feat_w * (num_classes + 5) + i * feat_w * (num_classes + 5) + j * (num_classes + 5) + t];
                    tp = sigmoid(tp);
                    if (tp > max_prob)
                    {
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = box_prob * max_prob;
                if (cof < prob_threshold)
                    continue;
                x = (x * 2.0 - 0.5 + j) * stride;
                y = (y * 2.0 - 0.5 + i) * stride;
                w = pow(w * 2.0, 2) * anchors[anchor_group * 6 + anchor * 2 + 0];
                h = pow(h * 2.0, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];

                LabeledBox obj;
                obj.rect.x = x - w / 2.0f;
                obj.rect.y = y - h / 2.0f;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = idx - 5;
                obj.prob = cof;
                objects.push_back(obj);
            }
        }
    }
}

void ObjectDetector::process_single_output(std::vector<LabeledBox> &proposals)
{
    // Handle single output case (one model output)
    int total_num = 25200;
    auto output_tensor = infer_request.get_output_tensor(0);
    const float *result = output_tensor.data<const float>();
    std::vector<LabeledBox> objects;
    for (int i = 0; i <= total_num - 1; i++)
    {
        double max_prob = 0;
        int idx = 0;
        float box_prob = result[i * (num_classes + 5) + 4];
        if (box_prob < prob_threshold)
            continue;
        for (int t = 5; t < (num_classes + 5); ++t)
        {
            double tp = result[i * (num_classes + 5) + t];
            if (tp > max_prob)
            {
                max_prob = tp;
                idx = t;
            }
            float cof = box_prob * max_prob;
            if (cof < prob_threshold)
                continue;
            LabeledBox obj;
            obj.rect.x = result[i * (num_classes + 5) + 0] - result[i * (num_classes + 5) + 2] / 2;
            obj.rect.y = result[i * (num_classes + 5) + 1] - result[i * (num_classes + 5) + 3] / 2;
            obj.rect.width = result[i * (num_classes + 5) + 2];
            obj.rect.height = result[i * (num_classes + 5) + 3];
            obj.label = idx - 5;
            obj.prob = cof;
            objects.push_back(obj);
        }
    }
    proposals.insert(proposals.end(), objects.begin(), objects.end());
}

void ObjectDetector::process_multiple_outputs(std::vector<LabeledBox> &proposals)
{
    auto output_tensor_p8 = infer_request.get_output_tensor(0);
    const float *result_p8 = output_tensor_p8.data<const float>();
    auto output_tensor_p16 = infer_request.get_output_tensor(1);
    const float *result_p16 = output_tensor_p16.data<const float>();
    auto output_tensor_p32 = infer_request.get_output_tensor(2);
    const float *result_p32 = output_tensor_p32.data<const float>();

    std::vector<LabeledBox> objects8;
    std::vector<LabeledBox> objects16;
    std::vector<LabeledBox> objects32;

    generate_proposals(8, result_p8, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    generate_proposals(16, result_p16, prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(32, result_p32, prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
}

void ObjectDetector::apply_nms_and_get_detections(const std::vector<LabeledBox> &proposals, std::vector<LabeledBox> &detections)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < proposals.size(); i++)
    {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
    }

    std::vector<int> picked;
    // do non maximum suppression for each bounding boxx
    cv::dnn::NMSBoxes(boxes, confidences, prob_threshold, nms_threshold, picked);

    for (size_t i = 0; i < picked.size(); i++)
    {
        int idx = picked[i];
        LabeledBox obj;
        obj.rect = boxes[idx];
        obj.label = classIds[idx];
        obj.prob = confidences[idx];
        detections.push_back(obj);
    }
}

float ObjectDetector::sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

void ObjectDetector::draw_detections(cv::Mat &frame, const std::vector<LabeledBox> &detections)
{
    for (const auto &obj : detections)
    {
        draw_box(obj.label, obj.prob, obj.rect, frame);
    }
}
