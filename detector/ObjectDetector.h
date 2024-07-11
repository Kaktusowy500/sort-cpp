#pragma once
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct LabeledBox
{
    cv::Rect_<double> rect;
    int label;
    float prob;
    std::pair<double, double> get_box_center() const
    {
        return std::make_pair(rect.x + rect.width / 2, rect.y + rect.height / 2);
    }
};

class ObjectDetector
{
public:
    ObjectDetector(const std::string &model_path, const std::string &device_name);

    std::vector<LabeledBox> detect(cv::Mat &frame);
    void draw_detections(cv::Mat &frame, const std::vector<LabeledBox> &detections);

private:
    const int img_h = 640;
    const int img_w = 640;
    const float prob_threshold = 0.6f;
    const float nms_threshold = 0.50f;
    const std::vector<std::string> class_names = {"vehicle"};
    const int num_classes = class_names.size();
    std::vector<float> padding;

    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;

    cv::Mat letterbox(cv::Mat &src, int h, int w);
    cv::Rect scale_box(cv::Rect box, float raw_h, float raw_w);
    void draw_box(int classId, float conf, cv::Rect box, cv::Mat &frame);

    void generate_proposals(int stride, const float *feat, float prob_threshold, std::vector<LabeledBox> &objects);
    void process_single_output(std::vector<LabeledBox> &proposals);
    void process_multiple_outputs(std::vector<LabeledBox> &proposals);
    void apply_nms_and_get_detections(const std::vector<LabeledBox> &proposals, std::vector<LabeledBox> &detections);

    static float sigmoid(float x);
};
