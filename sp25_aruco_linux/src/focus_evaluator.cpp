#include "focus_evaluator.hpp"



double FocusEvaluator::compute(
    const cv::Mat& image
)
{

    cv::Mat gray;


    if(image.channels()==3)
    {
        cv::cvtColor(
            image,
            gray,
            cv::COLOR_BGR2GRAY
        );
    }
    else
    {
        gray=image;
    }



    cv::Mat lap;


    cv::Laplacian(
        gray,
        lap,
        CV_64F
    );



    cv::Scalar mean;
    cv::Scalar stddev;



    cv::meanStdDev(
        lap,
        mean,
        stddev
    );



    return stddev[0]*stddev[0];

}