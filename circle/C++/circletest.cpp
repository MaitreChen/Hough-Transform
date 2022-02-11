#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
using namespace cv;
using namespace std;

int main()
{
	Mat src, gray, dst;
	src = imread("images/01.jpg");
	if (src.empty())
	{
		printf("can not load image \n");
		return -1;
	}
	cvtColor(src, gray, CV_RGB2GRAY);
	medianBlur(gray, dst , 7);
	//imshow("src", src);
	//imshow("gray", gray);
	//imshow("dst", dst);


	vector<Vec3f> circles;
	HoughCircles(dst, circles, CV_HOUGH_GRADIENT, 1, 120, 100, 30, 30, 220);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3f c = circles[i];
		circle(src, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 3, CV_AA);
		circle(src, Point(c[0], c[1]), 2, Scalar(0, 0, 255), 2);
		cout << "a = " << c[0] << "  " << "b = " << c[1] << "  " << "r = " << c[2] << endl;
	}
	imshow("output", src);
	waitKey(99999);

	return 0;
}
