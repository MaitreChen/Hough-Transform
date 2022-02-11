#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <iostream>  

using namespace cv;
using namespace std;


int main()
{
	Mat src, gray, edge;
	src = imread("C:/Users/19749/Desktop/Hough-Transform/line/images/01.jpg");
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Canny(gray, edge, 50, 200, 3);

	vector<Vec2f> lines;  
	HoughLines(edge, lines, 1, CV_PI / 180, 160, 0, 0);
	
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(src, pt1, pt2, Scalar(255, 0,0), 2, CV_AA);
	}
	

	imshow("src", src);
	imshow("edge", edge);
	waitKey(0);
	destroyAllWindows();

	return 0;
}