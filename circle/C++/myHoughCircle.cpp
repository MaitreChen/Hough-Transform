#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;


//创建a,b列表
void create_table(int num_a_cir, int num_b_cir, double a_min, double b_min, double step, vector<double>& tabA, vector<double>& tabB) {
	double a = a_min;
	double b = b_min;
	tabA.resize(num_a_cir);
	tabB.resize(num_b_cir);
	for (int i = 0; i < num_a_cir; i++) {
		tabA[i] = (double)a;
		a += step;
	}
	for (int j = 0; j < num_b_cir; j++) {
		tabB[j] = (double)b;
		b += step;
	}
}

vector<Vec3f> myHoughCircle(const Mat& img, double rho, double theta, int threshold, double min_theta = 0, double max_theta = CV_PI) {
	const int height = img.rows;
	const int width = img.cols;

	const int r_min = 0;
	const int r_max = round(sqrt(pow(width - 1, 2) + pow(height - 1, 2))) + 1;
	const int num_a_cir = width;
	const int num_b_cir = height;
	const int numrhos = r_max - r_min;


	//建立累加器
	vector<vector<vector<int>>>accum;
	for (int i = 0; i < num_a_cir; i++) {
		vector<vector<int>>temp1;
		for (int j = 0; j < num_b_cir; j++) {
			vector<int>temp2;
			for (int k = 0; k < numrhos; k++) {
				temp2.push_back(0);
			}
			temp1.push_back(temp2);
		}
		accum.push_back(temp1);
	}

	vector<double>tabA, tabB;
	create_table(num_a_cir, num_b_cir, 0, 0, 1, tabA, tabB);


	//统计累加器的值
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			if (img.ptr<uchar>(x)[y] != 0) {
				for (int a = 0; a < num_a_cir; a++) {
					for (int b = 0; b < num_b_cir; b++) {
						int r = cvRound(sqrt(pow(x - tabA[a], 2) + pow(y - tabB[b], 2)) / rho);
						accum[a][b][r]++;
					}
				}
			}
		}
	}


	//将符合要求的值添加到容器中
	vector<Vec3f>circles;
	for (int i = 0; i < num_a_cir; i++) {
		for (int j = 0; j < num_a_cir; j++) {
			for (int k = 0; k < numrhos; k++) {
				Vec3f circle;
				int a = accum[i][j][k];
				if (a > threshold) {
					//对圆心查找局部最大值
					int m1 = accum[i][j - 1][k];
					int m2 = accum[i][j + 1][k];
					int m3 = accum[i - 1][j][k];
					int m4 = accum[i + 1][j][k];
					if (a > m1 && a > m2 && a > m3 && a > m4) {
						circle[0] = i;
						circle[1] = j;
						circle[2] = k;
						circles.push_back(circle);
					}
				}
			}
		}
	}
	return circles;
}

int main() {
	Mat src, gray, edge;
	src = imread("images/01.jpg");
	imshow("src", src);
	if (src.data == 0) {
		cout << "failed to load!" << endl;
	}
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Canny(gray, edge, 50, 200, 5);

	vector<Vec3f> circles;
	double rho = 1;
	double theta = CV_PI / 180;
	double threshold = 170;
	circles = myHoughCircle(edge, rho, theta, threshold);

	cout << "the amount of circles is " << circles.size() << endl;
	for (int i = 0; i < circles.size(); i++) {
		Vec3f c = circles[i];
		circle(src, Point(c[0], c[1]), c[2], Scalar(0, 0, 255), 3, CV_AA);
		circle(src, Point(c[0], c[1]), 2, Scalar(0, 0, 255), 2);
		cout << "a = " << c[0] << "  " << "b = " << c[1] << "  " << "r = " << c[2] << endl;
	}
	imshow("circleDetection", src);
	waitKey(9999);

	return 0;
}