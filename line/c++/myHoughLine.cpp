#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;


//创建sin，cos列表
void createTrigTable(int numangle, double min_theta, double theta_step, vector<double>& tabSin, vector<double>& tabCos) {
	double angle = min_theta;
	tabSin.resize(numangle);
	tabCos.resize(numangle);
	for (int i = 0; i < numangle; i++) {
		tabSin[i] = (double)sin(angle);
		tabCos[i] = (double)cos(angle);
		angle += theta_step;
	}
}
//查找局部最大值,即该值比周围上下左右的值都大
void findLocalMaximums(const Mat& accumulator, int threshold, vector<int>& m_rhos, vector<int>& m_thetas) {
	//确保i-1不小于0，所以i从1开始；
	//accumulator.rows - 1防止越界；
	for (int i = 1; i < accumulator.rows - 1; i++) {
		for (int j = 1; j < accumulator.cols - 1; j++) {
			int a = accumulator.ptr<int>(i)[j];
			if (a > threshold) {
				int left = accumulator.ptr<int>(i)[j - 1];
				int right = accumulator.ptr<int>(i)[j + 1];
				int top = accumulator.ptr<int>(i + 1)[j];
				int bottom = accumulator.ptr<int>(i - 1)[j];
				if (a > left && a >= right && a > top && a >= bottom) {//将局部最大值添加到对应容器中
					m_rhos.push_back(i);
					m_thetas.push_back(j);
				}
			}
		}
	}
}

vector<Vec2f> myHoughLine(const Mat& img, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI) {
	int width = img.cols;
	int height = img.rows;

	const int max_rho = round(sqrt(pow(width - 1, 2) + pow(height - 1, 2))) + 1;
	const int min_rho = -max_rho;
	const int numangle = cvRound((max_theta - min_theta) / theta);
	const int numrho = cvRound((max_rho - min_rho + 1) / rho);

	//建立累加空间并初始化
	Mat accumulator = Mat::zeros(numangle + 2, numrho + 2, CV_32SC1);
	//定义矩阵rhos与thetas，保存相应矩阵网格对应的极值与角度值
	Mat rhos = Mat::zeros(numangle + 2, numrho + 2, CV_32FC1);
	Mat thetas = Mat::zeros(numangle + 2, numrho + 2, CV_32FC1);

	vector<double>sinTable, cosTable;
	createTrigTable(numangle, min_theta, theta, sinTable, cosTable);

	//统计累加器的值
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			if (img.ptr<uchar>(x)[y] != 0) {
				for (int n = 0; n < numangle; n++) {
					int r = cvRound((y * cosTable[n] + x * sinTable[n]) / rho);
					r += (numrho - 1) / 2;
					accumulator.ptr<int>(n + 1)[r + 1]++;
					thetas.ptr<double>(n + 1)[r + 1] = (double)min_theta + n * theta;
					rhos.ptr<double>(n + 1)[r + 1] = cvRound(y * cosTable[n] + x * sinTable[n]);
				}
			}
		}
		//累加器可视化
		//Mat accum;
		//accumulator.convertTo(accum, CV_8UC1);
		//imshow("accum", accum);

	}
	//按阈值把累加值大于阈值且是局部最大的网格找出来,
	//局部最大值网格的值比前后的值都大；
	vector<int>m_rhos, m_thetas;
	findLocalMaximums(accumulator, threshold, m_rhos, m_thetas);

	//将符合要求的rho和theta放到列表中；
	vector<Vec2f>lines;
	for (int i = 0; i < m_rhos.size(); i++) {
		Vec2f line;
		line[0] = rhos.ptr<double>(m_rhos[i])[m_thetas[i]];
		//cout << m_rhos[i] << m_thetas[i];
		line[1] = thetas.ptr<double>(m_rhos[i])[m_thetas[i]];
		lines.push_back(line);
	}
	return lines;
}

int main() {
	Mat src, gray, edge;
	src = imread("images/01.jpg");
	imshow("src", src);

	cvtColor(src, gray, COLOR_BGR2GRAY);
	Canny(gray, edge, 50, 200, 5);
	//imshow("gray", gray);
	//imshow("canny", edge);

	vector<Vec2f> lines;
	double rho = 1;
	double theta = CV_PI / 180;
	double threshold = 150;
	lines = myHoughLine(edge, rho, theta, threshold);

	for (int i = 0; i < lines.size(); i++) {
		double rrho = lines[i][0], ttheta = lines[i][1];
		Point p1, p2;
		double a = cos(ttheta), b = sin(ttheta);
		double x0 = a * rrho, y0 = b * rrho;
		p1.x = round(x0 + 1000 * (-b));
		p1.y = round(y0 + 1000 * a);
		p2.x = round(x0 - 1000 * (-b));
		p2.y = round(y0 - 1000 * a);
		line(src, p1, p2, Scalar(255, 0, 0), 2, LINE_AA);
	}

	imshow("houghDetection", src);
	waitKey(0);
	destroyAllWindows();

	return 0;
}