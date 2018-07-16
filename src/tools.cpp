#include <assert.h>
#include <iostream>
#include <math.h>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
	rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  assert(estimations.size() > 0);
  //  * the estimation vector size should equal ground truth vector size
  assert(estimations.size() == ground_truth.size());

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    // ... your code here
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array().square();
    rmse += residual;
  }

  //calculate the mean
  rmse /= estimations.size();
  //calculate the squared root
  rmse = rmse.cwiseSqrt();
  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float mag = pow(px, 2) + pow(py, 2);
  //check division by zero
  if (mag != 0) {
    float mag_sqrt = pow(mag, 0.5);
    //compute the Jacobian matrix
    Hj << px / mag_sqrt, py / mag_sqrt, 0, 0,
          -py / mag, px / mag, 0, 0,
          py * (vx * py - vy * px) / pow(mag_sqrt, 3),
          px * (vy * px - vx * py) / pow(mag_sqrt, 3),
          px / mag_sqrt, py / mag_sqrt;
	}

	return Hj;
}
