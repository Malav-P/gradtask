Extended Kalman Filter for 3D Position and Velocity Tracking

This project runs an Extended Kalman Filter to track the position and velocity of a moving object in 3D space. It uses range-only measurements from fixed observer points to estimate the state of the object over time.

The filter estimates both the position and velocity in the x y and z directions.

What This Code Does

Estimates the state of a moving object over time

The state includes position in x y and z

The state also includes velocity in x y and z

Simulates a real object moving with acceleration

Uses noisy range measurements from observer points to update the estimate

Prints the final estimated position and velocity

Stores all estimates and covariances at each time step

Main Files

ekftest2.py: This is the main file. It includes everything: the EKF function and a test example

You can run this file directly to see the filter working on simulated data

State Vector

The state vector has six elements

x position

y position

z position

x velocity

y velocity

z velocity

This means the state is a six element array

What the EKF Tracks

Where the object is at each time step

How fast the object is moving

It corrects the prediction using noisy measurements

Assumptions

The object moves with constant acceleration between steps

Acceleration values are provided as input

Observer points are fixed in space and do not move

Observers can measure distance to the object

The measurements are noisy

EKF Inputs

You must give the function the following things

Initial state guess like where you think the object starts and how fast it is going

Initial covariance matrix which tells how sure you are about that guess

Process noise covariance which controls how much you trust your motion model

Measurement noise covariance which controls how noisy your sensor is

Control array which contains the acceleration values at each time step

Time step which is the amount of time between updates

Observer positions which are the fixed locations of your sensors

Measurement array which contains the observed distances at each step

EKF Outputs

The function gives you two lists

One list of all estimated states at every time step

One list of all covariance matrices at every time step

You can use these to plot errors or check how confident the filter is

Inside the EKF Function
Predict Step

Uses the motion model to predict the next state

Updates the covariance matrix using the process noise

Update Step

Checks if there are any measurements

If there are valid measurements it compares expected ranges to actual ranges

Computes the Kalman Gain

Updates the state using the difference between expected and actual

Updates the covariance using the Kalman Gain

Measurement Model

Observers only measure range

Range is the straight line distance from observer to the object

The measurement function uses the x y and z position only

Velocity does not directly affect the measurement

Control Model

The object moves with a given acceleration in x y and z

This is modeled with basic physics equations

The control affects the state through a simple matrix
