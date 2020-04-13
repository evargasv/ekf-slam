# EKF-SLAM

Simultaneous Localization and Mapping (SLAM) algorithm using an Extended Kalman Filter (EKF) applied in a dataset gathered with a real Turtlebot.

Simultaneous Localization and mapping is used whenever a new place is visited. When a robot executes a SLAM algorithm, it builds a map while trying to localise itself into it. A map can contain multiple types of information.

This EKF-SLAM is based in two different sensors. The first one are the encoders of the wheels of the
Turtlebot which will give us information of the movement of the robot (odometry). The second one is
the Kinect sensor which allows the robot to sense the environment (walls, obstacles, doors, etc.) and map the features.
