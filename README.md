# EKF-SLAM

Simultaneous Localization and Mapping (SLAM) algorithm using an Extended Kalman Filter (EKF) using a dataset gathered with a real Turtlebot.

Simultaneous Localization and mapping is a concept that in real life is used whenever a new place is
visited. What a robot does while executing a SLAM algorithm is composed of two parts: build a map and
localize itself into it. A map can contain multiple types of information.

This EKF-SLAM is based in two different sensors. The first one are the encoders of the wheels of the
Turtlebot which will give us information of the movement of the robot (odometry). The second one is
the Kineckt sensor which allows the robot to sense the environment (walls, obstacles, doors, etc.) and map the features.
