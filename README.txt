run everything in docker container
to enter the docker container use command:
docker-join.bash -n gsam
and enter "/ws/Grounded-Segment-Anything"

use 

python3 demo_3cams_default.py
python3 demo_3cams_igev.py

to get point clouds of objects(mugs, microwave door)
to get the concanated point cloud

in local host,
use 
"
source catkin_ws/devel/setup.bash
roslaunch pointcloud_concatenate gsam.launch

the final topic is "/points_concatenated" 
