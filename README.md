# EECE5550_MobileRobotics

Clone the repo into your ROS workspace (i.e. `~/catkin_ws/src`).
```
git clone --recursive https://github.com/piraka9011/EECE5550_MobileRobotics.git
```
Install all the python modules needed (you might need to use `sudo`)
```
pip install -r requirements.txt
```
The `color_node.py` requires [opencv](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html).

### Apriltags

Navigate to the `apriltags2_ros` repo and change the `tags.yaml` file in the `config` folder.

```
cd apriltags2_ros/config
gedit tags.yaml
```
Add a tag based on the one you printed and size in meters to the `standalone_tags` array. Name can be anything you want:
```json
standalone_tags:
  [
     {id: 13, size: 0.171, name: 'object'}
  ]
```
Launch using continuous detection:
```
roslaunch apriltags2_ros continuous_detection.launch
```