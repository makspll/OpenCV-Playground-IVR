#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import math 

import os.path
class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
                # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send messages to a topic named image_topic
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=1)
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joints_pub = rospy.Publisher("joints_pos", Float64MultiArray, queue_size=10)
        # initialize a subscriber to receive messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub = rospy.Subscriber("/robot/camera1/image_raw", Image, self.callback)
        #(for lab3)
        # initialize a publisher to send robot end-effector position
        self.end_effector_pub = rospy.Publisher("end_effector_prediction", Float64MultiArray, queue_size=10)
        # initialize a publisher to send desired trajectory
        self.trajectory_pub = rospy.Publisher("trajectory", Float64MultiArray, queue_size=10)
        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        # record the beginning time
        self.time_trajectory = rospy.get_time()

    # Define a circular trajectory (for lab 3)
    def trajectory(self):
        # get current time
        cur_time = np.array([rospy.get_time() - self.time_trajectory])
        x_d = float(6 * np.cos(cur_time * np.pi / 100))
        y_d = float(6 + np.absolute(1.5 * np.sin(cur_time * np.pi / 100)))
        return np.array([x_d, y_d])

    # Receive data, process it, and publish
    def callback(self, data):
        # Receive the image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        h,w,c = cv_image.shape



        # loading template for links as binary image (used in lab 2)
        self.link1 = cv2.inRange(cv2.imread('link1.png', 1), (200, 200, 200), (255, 255, 255))
        self.link2 = cv2.inRange(cv2.imread('link2.png', 1), (200, 200, 200), (255, 255, 255))
        self.link3 = cv2.inRange(cv2.imread('link3.png', 1), (200, 200, 200), (255, 255, 255))

        # Perform image processing task (your code goes here)

        #BGR

        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        coms,comsImage,total_mask = threshold_joints(hsv_image)
        link_mask = threshold_links(hsv_image)
        total_mask = cv2.bitwise_or(link_mask,total_mask)
        
        # change te value of self.joint.data to your estimated value from thew images once you have finalized the code
        self.joints = Float64MultiArray()

        yellow_to_blue= coms[2] - coms[3]
        blue_to_green = coms[0] - coms[2]
        green_to_red = coms[1] - coms[0]

        # self.joints.data = np.arctan2([
        #                                 yellow_to_blue[0],
        #                                 blue_to_green[0],
        #                                 green_to_red[0]],
        #                             [
        #                                 yellow_to_blue[1],
        #                                 blue_to_green[1],
        #                                 green_to_red[1]
        #                                 ])

        yellow_to_blue_image = comsImage[2] - comsImage[3]
        blue_to_green_image = comsImage[0] - comsImage[2]
        green_to_red_image = comsImage[1] - comsImage[0]        
        
        link_centers = [comsImage[3] + yellow_to_blue_image/2,
                        comsImage[2] + blue_to_green_image/2,
                        comsImage[0] + green_to_red_image/2]
    
        link_1_shape = self.link1.shape
        link_2_shape = self.link2.shape 
        link_3_shape = self.link3.shape

        link_1_crop,link_2_crop,link_3_crop = [ crop_rectangle(link_mask,lc,ls) for lc,ls in zip(link_centers,[link_1_shape,link_2_shape,link_3_shape])]


        directions = [yellow_to_blue_image,blue_to_green_image,green_to_red_image]
        idx = 0
        chamfer_rotations = []
        for (chamfer,link) in zip([self.link1,self.link2,self.link3],[link_1_crop,link_2_crop,link_3_crop]):
            angTurn = 1

            phaseAngle = abs(np.arctan2(-directions[idx][1],directions[idx][0]))
            angStart = -90
            angMax = 90
            q = 0

            if(phaseAngle < math.pi * (1/4)): # q1
                angStart = -90
                angMax = -45
                q=1
            elif (phaseAngle < math.pi * (1/2)): #q 2
                angStart = -45
                angMax = 0
                q=2
            elif (phaseAngle < math.pi * (3/4)): #q3
                angStart = 0
                angMax = 45
                q=3
            else: #q4
                angStart = 45
                angMax = 90
                q=4


            minSum = math.inf
            minAng = math.inf
            currAng = angStart

            while(currAng <= angMax):
                currSum = chamfer_match_link(link,chamfer,currAng)
                if(currSum < minSum):
                    minSum = currSum
                    minAng = currAng
                currAng += angTurn

            chamfer_rotations.append(minAng)
            idx += 1
            

        self.joints.data = [ chamfer_rotations[0],
                            chamfer_rotations[1] - chamfer_rotations[0],
                            chamfer_rotations[2] - chamfer_rotations[1]]

        print(self.joints.data)
        # The image is loaded as cv_imag

        cv2.imshow('window', link_1_crop)
        cv2.imshow('window2', link_2_crop)
        cv2.imshow('window3', link_3_crop)

        cv2.waitKey(3)

        # publish the estimated position of robot end-effector (for lab 3)
        # x_e_image = np.array([0, 0, 0])
        # self.end_effector=Float64MultiArray()
        # self.end_effector.data= x_e_image

        # send control commands to joints (for lab 3)
        # self.joint1=Float64()
        # self.joint1.data= q_d[0]
        # self.joint2=Float64()
        # self.joint2.data= q_d[1]
        # self.joint3=Float64()
        # self.joint3.data= q_d[2]

        # Publishing the desired trajectory on a topic named trajectory(for lab 3)
        # x_d = self.trajectory()    # getting the desired trajectory
        # self.trajectory_desired= Float64MultiArray()
        # self.trajectory_desired.data=x_d

        # Publish the results - the images are published under a topic named "image_topic" and calculated joints angles are published under a topic named "joints_pos"
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.joints_pub.publish(self.joints)
            # (for lab 3)
            # self.trajectory_pub.publish(self.trajectory_desired)
            # self.end_effector_pub.publish(self.end_effector)
            # self.robot_joint1_pub.publish(self.joint1)
            # self.robot_joint2_pub.publish(self.joint2)
            # self.robot_joint3_pub.publish(self.joint3)
        except CvBridgeError as e:
            print(e)

def chamfer_match_link(link,chamfer,rotation):
    h,w = link.shape
    M = cv2.getRotationMatrix2D((w/2,h/2),rotation,1)


    rotated_chamfer = cv2.warpAffine(chamfer,M,(w,h))

    dst_map	= cv2.distanceTransform(cv2.bitwise_not(link),cv2.DIST_L2,0)
    sum_dists = (rotated_chamfer * dst_map).sum()
    return sum_dists     

    

def crop_rectangle(img, center, dims):
    x1 = int(center[0] - (dims[1] / 2))
    x2 = int(center[0] + (dims[1] / 2))

    y1 = int(center[1] - (dims[0] / 2))
    y2 = int(center[1] + (dims[0] / 2))


    return img[y1:y2,x1:x2]

def threshold_links(hsv_image):

    link_mask = threshold_binary_hue(hsv_image,[0,0,0],[179,255,10])

    return link_mask

def threshold_joints(hsv_image):
    h,w,c = hsv_image.shape

    bgr_planes = cv2.split(hsv_image)

    plot_hist_if_not_present(hsv_image,bgr_planes)

    blob_color_mins = [hue360to179(100),hue360to179(-20),hue360to179(220),hue360to179(40)]
    blob_saturation_mins = [127.5,127.5,127.5,127.5]
    blob_intensity_mins = [0,0,0,0]
    

    blob_color_maxs = [hue360to179(140),hue360to179(20),hue360to179(260),hue360to179(80)]
    blob_saturation_maxs = [255,255,255,255]
    blob_intensity_maxs = [255,255,255,255]

    green_mask = threshold_binary_hue(hsv_image,[blob_color_mins[0],blob_saturation_mins[0],blob_intensity_mins[0]],[blob_color_maxs[0],blob_saturation_maxs[0],blob_intensity_maxs[0]])
    red_mask = threshold_binary_hue(hsv_image,[blob_color_mins[1],blob_saturation_mins[1],blob_intensity_mins[1]],[blob_color_maxs[1],blob_saturation_maxs[1],blob_intensity_maxs[1]])
    blue_mask = threshold_binary_hue(hsv_image,[blob_color_mins[2],blob_saturation_mins[2],blob_intensity_mins[2]],[blob_color_maxs[2],blob_saturation_maxs[2],blob_intensity_maxs[2]])
    yellow_mask = threshold_binary_hue(hsv_image,[blob_color_mins[3],blob_saturation_mins[3],blob_intensity_mins[3]],[blob_color_maxs[3],blob_saturation_maxs[3],blob_intensity_maxs[3]])
    
    cv2.imwrite("green_mask.png",green_mask)
    cv2.imwrite("red_mask.png",red_mask)
    cv2.imwrite("blue_mask.png",blue_mask)
    cv2.imwrite("yellow_mask.png",yellow_mask)
    
    red_mask,green_mask,blue_mask,yellow_mask = [
        cv2.morphologyEx(m,
        cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize)) for m,ksize in  
        zip([red_mask,green_mask,blue_mask,yellow_mask],[(8,8),(10,10),(20,20),(30,30)])]




    total_mask = cv2.bitwise_or(green_mask,red_mask)
    total_mask = cv2.bitwise_or(total_mask,blue_mask)
    total_mask = cv2.bitwise_or(total_mask,yellow_mask)

    coms = np.zeros((4,2))
    comsImage = np.zeros((4,2))

    idx = 0
    for m in [green_mask,red_mask,blue_mask,yellow_mask]:
        contours,hierarchy = cv2.findContours(m, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)

        cx =int(M['m10']/M['m00']) 
        cy =int(M['m01']/M['m00'])
        
        total_mask = cv2.drawMarker(total_mask, (cx,cy),(80,255,255),markerSize=60,thickness=2)

        coms[idx,0] = int(cx- w/2)
        coms[idx,1] = int(cy - h/2) * -1

        comsImage[idx,0] = cx
        comsImage[idx,1] = cy

        idx = idx + 1

    scaleFactor = np.linalg.norm(coms[1] - coms[3]) / 9

    coms *= 1/scaleFactor

    return (coms,comsImage,total_mask)

def hue360to179(hue):
    return int((hue / 360) * 179)

def threshold_binary_hue(img,mins,maxs):
    h,w,c = img.shape
    mask = np.ones((h,w), np.uint8) * 255
    
    idx = 0
    for minV,maxV in zip(mins,maxs):
        mask_ranges_hsi = mask_ranges(minV,maxV, 179 if idx == 0 else 255)

        mask2 = np.ones((h,w),np.uint8) * 0
        for mask_range in mask_ranges_hsi:
            if(idx == 0):
                maskHue = cv2.inRange(img,(int(mask_range[0]),0,0),(int(mask_range[1]),255,255))
                mask2 = cv2.bitwise_or(mask2,maskHue)
            elif(idx == 1):
                maskSat = cv2.inRange(img,(0,int(mask_range[0]),0),(179,int(mask_range[1]),255))
                mask2 = cv2.bitwise_or(mask2,maskSat)
            elif(idx == 2):
                maskInt = cv2.inRange(img,(0,0,int(mask_range[0])),(179,255,int(mask_range[1])))
                mask2 = cv2.bitwise_or(mask2,maskInt)
        mask = cv2.bitwise_and(mask2,mask)

        idx = idx + 1
    return mask

def mask_ranges(minval,maxval,limit):
    ranges = []
    minMain,maxMain = (max(minval,0), min(maxval,limit))
    ranges.append((minMain,maxMain))

    if maxval > limit:
        minTop,maxTop = (0, maxval - (limit + 1))
        ranges.append((minTop,maxTop))
    if minval < 0:
        minBot,maxBot = ((limit + 1) - (0 - minval),limit)
        ranges.append((minBot,maxBot))

    return ranges

def plot_hist_if_not_present(img,img_split):
    hist_path = "hist.png"

    if(not os.path.exists(hist_path)):

        histSize = 256
        histRange = (0, 256) # the upper boundary is exclusive
        accumulate = False
        
        fig,ax = plt.subplots(1,3)
        b_hist = cv2.calcHist(img_split, [0], None, [histSize], histRange, accumulate=accumulate)[:,0]
        g_hist = cv2.calcHist(img_split, [1], None, [histSize], histRange, accumulate=accumulate)[:,0]
        r_hist = cv2.calcHist(img_split, [2], None, [histSize], histRange, accumulate=accumulate)[:,0]
        x = np.linspace(histRange[0],histRange[1],histSize)

        # b_hist = (b_hist - b_hist.mean())
        # remove gray effect
        b_hist,g_hist,r_hist = [np.clip(x,0,2000) for x in [b_hist,g_hist,r_hist]]

        for i in range (0,3):
            ax[i].plot(x,[b_hist,g_hist,r_hist][i],["--b","--g","--r"][i])

            # ax[i].set_ylim(0,1)
            ax[i].set_xlim(0,255)
            ax[i].legend(["hue","sat","int"][i])
            ax[i].set_xticks(np.arange(histRange[0],histRange[1],step=10))

        fig.tight_layout()
        fig.set_size_inches(30,6)
        plt.savefig(hist_path)

# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
