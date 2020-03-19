from argparse import ArgumentParser
import airsimneurips as airsim
import cv2
import threading
import time
import utils
import numpy as np

import tensorflow as tf
import random
import math
import os

# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(self, drone_name = "drone_1", plot_transform=True, viz_traj=True, viz_image_cv2=True):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.plot_transform = plot_transform
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands 
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03))
        self.odometry_callback_thread = threading.Thread(target=self.repeat_timer_odometry_callback, args=(self.odometry_callback, 0.02))
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False

    # loads desired level
    def load_level(self, level_name, sleep_sec = 2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection() # failsafe
        time.sleep(sleep_sec) # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):
        self.airsim_client.simStartRace(tier)

    # # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        # self.airsim_client.simResetRace()
        # self.airsim_client.simPause(True)
        self.airsim_client.reset()
        self.airsim_client.enableApiControl(vehicle_name="drone_1")
        self.airsim_client.arm(vehicle_name="drone_1")
        # self.airsim_client.simPause(False) # unpause sim to simresetrace works as it's supposed to
        self.airsim_client.simResetRace()
        # self.airsim_client.simStartRace(tier=1)

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains 
        traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0, 
                                                            kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0, 
                                                            kp_along_track = 0.4, kd_along_track = 0.0, 
                                                            kp_vel_along_track = 0.04, kd_vel_along_track = 0.0, 
                                                            kp_z_track = 2.0, kd_z_track = 0.0, 
                                                            kp_vel_z = 0.4, kd_vel_z = 0.0, 
                                                            kp_yaw = 3.0, kd_yaw = 0.1)

        self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        time.sleep(0.2)

    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height = 2.0):
        if self.level_name == "ZhangJiaJie_Medium":
            takeoff_height = 1.0

        # time.sleep(1)

        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position

        start_position.x_val += randf(0, 1)
        start_position.y_val += randf(-1, 0)
        # start_position.z_val += randf(0, 1)

        # start_position.x_val += -2
        # start_position.y_val += 2
        # start_position.z_val += -1

        takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val, start_position.z_val-takeoff_height)

        if(self.plot_transform):
            self.airsim_client.plot_transform([airsim.Pose(takeoff_waypoint, airsim.Quaternionr())], vehicle_name=self.drone_name)

        self.airsim_client.moveToPositionAsync(takeoff_waypoint.x_val, takeoff_waypoint.y_val, takeoff_waypoint.z_val, 1).join()

        # self.airsim_client.moveOnSplineAsync([takeoff_waypoint], vel_max=15.0, acc_max=5.0, add_position_constraint=True, 
        #     add_velocity_constraint=False, viz_traj=self.viz_traj, vehicle_name=self.drone_name).join()

    # stores gate ground truth poses as a list of airsim.Pose() objects in self.gate_poses_ground_truth
    def get_ground_truth_gate_poses(self):
        # print(self.airsim_client.simListSceneObjects())
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
        gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.gate_poses_ground_truth = [self.airsim_client.simGetObjectPose(gate_name) for gate_name in gate_names_sorted]

    # this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints() 
    # the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale = 1.0):
        import numpy as np
        # convert gate quaternion to rotation matrix. 
        # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return airsim.Vector3r(0.0, 1.0, 0.0)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
        gate_facing_vector = rotation_matrix[:,1]
        return airsim.Vector3r(scale * gate_facing_vector[0], scale * gate_facing_vector[1], scale * gate_facing_vector[2])

    def fly_through_all_gates_one_by_one_with_moveOnSpline(self):
        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0

        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy", "ZhangJiaJie_Medium"] :
            vel_max = 10.0
            acc_max = 5.0

        for gate_pose in self.gate_poses_ground_truth:
            if(self.plot_transform):
                self.airsim_client.plot_transform([gate_pose], vehicle_name=self.drone_name)

        return self.airsim_client.moveOnSplineAsync([gate_pose.position], vel_max=vel_max, acc_max=acc_max, 
            add_position_constraint=True, add_velocity_constraint=False, viz_traj=self.viz_traj, vehicle_name=self.drone_name)

    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy", "ZhangJiaJie_Medium"] :
            vel_max = 30.0
            acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0

        if(self.plot_transform):
            self.airsim_client.plot_transform(self.gate_poses_ground_truth, vehicle_name=self.drone_name)

        return self.airsim_client.moveOnSplineAsync([gate_pose.position for gate_pose in self.gate_poses_ground_truth], vel_max=30.0, acc_max=15.0, 
            add_position_constraint=True, add_velocity_constraint=False, viz_traj=self.viz_traj, vehicle_name=self.drone_name)

    def fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints(self):
        add_velocity_constraint = True

        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy"] :
            vel_max = 15.0
            acc_max = 3.0
            speed_through_gate = 2.5

        if self.level_name == "ZhangJiaJie_Medium":
            vel_max = 10.0
            acc_max = 3.0
            speed_through_gate = 1.0

        if self.level_name == "Building99_Hard":
            vel_max = 2.0
            acc_max = 0.5
            speed_through_gate = 0.5
            add_velocity_constraint = False

        for gate_pose in self.gate_poses_ground_truth:
            if(self.plot_transform):
                self.airsim_client.plot_transform([gate_pose], vehicle_name=self.drone_name)

        # scale param scales the gate facing vector by desired speed. 
        return self.airsim_client.moveOnSplineVelConstraintsAsync([gate_pose.position], 
                                                [self.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale = speed_through_gate)], 
                                                vel_max=vel_max, acc_max=acc_max, 
                                                add_curr_odom_position_constraint=True, add_velocity_constraint=add_velocity_constraint, 
                                                viz_traj=self.viz_traj, vehicle_name=self.drone_name)

    def fly_through_all_gates_at_once_with_moveOnSplineVelConstraints(self):
        if self.level_name in ["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium"]:
            vel_max = 15.0
            acc_max = 7.5
            speed_through_gate = 2.5

        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0
            speed_through_gate = 1.0

        if(self.plot_transform):
            self.airsim_client.plot_transform(self.gate_poses_ground_truth, vehicle_name=self.drone_name)

        return self.airsim_client.moveOnSplineVelConstraintsAsync([gate_pose.position for gate_pose in self.gate_poses_ground_truth], 
                [self.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale = speed_through_gate) for gate_pose in self.gate_poses_ground_truth], 
                vel_max=15.0, acc_max=7.5, 
                add_position_constraint=True, add_velocity_constraint=True, 
                viz_traj=self.viz_traj, vehicle_name=self.drone_name)

    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8) 
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)

    def odometry_callback(self):
        # get uncompressed fpv cam image
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        position = drone_state.kinematics_estimated.position 
        orientation = drone_state.kinematics_estimated.orientation
        linear_velocity = drone_state.kinematics_estimated.linear_velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity

    # call task() method every "period" seconds. 
    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_odometry_callback(self, task, period):
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

###########################################################
# reinforcement part 
###########################################################
# Parameters
epsilon = 1  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
epsilonMinimumValue = 0.01 # The minimum value we want epsilon to reach in training. (0 to 1)

# global position waypoint
nbActions = 5 # vx, vz, yaw 


epoch = 1000 # The number of games we want the system to run for.
hiddenSize = 500 # Number of neurons in the hidden layers.
maxMemory = 1000000 # How large should the memory be (where it stores its past experiences).
batchSize = 32 # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.

grid_size = 5*60
# State_size = (13) * gate_gap # plus gate pose

State_size = 7 * grid_size

discount = 0.99 # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)  
learningRate = 0.2 # Learning Rate for Stochastic Gradient Descent (our optimizer).




def Q_Net(input_state):
    with tf.variable_scope(name_or_scope = "Train") as scope:
        Train_W1 = tf.Variable(tf.truncated_normal([State_size, hiddenSize], stddev=1.0 / math.sqrt(float(State_size))))
        Train_b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))  
        
        Train_W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize],stddev=1.0 / math.sqrt(float(hiddenSize))))
        Train_b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
        
        Train_W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions],stddev=1.0 / math.sqrt(float(hiddenSize))))
        Train_b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
        
    input_layer = tf.nn.relu(tf.matmul(input_state, Train_W1) + Train_b1)
    hidden_layer = tf.nn.relu(tf.matmul(input_layer, Train_W2) + Train_b2)
    output_layer = tf.add(tf.matmul(hidden_layer, Train_W3), Train_b3, name = "Q")
    return output_layer

def Q_Target_Net(input_state):
    with tf.variable_scope(name_or_scope = "Target") as scope:
        Target_W1 = tf.Variable(tf.truncated_normal([State_size, hiddenSize], stddev=1.0 / math.sqrt(float(State_size))))
        Target_b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))  
        
        Target_W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize],stddev=1.0 / math.sqrt(float(hiddenSize))))
        Target_b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
        
        Target_W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions],stddev=1.0 / math.sqrt(float(hiddenSize))))
        Target_b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
        
    input_layer = tf.nn.relu(tf.matmul(input_state, Target_W1) + Target_b1)
    hidden_layer = tf.nn.relu(tf.matmul(input_layer, Target_W2) + Target_b2)
    output_layer = tf.matmul(hidden_layer, Target_W3) + Target_b3
    return output_layer


def update_target():
    op_holder = []
    
    src_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Train") 
    dest_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope="Target") 
    
    for src_var, dest_var in zip(src_vars, dest_vars):
         op_holder.append(dest_var.assign(src_var.value())) 
    
    return op_holder



g = tf.Graph()
with g.as_default():
    # Input
    X = tf.placeholder(tf.float32, [None, State_size], name = "X")
    
    # True labels
    Y = tf.placeholder(tf.float32, [None, nbActions])

    Q = Q_Net(X)
    Target_Q = Q_Target_Net(X)

    # Mean squared error cost function
    # train_cost = tf.clip_by_value(tf.reduce_sum(tf.square(Y-Q_Net(X))) / (2*batchSize), 0, 100)
    train_cost = tf.clip_by_value(tf.reduce_sum(tf.square(Y-Q_Net(X))) / (2*batchSize), 0, 10000000)
    
    t_vars = tf.trainable_variables() # return list
    train_vars = [var for var in t_vars if "Train" in var.name]

    # Stochastic Gradient Decent Optimizer
    train_optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(train_cost, var_list = train_vars)

    saver = tf.train.Saver()



# Helper function: Chooses a random value between the two boundaries.
def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s

# Helper function: distance two position
def dis(pose1, pose2):
    return math.sqrt ( (pose1.x_val-pose2.x_val)**2 + (pose1.y_val-pose2.y_val)**2 + (pose1.z_val-pose2.z_val)**2 )




class Env():
    def __init__(self, baseline_racer):
        self.state = np.empty(State_size, dtype = np.uint8) 

    def getState(self, baseline_racer):
        self.opponent_pose = baseline_racer.airsim_client.simGetVehiclePose("drone_2").position # drone_2 is opponent drone
        self.my_pose = baseline_racer.airsim_client.simGetVehiclePose("drone_1").position
        self.my_ori = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").orientation

        drone_heading_vec = baseline_racer.get_gate_facing_vector_from_quaternion(baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").orientation, scale = 1)
        gate_heading_vec = baseline_racer.get_gate_facing_vector_from_quaternion(self.gate_pose[self.next_gate_idx].orientation, scale = 1)
        gate_list = self.gate_pose_array[3*self.next_gate_idx:3*(self.next_gate_idx + 1)]
        if(gate_list[0] != gate_list[0]):
            time.sleep(1)
            gate_list = self.gate_pose_array[3*self.next_gate_idx:3*(self.next_gate_idx + 1)]


        g_min = -10
        g_max = 50
        g_mul = grid_size / (g_max - g_min)
        # -10 ~ 45 => -10 ~ 50 => -50 ~ 250 => 0 ~ 300
        drone_vec = [drone_heading_vec.y_val, -drone_heading_vec.x_val, 0]
        gate_vec = [gate_heading_vec.x_val, gate_heading_vec.y_val, 0]

        # rot_angle => -3.14 ~ 3.14 => 0 ~ 300
        rot_angle = math.acos(np.inner(drone_vec, gate_vec))
        if ((gate_vec[0] * drone_vec[1] - gate_vec[1] * drone_vec[0]) < 0):
            rot_angle = -rot_angle

        pose_list = [self.my_pose.x_val, self.my_pose.y_val, self.my_pose.z_val] + gate_list

        ## make grid world!
        state_output = np.zeros((len(pose_list) + 1, grid_size))

        # print(pose_list)

        for i in range(len(pose_list)):
            if (round(g_mul*(pose_list[i] - g_min)) < 0):
                tmp = 0
            elif (round(g_mul*(pose_list[i] - g_min)) > grid_size - 1):
                tmp = grid_size - 1
            else:
                tmp = round(g_mul*(pose_list[i] - g_min))

            state_output[i, int(tmp)] = 1

        ## Add rotation angle
        rot_angle = round((rot_angle + math.pi) * (grid_size / 2 / math.pi))
        if (rot_angle > grid_size -1):
            rot_angle = grid_size
        
        state_output[-1, rot_angle] = 1

        state_output = np.reshape(state_output, (-1, State_size))
        return state_output  

    def start(self, baseline_racer):
        # input state = our drone pose (x, y, z) + opponent drone pose (x, y, z)
        baseline_racer.start_race(args.race_tier)
        baseline_racer.initialize_drone()
        baseline_racer.takeoff_with_moveOnSpline()
        baseline_racer.get_ground_truth_gate_poses()
        
        self.gate_pose = baseline_racer.gate_poses_ground_truth

        self.my_pose = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position
        self.opponent_pose = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_2").position # drone_2 is opponent drone
        self.my_ori = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").orientation
        
        self.next_gate_idx = 0
        self.last_gate_passed_idx = -1
        self.gate_passed_thresh = 1.5 # threshold value
        
        self.isPass = 0

        self.gate_pose_array = []

        for gate_pose in self.gate_pose:
            self.gate_pose_array.append(gate_pose.position.x_val)
            self.gate_pose_array.append(gate_pose.position.y_val)
            self.gate_pose_array.append(gate_pose.position.z_val)
        
        self.start_time = time.time()
        self.prev_dis_drone_gate = dis(self.my_pose, self.gate_pose[self.next_gate_idx].position)


    
    def reset(self, baseline_racer):
        baseline_racer.reset_race()
    

    def isPassGate(self, baseline_racer):
        self.curr_position_linear = baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position

        dist_from_next_gate = math.sqrt( (self.curr_position_linear.x_val - self.gate_pose[self.next_gate_idx].position.x_val)**2
                                        + (self.curr_position_linear.y_val - self.gate_pose[self.next_gate_idx].position.y_val)**2
                                        + (self.curr_position_linear.z_val- self.gate_pose[self.next_gate_idx].position.z_val)**2)

        if(dist_from_next_gate < self.gate_passed_thresh):
            self.last_gate_passed_idx += 1
            self.next_gate_idx += 1
            return True
        return False
    
    def getReward(self, baseline_racer):
        reward = 0
        ## drone-drone collision
        if(baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").position.z_val > 3):
            reward -= 1
        
        # if(dis(self.my_pose, self.opponent_pose) < 0.4): 
        #     reward -= 10

        ## drone-env collision
        if (baseline_racer.airsim_client.simGetCollisionInfo(vehicle_name = "drone_1").has_collided):
            reward -= 1

        ## gate pass info
        if (self.isPassGate(baseline_racer)):
            self.isPass = 1
            reward += 10
        else:
            dis_reward = dis(self.my_pose, self.gate_pose[self.next_gate_idx].position)
            cal_reward = self.prev_dis_drone_gate - dis_reward
            self.prev_dis_drone_gate = dis_reward
            
            ## min max setting (-0.1 ~ 0.1)
            cal_reward = np.clip(cal_reward, -0.1, 0.1)
            cal_reward = math.exp(-5 * cal_reward)
            
            reward -= cal_reward

        # print(reward)

        return reward

    def isGameOver(self, baseline_racer, reward):
        ## gate pass info
        if (self.isPass > 0):
            print("Passing Gate!")
            return True
        if (reward <= -10):
            return True
        elif(time.time() - self.start_time > 10):
            return True
        else:
            return False

    
    def updateState(self, action, baseline_racer):
        max_vel = 10
        angle_threshold = 1
        duration = 0.05
        
        heading_vec = baseline_racer.get_gate_facing_vector_from_quaternion(baseline_racer.airsim_client.simGetVehiclePose(vehicle_name = "drone_1").orientation, scale = max_vel)

        if (action == 1):
            baseline_racer.airsim_client.moveByVelocityAsync(heading_vec.y_val, -heading_vec.x_val, 0, duration = duration)
        if (action == 2):
            baseline_racer.airsim_client.moveByVelocityAsync(0, 0, -max_vel, duration = duration)
        if (action == 3):
            baseline_racer.airsim_client.moveByVelocityAsync(0, 0, max_vel, duration = duration)
        if (action == 4):
            baseline_racer.airsim_client.moveByAngleRatesThrottleAsync(0, 0, angle_threshold, 0.5, duration = duration)
        if (action == 5):
            baseline_racer.airsim_client.moveByAngleRatesThrottleAsync(0, 0, -angle_threshold, 0.5, duration = duration)
            
        return self.getState(baseline_racer)

    
    def act(self, action, baseline_racer):
        self.updateState(action, baseline_racer)
        reward = self.getReward(baseline_racer)
        gameOver = self.isGameOver(baseline_racer, reward)
        return reward, gameOver, self.getState(baseline_racer)

    

class ReplayMemory:
    def __init__(self, maxMemory, discount):
        self.maxMemory = maxMemory
        self.discount = discount
        self.inputState = np.empty((self.maxMemory, State_size), dtype = np.float32)
        self.actions = np.zeros(self.maxMemory, dtype = np.int8)
        self.nextState = np.empty((self.maxMemory, State_size), dtype = np.float32)
        self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
        self.rewards = np.empty(self.maxMemory, dtype = np.int8)
        self.count = 0
        self.current = 0


    # Appends the experience to the memory.
    def remember(self, currentState, action, reward, nextState, gameOver):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.inputState[self.current, ...] = currentState
        self.nextState[self.current, ...] = nextState
        self.gameOver[self.current] = gameOver
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxMemory


    def getBatch(self, model, batchSize, nbActions, State_size, sess, X):
        # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        # batch we can (at the beginning of training we will not have enough experience to fill a batch).
        memoryLength = self.count
        chosenBatchSize = min(batchSize, memoryLength)

        inputs = np.zeros((chosenBatchSize, State_size))
        targets = np.zeros((chosenBatchSize, nbActions))

        # Fill the inputs and targets up.
        for i in range(chosenBatchSize):
            if (memoryLength == 1):
                memoryLength = 2
            # Choose a random memory experience to add to the batch.
            randomIndex = random.randrange(1, memoryLength)
            current_inputState = np.reshape(self.inputState[randomIndex], (1, State_size))

            target = sess.run(model, feed_dict={X: current_inputState})

            current_nextState =  np.reshape(self.nextState[randomIndex], (1, State_size))
            current_outputs = sess.run(model, feed_dict={X: current_nextState})      
            
            # Gives us Q_sa, the max q for the next state.
            nextStateMaxQ = np.amax(current_outputs)
            if (self.gameOver[randomIndex] == True):
                target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex]
            else:
                # reward + discount(gamma) * max_a' Q(s',a')
                # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
                # to give an error of 0 for those outputs.
                target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # Update the inputs and targets.
            inputs[i] = current_inputState
            targets[i] = target

        return inputs, targets


def DQN_Train(args):
  print("Training new model")

  baseline_racer = BaselineRacer(drone_name="drone_1", plot_transform=args.plot_transform, viz_traj=args.viz_traj, viz_image_cv2=args.viz_image_cv2)
  baseline_racer.load_level(args.level_name)

  # Define Environment
  env = Env(baseline_racer)
  
  # Define Replay Memory
  memory = ReplayMemory(maxMemory, discount)

  # Add ops to save and restore all the variables.

  with tf.Session(graph = g) as sess: 
    tf.initialize_all_variables().run() 

    copy_ops = update_target()
    sess.run(copy_ops)

    for i in range(epoch):
      if (i % 3 == 0):
          copy_ops = update_target()
          sess.run(copy_ops)
      
      # Initialize the environment.
      err = 0
      env.start(baseline_racer)
      count = 0
      isGameOver = False
      # The initial state of the environment.
      currentState = env.getState(baseline_racer)

      while (isGameOver != True):
        count += 1
        action = -9999  # action initilization
        # Decides if we should choose a random action, or an action from the policy network.

        global epsilon

        # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue and count % 30 == 0):
            epsilon = epsilon * 0.99 

        if (randf(0, 1) <= epsilon):
          action = random.randrange(1, nbActions+1)
        else:          
          # Forward the current state through the network.
          currentState = np.reshape(np.asarray(currentState), (-1, State_size))
          
          q = sess.run(Q, feed_dict={X: currentState})  

          # Find the max index (the chosen action).
          index = q.argmax()
          action = index + 1     
        
        reward, gameOver, nextState = env.act(action, baseline_racer)

        memory.remember(currentState, action, reward, nextState, gameOver)
        
        # Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver
                
        # We get a batch of training data to train the model.
        inputs, targets = memory.getBatch(Target_Q, batchSize, nbActions, State_size, sess, X)

        # Train the network which returns the error.
        _, loss = sess.run([train_optimizer, train_cost], feed_dict={X: inputs, Y: targets})  

        # print(loss)

        err = err + loss

      print("Epoch " + str(i) + ": err = " + str(err) + ": epsilon = " + str(epsilon))
      # Save the variables to disk.

      env.reset(baseline_racer)

      if(i % 100 == 0):
        save_path = saver.save(sess, os.getcwd()+"/model_one_by_one/model.ckpt", global_step = i)
        # print("Model saved in file: %s" % save_path)
###########################################################




def main(args):
    DQN_Train(args)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 'Qualifier_Tier_1', 'Qualifier_Tier_2', 'Qualifier_Tier_3'], default="Soccer_Field_Easy")
    parser.add_argument('--planning_baseline_type', type=str, choices=["all_gates_at_once","all_gates_one_by_one"], default="all_gates_at_once")
    parser.add_argument('--planning_and_control_api', type=str, choices=["moveOnSpline", "moveOnSplineVelConstraints"], default="moveOnSpline")
    parser.add_argument('--enable_plot_transform', dest='plot_transform', action='store_true', default=False)
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)
    
    
