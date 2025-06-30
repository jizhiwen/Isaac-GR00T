# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy.executors
import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
import threading

import numpy as np
import matplotlib.pyplot as plt
from rm_dh.srv import JointCommand  # 导入自定义服务接口
from sensor_msgs.msg import JointState

############################### temp ######################################
import pandas as pd


class ParquetToJointState(Node):
    def __init__(self):
        super().__init__('parquet_to_joint_state')

        # 创建 JointState 消息发布者，队列大小为 10
        self.publisher_ = self.create_publisher(
            JointState, 'rm_joint_topic', 10)

        # 定义关节名称列表，与 action 数据对应
        self.joint_names = ["joint1", "joint2", "joint3",
                            "joint4", "joint5", "joint6", "finger1_joint"]

        # 声明参数：Parquet 文件路径
        self.declare_parameter(
            'parquet_file', '/home/robot/datasets/realman-eye-to-hand/data/chunk-000/episode_000050.parquet')
        self.declare_parameter('publish_rate', 15.0)  # 发布频率 (Hz)

        # 获取参数值
        self.parquet_file = self.get_parameter(
            'parquet_file').get_parameter_value().string_value
        self.publish_rate = self.get_parameter(
            'publish_rate').get_parameter_value().double_value

        # 读取 Parquet 文件
        try:
            self.data = pd.read_parquet(self.parquet_file)
            self.get_logger().info(f"成功读取 Parquet 文件: {self.parquet_file}")
            self.get_logger().info(f"数据包含 {len(self.data)} 行")

            # 检查是否存在 action 列
            if 'action' not in self.data.columns:
                self.get_logger().error("Parquet 文件中未找到 'action' 列")
                exit(1)

            # 开始发布数据
            self.timer = self.create_timer(
                1.0 / self.publish_rate, self.publish_next_state)
            self.current_index = 0

        except Exception as e:
            self.get_logger().error(f"读取 Parquet 文件失败: {str(e)}")
            exit(1)

    def publish_next_state(self):
        """发布下一行 action 数据作为 JointState 消息"""
        if self.current_index >= len(self.data):
            self.get_logger().info("已发布所有数据，重置到开始位置")
            self.current_index = 0

        # 创建 JointState 消息
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # 获取当前行的 action 数据
        action_data = self.data.iloc[self.current_index]['action']

        # 确保 action 数据是列表类型
        if isinstance(action_data, str):
            # 如果 action 是字符串格式，尝试转换为列表
            try:
                # 移除括号和空格，分割字符串并转换为浮点数
                action_data = [float(x)
                               for x in action_data.strip('[]').split(',')]
            except:
                self.get_logger().error(f"无法解析 action 数据: {action_data}")
                self.current_index += 1
                return

        # 设置关节位置
        for val in action_data:
            msg.position.append(val)

        # 发布消息
        self.publisher_.publish(msg)
        self.get_logger().info(
            f"发布关节状态 #{self.current_index}: {action_data[:]}")

        # 增加索引
        self.current_index += 1
############################### temp ######################################


class JointCommandClient(Node):
    def __init__(self):
        super().__init__('my_service_client')
        self.client = self.create_client(JointCommand, 'gr00t_action_topic')

        # 等待服务上线
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务...')
        self.get_logger().info('服务已连接')


lock = threading.Lock()


class CustomRos2Subscriber(Node):
    def __init__(self, node_name="custom_subscriber"):
        super().__init__(node_name)
        # 订阅关节信息
        self.sub_joint = self.create_subscription(
            JointState,
            "/rm_joint_topic",
            self.joint_topic_callback,
            10
        )
        self.sub_joint  # 防止未使用变量警告
        # 订阅相机信息
        self.sub_pos_front = self.create_subscription(
            Image,
            "/camera2/camera2/color/image_raw",
            self.front_image_topic_callback,
            10
        )
        # 订阅相机信息
        self.sub_pos_right = self.create_subscription(
            Image,
            "/camera1/camera1/color/image_rect_raw",
            self.right_image_topic_callback,
            10
        )
        self.sub_joint  # 防止未使用变量警告
        # self.pub_action = self.create_publisher(
        #     JointState,
        #     "/gr00t_action_topic",
        #     10
        # )
        # self.pub_action

        self.obs = {
            # "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.single_arm": np.random.rand(1, 6),
            "state.gripper": np.random.rand(1, 1),
            "video.front_view": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "video.right_view": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
            "annotation.human.action.task_description": ["Move above the red square."],
        }

    def joint_topic_callback(self, msg):
        print("""接收到 /rm_joint_topic 消息后的处理逻辑""")
        with lock:
            from datetime import datetime
            print(f"joint state {datetime.now()}")
            self.obs["state.single_arm"] = np.array(
                msg.position[0:6]).reshape(1, -1)
            self.obs["state.gripper"] = np.array(
                msg.position[6]).reshape(1, -1)
            # print(msg.position)

    def front_image_topic_callback(self, msg):
        print("""接收到 /camera2/camera2/color/image_raw 消息后的处理逻辑""")
        with lock:
            from datetime import datetime
            print(f"image_raw {datetime.now()}")
            if msg.encoding != 'rgb8' and msg.encoding != 'bgr8':
                raise ValueError(f"暂不支持的编码格式: {msg.encoding}，请确保是rgb8或bgr8")
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_reshaped = img_array.reshape(msg.height, msg.width, 3)
            img_batched = np.expand_dims(img_reshaped, axis=0)
            self.obs["video.front_view"] = img_batched
            print(self.obs["video.front_view"].shape)

            # print(img_batched.shape)
            # plt.imshow(img_reshaped)
            # plt.axis('off')  # 不显示坐标轴
            # plt.savefig('simulated_image.png')
            # plt.show()

    def right_image_topic_callback(self, msg):
        print("""接收到 /camera1/camera1/color/image_rect_raw 消息后的处理逻辑""")
        with lock:
            from datetime import datetime
            print(f"image_rect_raw {datetime.now()}")
            if msg.encoding != 'rgb8' and msg.encoding != 'bgr8':
                raise ValueError(f"暂不支持的编码格式: {msg.encoding}，请确保是rgb8或bgr8")
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img_reshaped = img_array.reshape(msg.height, msg.width, 3)
            img_batched = np.expand_dims(img_reshaped, axis=0)
            self.obs["video.right_view"] = img_batched

    def get_obs_data(self):
        return self.obs.copy()


def ros_thread(node):
    """ROS2节点运行线程"""
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()  # 使用 executor.spin() 替代 rclpy.spin(node)
    except StopIteration:
        pass
    finally:
        node.destroy_node()


def custom_logic_thread(jointcommand_node, node):
    """自定义逻辑线程"""
    while rclpy.ok():
        print("*********** is in custom_logic_thread ************")
        time_start = time.time()
        with lock:
            action = policy_client.get_action(node.get_obs_data())
            # print(f"Total time taken to get action: {time.time() - time_start} seconds")
        msg = JointState()
        msg.name = ["joint1", "joint2", "joint3",
                    "joint4", "joint5", "joint6", "finger1_joint"]
        msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for j in range(16):
            for i in range(len(action["action.single_arm"][j])):
                msg.position[i] = action["action.single_arm"][j][i]
            msg.position[6] = action["action.gripper"][j]
            print(msg.position)
            # print(action)
            # node.pub_action.publish(msg)
            req = JointCommand.Request()
            req.joint_state = msg

            future = jointcommand_node.client.call_async(req)
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(jointcommand_node)
            # 等待响应
            executor.spin_until_future_complete(future)
            # if future.result() is not None:
            #     jointcommand_node.get_logger().info(
            #         f"结果: {'成功' if future.result().success else '失败'}")
            # else:
            #     jointcommand_node.get_logger().error('服务调用失败')

            # threading.Event().wait(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="gr1_arms_waist",
    )

    parser.add_argument("--port", type=int,
                        help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    # server mode
    parser.add_argument("--server", action="store_true",
                        help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int,
                        help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        import time

        rclpy.init()
        # 创建节点实例
        subscriber_node = CustomRos2Subscriber()
        jointcommand_node = JointCommandClient()

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # Create a policy wrapper
        policy_client = RobotInferenceClient(host=args.host, port=args.port)

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # # Making prediction...
        # # - obs: video.ego_view: (1, 256, 256, 3)
        # # - obs: state.left_arm: (1, 7)
        # # - obs: state.right_arm: (1, 7)
        # # - obs: state.left_hand: (1, 6)
        # # - obs: state.right_hand: (1, 6)
        # # - obs: state.waist: (1, 3)

        # # - action: action.left_arm: (16, 7)
        # # - action: action.right_arm: (16, 7)
        # # - action: action.left_hand: (16, 6)
        # # - action: action.right_hand: (16, 6)
        # # - action: action.waist: (16, 3)
        # obs = {
        #     "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        #     "state.left_arm": np.random.rand(1, 7),
        #     "state.right_arm": np.random.rand(1, 7),
        #     "state.left_hand": np.random.rand(1, 6),
        #     "state.right_hand": np.random.rand(1, 6),
        #     "state.waist": np.random.rand(1, 3),
        #     "annotation.human.action.task_description": ["do your thing!"],
        # }

        # 创建并启动线程
        ros_t = threading.Thread(target=ros_thread, args=(subscriber_node,))
        custom_t = threading.Thread(target=custom_logic_thread, args=(
            jointcommand_node, subscriber_node,))

        ros_t.start()
        custom_t.start()

        try:
            # 等待线程结束
            ros_t.join()
            custom_t.join()
        except KeyboardInterrupt:
            # 中断信号处理
            subscriber_node.get_logger().info('收到中断信号，退出...')
        finally:
            subscriber_node.destroy_node()
            rclpy.shutdown()

    else:
        raise ValueError("Please specify either --server or --client")
