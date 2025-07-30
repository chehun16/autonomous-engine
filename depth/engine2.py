import cv2
import numpy as np
import sys
import tensorrt as trt
from scipy.ndimage import zoom  # for resizing
import pycuda.driver as cuda
import pycuda.autoinit  # automatically initialize CUDA driver

import time
import os
from typing import Tuple
from datetime import datetime ## 폴더 만들기용
import pytz ## 폴더 만들기용

# 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# from lane.engine import LaneEngine

# def overlay_z_on_rgb(rgb_image, depth_map, intrinsic_matrix, Y_world):

#     height, width, _ = rgb_image.shape
#     output_image = rgb_image.copy()  # 원본 이미지 복사
#     output_image2 = rgb_image.copy()  # 원본 이미지 복사

#     # 카메라 내부 파라미터
#     f_x, f_y = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]  # 초점 거리
#     c_x, c_y = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]  # 중심점


#     for i in range(0, height, 50):
#         for j in range(0, width, 50):

#             depth = depth_map[i, j]
#             if depth > 0:  # 유효한 깊이 값만 사용
#                 # 월드 Z 좌표를 RGB 이미지 위에 텍스트로 표시
#                 # 텍스트 위치: (j, i)
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(output_image2, f"{Y_world[i, j]:.2f}", (j, i), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
#     # 결과 이미지를 파일로 저장
#     cv2.imwrite('output_image_with_z.jpg', output_image2)

#     return output_image

# def histogram(z):
#     import matplotlib.pyplot as plt

#     # Sample 1D data
#     # z = np.random.normal(loc=0, scale=1, size=1000)  # 평균 0, 표준편차 1인 정규분포 데이터
#     # print(z.shape)

#     # npy 파일 읽기
#     # file_path = "world_coordinates.npy"  # npy 파일 경로
#     # data = np.load(file_path)

#     # z = z[:, :, 2]
#     z = z.reshape(-1)
#     print(z.shape)

#     print(f"z.min() : {z.min()}, z.max() : {z.max()}")

#     # x축 범위 정의
#     x = np.linspace(z.min(), z.max(), 1000)


#     # 히스토그램 데이터를 직접 추출
#     bin_counts, bin_edges = np.histogram(z, bins=30)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Bin 중심 계산

#     # 히스토그램 값으로 그래프 그리기
#     plt.bar(bin_centers, bin_counts, width=bin_edges[1] - bin_edges[0], color='blue', alpha=0.6, label='Histogram')

#     plt.xlim(-5.0, 5.0)
#     # 그래프 꾸미기
#     plt.title("Histogram with Actual Frequency (z data)")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.legend()

#     plt.savefig("histogram.png")
#     plt.close()



class DepthEngine:
    def __init__(self,
                lane_engine,
                vis=False,
                tensorRT_engine_path="./depth_anything_vits14_308.trt",
                # input_size = (308, 504),
                input_size = (308, 308),
                # input_size = (504, 308),
                cam_intrinsics=np.array([
                                            [809.5233, 0, 339.2379],  # f_x, 0, c_x
                                            [0, 808.7865, 265.3243],  # 0, f_y, c_y
                                            [0, 0, 1]
                                        ]),
                grid_size=100,
                cell_size=0.1,
                pitch_angle_degree=-18,
                min_height=-np.inf,
                max_height=-1,
                min_occupied=30, ## _obstacleCheck를 위한 변수라 사실상 필요 없을듯?
                initial_steering=0,
                initial_throttle=0,
                ):

        self.vis = vis
        self.input_size = input_size
        self.cam_intrinsics = cam_intrinsics
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.pitch_angle = np.radians(pitch_angle_degree)
        self.min_height = min_height
        self.max_height = max_height
        self.min_occupied = min_occupied

        self.width, self.height = input_size

        # self.camera = self._initialize_camera()
        self._width = 1280
        self._height = 720
        self.engine = self._load_engine(tensorRT_engine_path)
        # self.car = NvidiaRacecar(throttle_gain=1)

        # self.lane_engine = LaneEngine(trt_engine_path="../lane/epoch_99_jjuu.trt"
        #                               , input_size=(288,800), image_size=(480, 640), num_lanes=2, num_grid=100, cls_num_per_lane=56, save=True)

        # self.lane_engine = LaneEngine(trt_engine_path="lane/epoch_99_jjuu.trt"
        #                               , input_size=(288,800), image_size=(480, 640), num_lanes=2, num_grid=100, cls_num_per_lane=56, save=True)

        self.lane_engine = lane_engine

        # self.model_input_size = 1862784
        # self.model_output_size = input_size[0] * input_size[1] * 4

        # Allocate pagelocked memory
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.width, self.height)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.width, self.height)), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = cuda.Stream()
        self.context = self.engine.create_execution_context()

        # print(f"Host Input Bytes: {self.h_input.nbytes}")
        # print(f"Device Input Size: {self.d_input.size}")

        # self._initialize_car(initial_steering, initial_throttle)


        self.rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch_angle), -np.sin(self.pitch_angle)],
            [0, np.sin(self.pitch_angle), np.cos(self.pitch_angle)]
        ])

        if vis:
            self.dir_name = self._make_dir()

    def _make_dir(self):

        # KST 타임존 설정
        kst = pytz.timezone("Asia/Seoul")

        # 현재 시간을 KST로 가져오기
        now = datetime.now(kst)

        # 현재 년월일시분초 로 디렉토리 생성
        formatted_time = now.strftime("%Y%m%d%H%M%S")
        dir_name = f"occupancy_map_{formatted_time}"
        os.mkdir(dir_name)
        return dir_name


    def _initialize_car(self, initial_steering, initial_throttle):
        self.car.steering = initial_steering
        self.car.throttle = initial_throttle

    # def _initialize_camera(self):

    #     camera = cv2.VideoCapture(
    #         "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    #         "nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
    #         "videoconvert ! appsink",
    #         cv2.CAP_GSTREAMER
    #     )
    #     if not camera.isOpened():
    #         raise RuntimeError("CSI camera가 인식되지 않습니다.")
    #     return camera

    def _load_engine(self, tensorRT_engine_path):
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        # trt_runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
        with open(tensorRT_engine_path, 'rb') as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)

    # def _take_picture_using_CSICamera(self):
    #     last_saved_time = time.time()

    #     # Capture frame from CSI camera
    #     ret, frame = self.camera.read()
    #     # ret, frame = camera.read(camera)
    #     if not ret:
    #         raise RuntimeError("CSI 카메라로 사진을 찍지 못했습니다.")

    #     # Rotate the frame 180 degrees
    #     frame = cv2.rotate(frame, cv2.ROTATE_180)
    #     # frame = cv2.flip(frame, 0)

    #     # 밝기 증가
    #     brightness_value = 30
    #     bright_image = cv2.convertScaleAbs(frame, alpha=1, beta=brightness_value)

    #     # 샤프닝 필터 적용
    #     kernel = np.array([[0, -1, 0],
    #                     [-1, 5, -1],
    #                     [0, -1, 0]])
    #     frame = cv2.filter2D(bright_image, -1, kernel)
    #     return frame

    def _prepare_input(self, rgb_image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
            h, w = rgb_image.shape[:2]
            scale = min(input_size[0] / h, input_size[1] / w)
            rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

            intrinsic_scaled = [
                                self.cam_intrinsics[0, 0] * scale,
                                self.cam_intrinsics[1, 1] * scale,
                                self.cam_intrinsics[0, 2] * scale,
                                self.cam_intrinsics[1, 2] * scale
                                ]

            padding = [123.675, 116.28, 103.53]  # Mean values for normalization (e.g., ImageNet)
            h, w = rgb.shape[:2]
            pad_h = input_size[0] - h
            pad_w = input_size[1] - w
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2
            pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)

            # Preprocess image to match the input tensor shape (1, 3, H, W)
            rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)

            rgb = rgb[None]

            return rgb, intrinsic_scaled, pad_info

    # def _infer(self, image_input: np.ndarray):
    #         # Ensure input is contiguous
    #         image_input = np.ascontiguousarray(image_input)

    #         output_shape = self.engine.get_tensor_shape("pred_depth")

    #         # Transfer data to device
    #         cuda.memcpy_htod_async(self.d_input, image_input, self.stream)


    #         # Run inference
    #         self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)

    #         # Prepare output array with exact shape
    #         output_data = np.empty(output_shape, dtype=np.float32)

    #         # Transfer result back to host
    #         cuda.memcpy_dtoh_async(output_data, self.d_output, self.stream)
    #         self.stream.synchronize()
    #         return output_data

    # def _metric3D_tensorRT(self, input_image):

    #         # Read and preprocess the image
    #         rgb_image = input_image[:, :, ::-1]
    #         ori_shape = rgb_image.shape
    #         image_input, intrinsic_scaled, pad_info = self._prepare_input(rgb_image, self.input_size)

    #         # Add batch dimension to the input (1, 3, H, W)
    #         image_input = np.expand_dims(image_input, axis=0)

    #         # Perform inference
    #         depth_map = self._infer(image_input)

    #         # depth_map을 2로 나누기
    #         pred_depth = depth_map / 2.0
    #         # pred_depth = depth_map

    #         # squeeze와 같은 효과: 불필요한 차원 제거
    #         pred_depth = pred_depth.squeeze()

    #         # pad_info를 사용해 패딩 제거
    #         pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1],
    #                                 pad_info[2] : pred_depth.shape[1] - pad_info[3]]

    #         # 원본 크기로 upsampling (bilinear 방식)
    #         scale_h = ori_shape[0] / pred_depth.shape[0]
    #         scale_w = ori_shape[1] / pred_depth.shape[1]
    #         pred_depth_resized = zoom(pred_depth, (scale_h, scale_w), order=1)  # order=1은 bilinear

    #         # metric 스케일로 변환
    #         canonical_to_real_scale = intrinsic_scaled[0] / 1000.0  # 1000은 기본 초점 거리
    #         pred_depth_metric = pred_depth_resized * canonical_to_real_scale

    #         # depth 값을 0~300으로 클램프
    #         pred_depth_np = np.clip(pred_depth_metric, 0, 300)

    #         # return rgb_image, depth_map_resized
    #         return rgb_image, pred_depth_np

    # def preprocess(self, image: np.ndarray) -> np.ndarray:
    #     """
    #     Preprocess the image
    #     """
    #     image = image.astype(np.float32)
    #     image /= 255.0
    #     image = self.transform({'image': image})['image']
    #     image = image[None]

    #     return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: Resize, Normalize, and Prepare for network.

        Parameters:
        - image (np.ndarray): Input image in HWC format with values in range [0, 255].
        - input_size (int): Target size for resizing (width and height will be the same).

        Returns:
        - np.ndarray: Preprocessed image in CHW format with normalized values.
        """
        # Resize the image to (input_size, input_size)
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]), interpolation=cv2.INTER_CUBIC)

        # Convert image to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply mean and std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Convert HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
        image = np.transpose(image, (2, 0, 1))

        # Add a batch dimension (1, Channel, Height, Width)
        image = image[None]

        return image

    def postprocess(self, depth: np.ndarray) -> np.ndarray:
        """
        Postprocess the depth map
        """
        depth = np.reshape(depth, (self.width, self.height))
        # depth = cv2.resize(depth, (self._width, self._height))
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)

        return depth

    def _infer(self, image: np.ndarray) -> np.ndarray:
        """
        Infer depth from an image using TensorRT
        """

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # ## input image 그리기
        # ax.imshow(image)
        # ax.set_title('RGB image')

        # image_name = "hello.png"
        # plt.savefig(image_name)

        # plt.close()

        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Preprocess the image
        # image = self.preprocess(image)

        # rgb, intrinsic_scaled, pad_info = self._prepare_input(self, image, self.input_size)

        t0 = time.time()

        # Copy the input image to the pagelocked memory
        np.copyto(self.h_input, image.ravel())

        # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)

        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()

        print(f"Inference time: {time.time() - t0:.4f}s")

        # return rgb_image, self.postprocess(self.h_output) # Postprocess the depth map
        return np.reshape(self.h_output, (self.width, self.height))

    # @njit(pararell=True)
    # def _depth_to_occupancy_map(self, rgb_image, depth_map):

    #     # 카메라 내부 파라미터
    #     f_x, f_y = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]  # 초점 거리
    #     c_x, c_y = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]  # 중심점


    #     # 필요한 변수들
    #     height, width = depth_map.shape

    #     # 3D 좌표 변환
    #     depth_map = depth_map.astype(float)  # depth_map을 float 타입으로 변환
    #     y_indices, x_indices = np.indices((height, width))  # (i, j) 인덱스를 2D 배열로 생성

    #     # 유효한 깊이 값만 필터링
    #     valid_depth_mask = depth_map > 0

    #     # 3D 카메라 좌표 계산
    #     X_camera = (x_indices - c_x) * depth_map / f_x
    #     Y_camera = (y_indices - c_y) * depth_map / f_y
    #     Z_camera = depth_map

    #     # 회전된 좌표 계산
    #     camera_coordinates = np.stack([X_camera, Y_camera, Z_camera], axis=-1)  # (height, width, 3)
    #     world_coordinates = np.einsum('ij,klj->kli', self.rotation_matrix, camera_coordinates)  # (height, width, 3)


    #     x_range = [-self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size / 2]
    #     z_range = [0, self.grid_size * self.cell_size]

    #     # Load coordinates (assuming coords is loaded from the given .npy file)
    #     x_coords = world_coordinates[:, :, 0].reshape(-1)
    #     y_coords = world_coordinates[:, :, 1].reshape(-1)
    #     z_coords = world_coordinates[:, :, 2].reshape(-1)

    #     # Filter based on y condition (e.g., y > 0.5)
    #     y_mask = (self.min_height < y_coords) & (y_coords < self.max_height)
    #     x_coords = x_coords[y_mask]
    #     z_coords = z_coords[y_mask]


    #     # # Initialize binary occupancy map
    #     occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

    #     # Mask points that are within the valid range
    #     mask = (x_range[0] <= x_coords) & (x_coords <= x_range[1]) & (z_range[0] <= z_coords) & (z_coords <= z_range[1])

    #     # Filter valid points
    #     x_valid = x_coords[mask]
    #     z_valid = z_coords[mask]

    #     # Compute grid indices for valid points
    #     x_indices = ((x_valid - x_range[0]) / (x_range[1] - x_range[0]) * self.grid_size).astype(int)
    #     z_indices = ((z_valid - z_range[0]) / (z_range[1] - z_range[0]) * self.grid_size).astype(int)

    #     # Ensure indices are within bounds
    #     x_indices = np.clip(x_indices, 0, self.grid_size - 1)
    #     z_indices = np.clip(z_indices, 0, self.grid_size - 1)

    #     # Mark the corresponding cells in the occupancy map as occupied
    #     occupancy_map[z_indices, x_indices] = 1

    #     # 결과 출력
    #     return occupancy_map, world_coordinates[:, :, 1]

    def _depth_to_occupancy_map(self, rgb_image, depth_map, lane_mask):
        # 카메라 내부 파라미터
        f_x, f_y = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        c_x, c_y = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]

        height, width = depth_map.shape

        # 깊이 맵을 float32로 변환
        depth_map = depth_map.astype(np.float32)
        y_indices, x_indices = np.indices((height, width))

        # 3D 좌표 계산
        X_camera = (x_indices - c_x) * depth_map / f_x
        Y_camera = (y_indices - c_y) * depth_map / f_y
        Z_camera = depth_map


        # 유효한 깊이 값과 높이 범위 필터링
        print(f"min of y: {np.min(Y_camera)}")
        print(f"max of y: {np.max(Y_camera)}")
        valid_mask = (depth_map > 0) & (self.min_height < Y_camera) & (Y_camera < self.max_height)
        print(f"length of valid_mask : {len(valid_mask)}")
        x_valid = X_camera[valid_mask]
        z_valid = Z_camera[valid_mask]

        # 격자 범위와 인덱스 계산
        x_range = [-self.grid_size * self.cell_size / 2, self.grid_size * self.cell_size / 2]
        z_range = [0, self.grid_size * self.cell_size]

        mask = (x_range[0] <= x_valid) & (x_valid <= x_range[1]) & (z_range[0] <= z_valid) & (z_valid <= z_range[1])
        x_valid = x_valid[mask]
        z_valid = z_valid[mask]

        x_indices = np.clip(
            ((x_valid - x_range[0]) / (x_range[1] - x_range[0]) * self.grid_size).astype(np.int32),
            0, self.grid_size - 1
        )
        z_indices = np.clip(
            ((z_valid - z_range[0]) / (z_range[1] - z_range[0]) * self.grid_size).astype(np.int32),
            0, self.grid_size - 1
        )

        # 점유 맵 업데이트
        occupancy_map = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        unique_indices = np.unique(np.stack((z_indices, x_indices), axis=1), axis=0)
        print(f"length : {len(unique_indices)}")
        occupancy_map[unique_indices[:, 0], unique_indices[:, 1]] = 1

        # 차선 정보 추가_________________________________________________________________________
        # lane_mask_flat = lane_mask.reshape(-1)  # 1D로 변환
        # lane_x_coords = world_coordinates[:, :, 0].reshape(-1)[lane_mask_flat > 0]  # 활성화된 차선 픽셀의 X 좌표
        # lane_z_coords = world_coordinates[:, :, 2].reshape(-1)[lane_mask_flat > 0]  # 활성화된 차선 픽셀의 Z 좌표

        lane_x_valid = X_camera[lane_mask > 0]  # 활성화된 차선 픽셀의 X 좌표
        lane_z_valid = Z_camera[lane_mask > 0]  # 활성화된 차선 픽셀의 Z 좌표


        # 차선 좌표 필터링
        lane_mask_valid = (x_range[0] <= lane_x_valid) & (lane_x_valid <= x_range[1]) & \
                          (z_range[0] <= lane_z_valid) & (lane_z_valid <= z_range[1])
        lane_x_valid = lane_x_valid[lane_mask_valid]
        lane_z_valid = lane_z_valid[lane_mask_valid]

        # 차선 좌표를 격자 인덱스로 변환
        lane_x_indices = ((lane_x_valid - x_range[0]) / (x_range[1] - x_range[0]) * self.grid_size).astype(int)
        lane_z_indices = ((lane_z_valid - z_range[0]) / (z_range[1] - z_range[0]) * self.grid_size).astype(int)

        # 인덱스가 격자 범위를 벗어나지 않도록 클리핑
        lane_x_indices = np.clip(lane_x_indices, 0, self.grid_size - 1)
        lane_z_indices = np.clip(lane_z_indices, 0, self.grid_size - 1)

        # 차선 정보 추가 (2로 설정)

        # 90도 시계 방향 회전 변환
        # rotated_x_indices = lane_z_indices  # z 값이 x로
        # rotated_z_indices = self.grid_size - 1 - lane_x_indices  # x 값이 z로, 뒤집힘

        # occupancy_map[lane_z_indices, lane_x_indices] = 2
        # occupancy_map[self.grid_size - lane_z_indices, lane_x_indices] = 1
        occupancy_map[self.grid_size - lane_z_indices, lane_x_indices] = 2


        # return np.fliplr(occupancy_map), Y_camera
        return occupancy_map, Y_camera



    def _visualize(self, rgb_image, depth_map, occupancy_map):
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        print(f"rgb : {rgb_image.shape}")
        print(f"depth : {depth_map.shape}")
        print(f"occupancy : {occupancy_map.shape}")

        ## input image 그리기
        ax[0].imshow(rgb_image)
        ax[0].set_title('RGB image')

        # Depth map 정규화 후 그리기
        depth_map_log_scaled = np.log(depth_map - np.min(depth_map) + 1)  # 1을 더해 0을 방지

        ax[1].imshow(depth_map_log_scaled, cmap='plasma')
        ax[1].set_title('Predicted Depth Map')

        ## Occupancy Map 그리기

        # 사용자 정의 컬러맵 생성
        colors = ['black', 'white', 'yellow']  # 0: 검은색, 1: 흰색, 2: 노란색
        cmap = ListedColormap(colors)

        ax[2].imshow(occupancy_map, cmap=cmap, origin='lower')
        ax[2].set_title('Occupancy Map')

        # ax[2].imshow(occupancy_map, cmap='gray', origin='lower')
        # ax[2].set_title('Occupancy Map')

        # x축 중앙을 0으로 설정
        map_width = occupancy_map.shape[1]
        tick_positions = np.linspace(0, map_width - 1, 5)  # 5개의 주요 tick 위치
        tick_labels = np.linspace(-map_width // 2, map_width // 2, 5).astype(int)  # 중앙 0을 기준으로 -와 +

        ax[2].set_xticks(tick_positions)
        ax[2].set_xticklabels(tick_labels)


        # Save the figure to a file[]
        plt.tight_layout()

        # KST 타임존 설정
        kst = pytz.timezone("Asia/Seoul")

        # 현재 시간을 KST로 가져오기
        now = datetime.now(kst)

        # image_name = f"{self.dir_name}/{now.strftime('%H%M%S%f')[:-3]}.png"
        image_name = "sample_depthanything.png"
        plt.savefig(image_name)

        plt.close()

    def _obstacleCheck(self, occupancy_map):

        half = self.grid_size // 2

        #left_map, right_map = occupancy_map[:half, half:], occupancy_map[half:, half:]
        # left_map, right_map = occupancy_map[:half, :half], occupancy_map[:half, half:]
        left_map, right_map = occupancy_map[half:, :half], occupancy_map[half:, half:]
        #left_map, right_map = occupancy_map[:, :half], occupancy_map[:, half:]
        left_map_sum, right_map_sum = np.sum(left_map), np.sum(right_map)
        total_sum = left_map_sum + right_map_sum


        print(left_map_sum, right_map_sum)

        # 배열의 크기
        rows, cols = occupancy_map.shape

        # 각 좌표의 인덱스 생성
        y_indices, x_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        centroid = (0, 0)
        # 무게중심 계산
        total_weight = occupancy_map.sum()
        if total_weight == 0:
            # 배열이 전부 0인 경우 무게중심은 정의되지 않음
            centroid = (0, 0)
        else:
            x_centroid = (x_indices * occupancy_map).sum() / total_weight
            y_centroid = (y_indices * occupancy_map).sum() / total_weight
            centroid = (y_centroid, x_centroid)


        # if left_map_sum < self.min_occupied and right_map_sum < self.min_occupied:
        #     return

        # car = NvidiaRacecar()


        if 2 < centroid[1] < half:
            print("---------------------무게중심:", centroid, "오오오오오오오오오")
            print("오른쪽")
            self.car.steering = 0.3

        elif centroid[1] > half+2:
            print("---------------------무게중심:", centroid, "왼왼왼왼왼왼왼왼왼")
            print("왼쪽")
            self.car.steering = -0.3
        elif centroid[1] == 0:
            print("---------------------무게중심:", centroid, "가가가가가가가가가")
            print("가운데")
            self.car.steering = 0


    def _depthAnything_tensorRT(self, input_image):

        # Read and preprocess the image
        rgb_image = input_image[:, :, ::-1]
        ori_shape = rgb_image.shape
        # image_input, intrinsic_scaled, pad_info = self._prepare_input(rgb_image, self.input_size)
        image_input = self.preprocess(input_image)


        # Add batch dimension to the input (1, 3, H, W)
        image_input = np.expand_dims(image_input, axis=0)

        # Perform inference
        depth_map = self._infer(image_input)


        depth_map = self.postprocess(depth_map)

        # # depth_map을 2로 나누기
        # pred_depth = depth_map / 2.0
        # # pred_depth = depth_map

        # # squeeze와 같은 효과: 불필요한 차원 제거
        # pred_depth = pred_depth.squeeze()

        # # pad_info를 사용해 패딩 제거
        # pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1],
        #                         pad_info[2] : pred_depth.shape[1] - pad_info[3]]

        # # 원본 크기로 upsampling (bilinear 방식)
        # scale_h = ori_shape[0] / pred_depth.shape[0]
        # scale_w = ori_shape[1] / pred_depth.shape[1]
        # pred_depth_resized = zoom(pred_depth, (scale_h, scale_w), order=1)  # order=1은 bilinear

        # # metric 스케일로 변환
        # canonical_to_real_scale = intrinsic_scaled[0] / 1000.0  # 1000은 기본 초점 거리
        # pred_depth_metric = pred_depth_resized * canonical_to_real_scale

        # # depth 값을 0~300으로 클램프
        # pred_depth_np = np.clip(pred_depth_metric, 0, 300)

        # return rgb_image, depth_map_resized
        return rgb_image, depth_map

    def run_local(self, frame):

        start = time.time()

        # Step 0: CSI 카메라로 사진을 찍고 저장하기
        # frame = self._take_picture_using_CSICamera()
        # print("사진찍기 완료")

        # frame = cv2.imread("./lanePics/frame_1734770406889.jpg")
        # if frame is None:
        #     raise RuntimeError("사진을 찾지 못했습니다.")
        # print("이미지 로드 완료")


        # print(frame.shape)
        # frame = np.zeros((480, 640, 3)).astype(np.uint8)


        infer_start = time.time()
        # Step 1: 사진파일로부터 rgb정보, depth 정보 추출하기
        # rgb_image, depth_map = self._metric3D_tensorRT(frame)
        # rgb_image, depth_map = self._infer(frame)
        rgb_image, depth_map = self._depthAnything_tensorRT(frame)
        infer_end = time.time()
        print(f"infer_time : {infer_end - infer_start:.4f}")

        print("Depth info 추출 완료")

        # Step 2: LaneEngine으로 차선 정보 추출
        lane_mask = self.lane_engine.run(frame)  # LaneEngine 호출
        print("차선 정보 추출 완료")
        print("\n\n\n\n\n\n")
        print(lane_mask.shape)
        print("\n\n")

        occupancy_start = time.time()
        # Step 2: Depth 정보를 Occupancy map으로 변환하기
        occupancy_map, Y_world = self._depth_to_occupancy_map(rgb_image, depth_map, lane_mask)
        occupancy_end = time.time()
        print(f'occupancy: {occupancy_end - occupancy_start:.4f}s')
        print("Occupany Map 추출 완료")

        # # (Optinal) Step 4: 바퀴 움직이기
        # self._obstacleCheck(occupancy_map)
        # print("바퀴 이동 완료")


        if self.vis:
            # (Optional) Step 3: Occupancy map 시각화하기
            self._visualize(rgb_image, depth_map, occupancy_map)
            # histogram(Y_world)
            # overlay_z_on_rgb(rgb_image, depth_map, self.cam_intrinsics, Y_world) ## z값과 이미지를 겹쳐놓는 함수
            print("시각화 완료")

        end = time.time()

        print(f"전체시간 : {end - start:.4f}")

        return np.flipud(occupancy_map)

    def run(self, frame):

        start = time.time()

        # Step 0: CSI 카메라로 사진을 찍고 저장하기
        # frame = self._take_picture_using_CSICamera()
        # print("사진찍기 완료")

        # print(frame.shape)
        # frame = np.zeros((480, 640, 3)).astype(np.uint8)


        infer_start = time.time()
        # Step 1: 사진파일로부터 rgb정보, depth 정보 추출하기
        # rgb_image, depth_map = self._metric3D_tensorRT(frame)
        # rgb_image, depth_map = self._infer(frame)
        rgb_image, depth_map = self._depthAnything_tensorRT(frame)
        infer_end = time.time()
        print(f"infer_time : {infer_end - infer_start:.4f}")

        print("Depth info 추출 완료")


        occupancy_start = time.time()
        # Step 2: Depth 정보를 Occupancy map으로 변환하기
        occupancy_map, Y_world = self._depth_to_occupancy_map(rgb_image, depth_map)
        occupancy_end = time.time()
        print(f'occupancy: {occupancy_end - occupancy_start:.4f}s')
        print("Occupany Map 추출 완료")

        # # (Optinal) Step 4: 바퀴 움직이기
        # self._obstacleCheck(occupancy_map)
        # print("바퀴 이동 완료")


        if self.vis:
            # (Optional) Step 3: Occupancy map 시각화하기
            self._visualize(rgb_image, depth_map, occupancy_map)
            histogram(Y_world)
            overlay_z_on_rgb(rgb_image, depth_map, self.cam_intrinsics, Y_world) ## z값과 이미지를 겹쳐놓는 함수
            print("시각화 완료")

        end = time.time()

        print(f"전체시간 : {end - start:.4f}")

        return np.flipud(occupancy_map)

if __name__ == "__main__":

    # depthEngine = DepthEngine()
    depthEngine = DepthEngine(vis=True)
    # occupancy_map = depthEngine.run_local()

    scene = "scene-0005"
    images = sorted(os.listdir(scene))

    for image in images:
        frame = cv2.imread(f"./{scene}/{image}")
        if frame is None:
            raise RuntimeError("사진을 찾지 못했습니다.")
        print("이미지 로드 완료")
        occupancy_map = depthEngine.run_local(frame)
    # try:
    #     while True:
    #         occupancy_map = depthEngine.run_local()

    # except KeyboardInterrupt:
    #     # depthEngine.car.throttle = 0
    #     # depthEngine.car.steering = 0
    #     print("\n\nDepthEngine 강제 종료\n\n")