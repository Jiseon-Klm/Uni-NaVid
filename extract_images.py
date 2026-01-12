import os
import cv2
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

def extract_images(bag_path, output_dir, topic_name):
    # 1. 설정
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bridge = CvBridge()
    
    # 2. Bag Reader 초기화
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # 3. 토픽 필터링
    storage_filter = rosbag2_py.StorageFilter(topics=[topic_name])
    reader.set_filter(storage_filter)

    count = 0
    print(f"Start extracting from {topic_name}...")

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        
        # 메시지 타입 가져오기 및 역직렬화 (Deserialize)
        msg_type = get_message('sensor_msgs/msg/Image')
        msg = deserialize_message(data, msg_type)

        # 4. ROS Image -> OpenCV 변환
        # encoding은 'bgr8' (컬러) 또는 'passthrough' (Depth)
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(f"Error converting image: {e}")
            continue

        # 5. 파일 저장 (타임스탬프 파일명 권장)
        # t는 나노초 단위입니다.
        filename = f"{output_dir}/frame_{t}.jpg"
        cv2.imwrite(filename, cv_image)
        
        count += 1
        if count % 100 == 0:
            print(f"Saved {count} images...")

    print(f"Done! Total {count} images saved to {output_dir}")

if __name__ == "__main__":
    # 사용 예시
    BAG_FILE = "session_01"  # .mcap 파일이 들어있는 폴더나 파일 경로
    OUTPUT_DIR = "output_images"
    TOPIC = "/camera/color/image_raw"
    
    extract_images(BAG_FILE, OUTPUT_DIR, TOPIC)
