# coding: utf-8
import os
import json
import cv2
import numpy as np
import imageio
import torch
import time
import argparse
import rclpy
from std_msgs.msg import String
from rclpy.node import Node

# Try to import RealSense, but make it optional
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Will use webcam fallback.")

from uninavid.mm_utils import get_model_name_from_path
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria


seed = 30
torch.manual_seed(seed)
np.random.seed(seed)


class UniNaVid_Agent():
    def __init__(self, model_path):
        
        print("Initialize UniNaVid")
        
        self.conv_mode = "vicuna_v1"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        assert self.image_processor is not None

        print("Initialization Complete")
        
        self.promt_template = "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations and an image of the current observation <image>. Your assigned task is: '{}'. Analyze this series of images to determine your next four actions. The predicted action should be one of the following: forward, left, right, or stop."
        self.rgb_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):

        
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        return [video]


    def predict_inference(self, prompt):
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs




    def reset(self, task_type='vln'):

        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.task_type = task_type

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0


    def act(self, data):
    
        rgb = data["observations"]
        self.rgb_list.append(rgb)


        navigation_qs = self.promt_template.format(data["instruction"])
        
        navigation = self.predict_inference(navigation_qs)
                
        action_list = navigation.split(" ")

        traj = [[0.0, 0.0, 0.0]]
        for action in action_list: 
            if action == "stop":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, 0.0])]
                traj = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                break
            elif action == "forward":
                waypoint = [x + y for x, y in zip(traj[-1], [0.5, 0.0, 0.0])]
                traj.append(waypoint)
            elif action == "left":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, -np.deg2rad(30)])]
                traj.append(waypoint)
            elif action == "right":
                waypoint = [x + y for x, y in zip(traj[-1], [0.0, 0.0, np.deg2rad(30)])]
                traj.append(waypoint)

                                    
        if len(action_list)==0:
            raise ValueError("No action found in the output")
        
        # Accumulate actions in the queue
        self.pending_action_list.extend(action_list)
            
        self.executed_steps += 1
            
        self.latest_action = {"step": self.executed_steps, "path":[traj], "actions":action_list}
            
        return self.latest_action.copy()

def initialize_realsense(width=640, height=480, fps=30):
    """Initialize RealSense camera pipeline"""
    if not REALSENSE_AVAILABLE:
        raise ImportError("pyrealsense2 is not installed. Install it with: pip install pyrealsense2")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Check for available devices
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense devices found. Please connect a RealSense camera.")
    
    print(f"Found {len(devices)} RealSense device(s)")
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev.get_info(rs.camera_info.name)}")
    
    # Enable color stream
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    # Start streaming
    print(f"Starting RealSense camera (width={width}, height={height}, fps={fps})...")
    try:
        pipeline.start(config)
    except RuntimeError as e:
        error_msg = str(e)
        if "Couldn't resolve requests" in error_msg:
            print("\nError: Couldn't resolve camera stream requests.")
            print("Possible causes:")
            print("  1. The requested resolution/FPS combination is not supported by your camera")
            print("  2. Another application is using the camera")
            print("  3. Camera firmware/driver issue")
            print("\nTrying to find a supported configuration...")
            
            # Try common configurations
            configs_to_try = [
                (640, 480, 30),
                (640, 480, 15),
                (640, 480, 6),
                (424, 240, 30),
                (424, 240, 15),
            ]
            
            for w, h, f in configs_to_try:
                try:
                    config = rs.config()
                    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                    pipeline.start(config)
                    print(f"Successfully started with width={w}, height={h}, fps={f}")
                    width, height, fps = w, h, f
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("Could not find a supported camera configuration. Please check your camera connection and try a different resolution/FPS.")
        else:
            raise
    
    # Wait for a few frames to allow auto-exposure to stabilize
    print("Waiting for camera to stabilize...")
    for _ in range(30):
        pipeline.wait_for_frames()
    
    print("RealSense camera ready!")
    return pipeline

def get_realsense_frame(pipeline):
    """Capture a frame from RealSense camera"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        return None
    
    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    return color_image

def initialize_webcam(camera_id=0, width=640, height=480):
    """Initialize webcam using OpenCV"""
    print(f"Initializing webcam (camera_id={camera_id}, width={width}, height={height})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with camera_id={camera_id}. Make sure the camera is connected and not in use by another application.")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Read a few frames to let the camera stabilize
    print("Waiting for camera to stabilize...")
    for _ in range(10):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read frames from webcam")
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam ready! Actual resolution: {actual_width}x{actual_height}")
    
    return cap

def get_webcam_frame(cap):
    """Capture a frame from webcam"""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def draw_traj_arrows_fpv(
    img,
    actions,
    arrow_len=10,                
    arrow_gap=2,                 
    arrow_color=(0, 255, 0),    
    arrow_thickness=2,
    tipLength=0.35,
    stop_color=(0, 0, 255),      
    stop_radius=5
):
 
    out = img.copy()
    h, w = out.shape[:2]

    base_x, base_y = w // 2, int(h * 0.95)

    for i, action in enumerate(actions):
        if action == "stop":
            waypoint = [0.0, 0.0, 0.0]
        elif action == "forward":
            waypoint = [0.5, 0.0, 0.0]
        elif action == "left":
            waypoint = [0.0, 0.0, -np.deg2rad(30)]
        elif action == "right":
            waypoint = [0.0, 0.0, np.deg2rad(30)]
        else:
            continue  

        x, y, yaw = waypoint

        start = (
            int(base_x),
            int(base_y - i * (arrow_len + arrow_gap))
        )

        if action == "stop":
            cv2.circle(out, start, stop_radius, stop_color, 2)
        else:
            end = (
                int(start[0] + arrow_len * np.sin(yaw)),
                int(start[1] - arrow_len * np.cos(yaw))
            )
            cv2.arrowedLine(out, start, end, arrow_color, arrow_thickness, tipLength=tipLength)
    
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Online evaluation with camera')
    parser.add_argument('--instruction', type=str, required=True, help='Navigation instruction/task description')
    parser.add_argument('--output_dir', type=str, default='Real_test_output', help='Output directory to save results (default: output_dir)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps to process (default: unlimited)')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--fps', type=int, default=1, help='Camera FPS for RealSense (default: 1, not used for webcam)')
    parser.add_argument('--model_path', type=str, default='./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2', help='Path to model (default: ./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2)')
    parser.add_argument('--save_gif', action='store_true', help='Save visualization as GIF')
    parser.add_argument('--display', action='store_true', help='Display frames in real-time')
    parser.add_argument('--camera_type', type=str, default='auto', choices=['auto', 'realsense', 'webcam'], 
                        help='Camera type: auto (try RealSense first, fallback to webcam), realsense, or webcam (default: auto)')
    parser.add_argument('--camera_id', type=int, default=0, help='Webcam camera ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
        #ros
    rclpy.init(args=None)
    ros_node = rclpy.create_node('publish_node')
    publisher_sign = ros_node.create_publisher(String, 'sign', 10)
    print("*" * 10 +"ROS2 Node Ready" + "*" * 10)

    # Initialize agent
    agent = UniNaVid_Agent(args.model_path)
    agent.reset()
    
    # Initialize camera based on camera_type
    camera_type = None
    pipeline = None
    cap = None
    
    if args.camera_type == 'realsense':
        # Force RealSense
        if not REALSENSE_AVAILABLE:
            print("Error: RealSense requested but pyrealsense2 is not available.")
            print("Install it with: pip install pyrealsense2")
            exit(1)
        try:
            pipeline = initialize_realsense(width=args.width, height=args.height, fps=args.fps)
            camera_type = 'realsense'
        except Exception as e:
            print(f"Error initializing RealSense camera: {e}")
            exit(1)
    elif args.camera_type == 'webcam':
        # Force webcam
        try:
            cap = initialize_webcam(camera_id=args.camera_id, width=args.width, height=args.height)
            camera_type = 'webcam'
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            exit(1)
    else:
        # Auto: try RealSense first, fallback to webcam
        if REALSENSE_AVAILABLE:
            try:
                pipeline = initialize_realsense(width=args.width, height=args.height, fps=args.fps)
                camera_type = 'realsense'
                print("Using RealSense camera")
            except Exception as e:
                print(f"RealSense initialization failed: {e}")
                print("Falling back to webcam...")
                try:
                    cap = initialize_webcam(camera_id=args.camera_id, width=args.width, height=args.height)
                    camera_type = 'webcam'
                    print("Using webcam")
                except Exception as e2:
                    print(f"Webcam initialization also failed: {e2}")
                    exit(1)
        else:
            print("RealSense not available, using webcam...")
            try:
                cap = initialize_webcam(camera_id=args.camera_id, width=args.width, height=args.height)
                camera_type = 'webcam'
            except Exception as e:
                print(f"Webcam initialization failed: {e}")
                exit(1)
    
    result_vis_list = []
    step_count = 0
    
    try:
        print(f"\nStarting online evaluation with instruction: '{args.instruction}'")
        print("Press Ctrl+C to stop\n")
        
        while True:
            # Check max_steps limit
            if args.max_steps is not None and step_count >= args.max_steps:
                print(f"Reached maximum steps ({args.max_steps})")
                break
            
            # Capture frame based on camera type
            if camera_type == 'realsense':
                frame = get_realsense_frame(pipeline)
            else:
                frame = get_webcam_frame(cap)
            
            if frame is None:
                print("Warning: Failed to capture frame, skipping...")
                continue
            
            # Process frame through agent
            t_s = time.time()
            result = agent.act({'instruction': args.instruction, 'observations': frame})
            step_count += 1
            
            inference_time = time.time() - t_s
            print(f"Step {step_count}, inference time: {inference_time:.3f}s, actions: {result['actions']}")
            
            # Get trajectory and actions
            traj = result['path'][0]
            actions = result['actions']
            print("Current frame actions:", actions)
            
            # Publish the most recent action from pending_action_list
            msg = String()
            msg.data = agent.pending_action_list[-1]
            print(f"Publishing action: {msg.data} (inference time: {inference_time:.3f}s)")
            publisher_sign.publish(msg)

            # Draw visualization
            vis = draw_traj_arrows_fpv(frame, actions, arrow_len=20)
            result_vis_list.append(vis)
            
            # Display frame if requested
            if args.display:
                display_frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imshow('UniNaVid Online Evaluation', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user (pressed 'q')")
                    break
            
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop camera
        print("\nStopping camera...")
        if camera_type == 'realsense' and pipeline is not None:
            pipeline.stop()
        elif camera_type == 'webcam' and cap is not None:
            cap.release()
        if args.display:
            cv2.destroyAllWindows()
    
    # Save results if requested
    if args.save_gif and len(result_vis_list) > 0:
        gif_path = os.path.join(output_dir, "indoor1.gif")
        print(f"\nSaving visualization to {gif_path}...")
        imageio.mimsave(gif_path, result_vis_list, fps=2)
        print(f"Saved {len(result_vis_list)} frames to {gif_path}")
    
    print(f"\nCompleted {step_count} steps")

