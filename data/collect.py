"""
Needs orginal firmware (can be loaded from the Crazyflie client)

"""
# imports
import socket
import struct
import os
import datetime
import numpy as np
import cv2
import time
from cflib.utils import uri_helper
# from fly_fpv import MainWindow  # Importing the FPV flying functionality

# Constants for image processing
CAM_HEIGHT = 244
CAM_WIDTH = 324
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')


# Function to connect to the Crazyflie via WiFi
def connect_to_drone():
    deck_ip = '192.168.4.1'
    deck_port = 5000

    print(f"Connecting to socket on {deck_ip}:{deck_port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((deck_ip, deck_port))
    print("Socket connected")
    return client_socket

# Function to receive a specified number of bytes from the socket
def rx_bytes(client_socket, size):
    data = bytearray()
    while len(data) < size:
        data.extend(client_socket.recv(size - len(data)))
    return data

# Function to capture images from the Crazyflie and save them when user presses Enter
def collect_images(client_socket, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    count = 0
    start_time = time.time()
    
    while True:
        input("Press Enter to capture an image:  ")
        
        # Receive packet info and image header
        packet_info = rx_bytes(client_socket, 4)
        length, routing, function = struct.unpack('<HBB', packet_info)
        img_header = rx_bytes(client_socket, length - 2)
        magic, width, height, depth, format, size = struct.unpack('<BHHBBI', img_header)
        
        if magic == 0xBC:  # If image packet is valid
            img_stream = bytearray()
            
            while len(img_stream) < size:
                packet_info = rx_bytes(client_socket, 4)
                length, dst, src = struct.unpack('<HBB', packet_info)
                chunk = rx_bytes(client_socket, length - 2)
                img_stream.extend(chunk)
            
            # Convert the raw image data to a Bayer image and then to color
            bayer_img = np.frombuffer(img_stream, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH)
            color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
            
            # Save the images
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_img_path = os.path.join(save_folder, f"img_raw_{timestamp}.png")
            color_img_path = os.path.join(save_folder, f"img_color_{timestamp}.png")
            
            cv2.imwrite(raw_img_path, bayer_img)
            cv2.imwrite(color_img_path, color_img)
            count += 1
            
            # Display statistics
            elapsed_time = time.time() - start_time
            print(f"Captured image {count}. Avg time per image: {elapsed_time / count:.2f} seconds")


if __name__ == '__main__':
    # Define what you want to do
    SAVE_FOLDER = "data/training_data/raw/my_test_images"

    # Connect to the drone via WiFi
    client_socket = connect_to_drone()

    # Collect images on Enter press
    collect_images(client_socket, SAVE_FOLDER)
