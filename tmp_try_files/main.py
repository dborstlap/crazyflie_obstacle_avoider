# FIle that takes different functions from my_code and runs them.
# For example, run commander, and get images, and livestream, and log certain data.

import time
from my_functions import get_image, show_image
import argparse


# radio
global URI_RADIO = 'radio://0/80/2M/E7E7E7E7E7'

# Wifi 
global URI_WIFI = 'tcp://192.168.4.1:5000'


CAM_HEIGHT = 244
CAM_WIDTH = 324
# Set the speed factor for moving and rotating
SPEED_FACTOR = 0.3


start_time = time.time()
count = 0


running = True
while running:

    image, count = get_image(count, start_time)
    show_image(image)

    steering angle = nn_compute_direction()
















