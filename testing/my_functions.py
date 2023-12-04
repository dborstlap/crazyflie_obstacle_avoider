

import argparse
import time
import socket,struct, time
import numpy as np
import cv2



parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
args = parser.parse_args()
deck_port = args.p
deck_ip = args.n


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))  # try using global variable!!!!!!

def rx_bytes(size):
  data = bytearray()
  while len(data) < size:
    data.extend(client_socket.recv(size-len(data)))
  return data





def get_image(count, start):
    # First get the info
    packetInfoRaw = rx_bytes(4)
    [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)

    imgHeader = rx_bytes(length - 2)
    [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

    if magic == 0xBC:
      # Now we start rx the image, this will be split up in packages of some size
      imgStream = bytearray()

      while len(imgStream) < size:
          packetInfoRaw = rx_bytes(4)
          [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
          #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
          chunk = rx_bytes(length - 2)
          imgStream.extend(chunk)
     
      count = count + 1
      meanTimePerImage = (time.time()-start) / count

      if format == 0:
          bayer_img = np.frombuffer(imgStream, dtype=np.uint8)   
          bayer_img.shape = (244, 324)
          final_image = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)

        #   cv2.imshow('Raw', bayer_img)
        #   cv2.imshow('Color', final_image)
        #   cv2.waitKey(1)

      else:
          print('format != 0')
          with open("img.jpeg", "wb") as f:
              f.write(imgStream)
          nparr = np.frombuffer(imgStream, np.uint8)
          final_image = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)

        #   cv2.imshow('JPEG', final_image)
        #   cv2.waitKey(1)

    return final_image, count


def show_image(image):
    cv2.imshow('bayer or something', image)    
    cv2.waitKey(1)

    return 0










