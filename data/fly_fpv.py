"""
fly the drone in fpv mode

Follow these steps
1) connect to drone wifi
2) 
3) 

"""


import logging
logging.basicConfig(level=logging.INFO)

import struct
import sys
import threading
import os
import datetime

import numpy as np

import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt5 import QtCore, QtWidgets, QtGui


# URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')


CAM_HEIGHT = 244
CAM_WIDTH = 324
SPEED_FACTOR = 1.5
KEY_LABELS = {
    QtCore.Qt.Key_Left: 'left',
    QtCore.Qt.Key_Right: 'right',
    QtCore.Qt.Key_Up: 'forward',
}



class ImageDownloader(threading.Thread):
    def __init__(self, cpx, cb):
        threading.Thread.__init__(self)
        self.daemon = True
        self._cpx = cpx
        self._cb = cb

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', p.data[0:11])
            if (magic == 0xBC):
                imgStream = bytearray()
                while len(imgStream) < size:
                    p = self._cpx.receivePacket(CPXFunction.APP)
                    imgStream.extend(p.data)

                bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                self._cb(bayer_img)


class MainWindow(QtWidgets.QWidget):

    def __init__(self, URI, save_data, save_folder):
        self.save_data = save_data
        self.save_folder = save_folder
        self.last_key_pressed = 'forward' # forward is taken as standard (=no obstacles in the way)

        QtWidgets.QWidget.__init__(self)

        self.setWindowTitle('Crazyflie / AI deck FPV demo')

        self.mainLayout = QtWidgets.QVBoxLayout()

        self.image_frame = QtWidgets.QLabel()
        self.mainLayout.addWidget(self.image_frame)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.addWidget(QtWidgets.QLabel('Position (X/Y/Z)'), 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.gridLayout.addWidget(QtWidgets.QLabel('Pose (roll/pitch/yaw)'), 1, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.labels = {
            'stateEstimate.x': {
                'widget': QtWidgets.QLabel('X'),
                'x_grid': 0, 'y_grid': 1,
                'alignment': QtCore.Qt.AlignLeft
            },
            'stateEstimate.y': {
                'widget': QtWidgets.QLabel('Y'),
                'x_grid': 0, 'y_grid': 2,
                'alignment': QtCore.Qt.AlignLeft
            },
            'stateEstimate.z': {
                'widget': QtWidgets.QLabel('Z'),
                'x_grid': 0, 'y_grid': 3,
                'alignment': QtCore.Qt.AlignLeft
            },
            'stabilizer.roll': {
                'widget': QtWidgets.QLabel('roll'),
                'x_grid': 1, 'y_grid': 1,
                'alignment': QtCore.Qt.AlignLeft
            },
            'stabilizer.pitch': {
                'widget': QtWidgets.QLabel('pitch'),
                'x_grid': 1, 'y_grid': 2,
                'alignment': QtCore.Qt.AlignLeft
            },
            'stabilizer.yaw': {
                'widget': QtWidgets.QLabel('yaw'),
                'x_grid': 1, 'y_grid': 3,
                'alignment': QtCore.Qt.AlignLeft
            }
        }

        for name in self.labels:
            w = self.labels[name]
            self.gridLayout.addWidget(w['widget'], w['x_grid'], w['y_grid'], 1, 1, w['alignment'])

        self.mainLayout.addLayout(self.gridLayout)

        self.setLayout(self.mainLayout)

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        # Connect callbacks from the Crazyflie API
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)

        # Connect to the Crazyflie
        self.cf.open_link(URI)

        if not self.cf.link:
            print('Could not connect to Crazyflie')
            sys.exit(1)

        if not hasattr(self.cf.link, 'cpx'):
            print('Not connecting with WiFi')
            self.cf.close_link()
        else:
            self._imgDownload = ImageDownloader(self.cf.link.cpx, self.updateImage)
            self._imgDownload.start()

            self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.3}

            self.hoverTimer = QtCore.QTimer()
            self.hoverTimer.timeout.connect(self.sendHoverCommand)
            self.hoverTimer.setInterval(100)
            self.hoverTimer.start()

    def updateImage(self, image):
        i = QtGui.QImage(image, CAM_WIDTH, CAM_HEIGHT, QtGui.QImage.Format_Grayscale8).scaled(324*2, 244*2)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(i))

        # Save the image with the last key pressed
        if self.save_data:
            self.saveImage(i, self.last_key_pressed)

    def saveImage(self, image, key_pressed):
        
        script_dir = os.path.abspath(os.path.dirname(__file__))
        save_dir = os.path.join(script_dir, '..', 'data/datasets', self.save_folder)

        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct filename
        filename = f"{save_dir}/img_{timestamp}_{key_pressed}.png"

        # Save the image
        image.save(filename)

    def keyPressEvent(self, event):
        if (not event.isAutoRepeat()):
            if (event.key() == QtCore.Qt.Key_Up):
                self.updateHover('x', 1)
                self.last_key_pressed = 'forward'

            if (event.key() == QtCore.Qt.Key_Down):
                self.updateHover('x', -1)
                self.last_key_pressed = 'forward'  # should be backward but training code would break

            if (event.key() == QtCore.Qt.Key_Left):
                self.updateHover('y', 1)
                self.last_key_pressed = 'left'

            if (event.key() == QtCore.Qt.Key_Right):
                self.updateHover('y', -1)
                self.last_key_pressed = 'right'

            if (event.key() == QtCore.Qt.Key_A):
                self.updateHover('yaw', -70)
                self.last_key_pressed = 'left'

            if (event.key() == QtCore.Qt.Key_D):
                self.updateHover('yaw', 70)
                self.last_key_pressed = 'right'

            if (event.key() == QtCore.Qt.Key_Z):
                self.updateHover('yaw', -200)
                self.last_key_pressed = 'left'

            if (event.key() == QtCore.Qt.Key_X):
                self.updateHover('yaw', 200)
                self.last_key_pressed = 'right'

            if (event.key() == QtCore.Qt.Key_W):
                self.updateHover('height', 0.1)
            if (event.key() == QtCore.Qt.Key_S):
                self.updateHover('height', -0.1)

    def keyReleaseEvent(self, event):
        if (not event.isAutoRepeat()):
            if (event.key() == QtCore.Qt.Key_Left):
                self.updateHover('y', 0)
            if (event.key() == QtCore.Qt.Key_Right):
                self.updateHover('y', 0)
            if (event.key() == QtCore.Qt.Key_Up):
                self.updateHover('x', 0)
            if (event.key() == QtCore.Qt.Key_Down):
                self.updateHover('x', 0)
            if (event.key() == QtCore.Qt.Key_A):
                self.updateHover('yaw', 0)
            if (event.key() == QtCore.Qt.Key_D):
                self.updateHover('yaw', 0)
            if (event.key() == QtCore.Qt.Key_W):
                self.updateHover('height', 0)
            if (event.key() == QtCore.Qt.Key_S):
                self.updateHover('height', 0)
            if (event.key() == QtCore.Qt.Key_Z):
                self.updateHover('yaw', 0)
            if (event.key() == QtCore.Qt.Key_X):
                self.updateHover('yaw', 0)

    def sendHoverCommand(self):
        self.cf.commander.send_hover_setpoint(
            self.hover['x'], self.hover['y'], self.hover['yaw'],
            self.hover['height'])

    def updateHover(self, k, v):
        if (k != 'height'):
            self.hover[k] = v * SPEED_FACTOR
        else:
            self.hover[k] += v

    def disconnected(self, URI):
        print('Disconnected')
        sys.exit(1)

    def connected(self, URI):
        print('We are now connected to {}'.format(URI))

        # The definition of the logconfig can be made before connecting
        lp = LogConfig(name='Position', period_in_ms=100)
        lp.add_variable('stateEstimate.x')
        lp.add_variable('stateEstimate.y')
        lp.add_variable('stateEstimate.z')
        lp.add_variable('stabilizer.roll')
        lp.add_variable('stabilizer.pitch')
        lp.add_variable('stabilizer.yaw')

        try:
            self.cf.log.add_config(lp)
            lp.data_received_cb.add_callback(self.pos_data)
            lp.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Position log config, bad configuration.')

    def pos_data(self, timestamp, data, logconf):
        for name in data:
            self.labels[name]['widget'].setText('{:.02f}'.format(data[name]))

    def closeEvent(self, e):
        if (self.cf is not None):
            self.cf.close_link()




if __name__ == '__main__':

    SAVE_DATA = True
    SAVE_FOLDER = 'cyberzoo_set4'


    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow(URI, SAVE_DATA, SAVE_FOLDER)
    win.show()
    appQt.exec_()



