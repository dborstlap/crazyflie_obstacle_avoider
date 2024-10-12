"""
Fly crazyflie using arrow keys. Yaw and go up-down using using A, D, W, S keys respecitvely.

"""



# imports
import sys
import threading
import numpy as np
import logging
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cflib.crtp import init_drivers
from cflib.crazyflie.log import LogConfig
from PyQt5 import QtCore, QtWidgets

# Standard wifi URI to connect to the Crazyflie
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
SPEED_FACTOR = 1.5

class FlyCrazyflie(QtWidgets.QWidget):
    def __init__(self, URI):
        super().__init__()

        self.setWindowTitle('Crazyflie FPV Control')
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.labels = {
            'stateEstimate.x': QtWidgets.QLabel('X: 0'),
            'stateEstimate.y': QtWidgets.QLabel('Y: 0'),
            'stateEstimate.z': QtWidgets.QLabel('Z: 0'),
            'stabilizer.roll': QtWidgets.QLabel('roll: 0'),
            'stabilizer.pitch': QtWidgets.QLabel('pitch: 0'),
            'stabilizer.yaw': QtWidgets.QLabel('yaw: 0')
        }

        for label in self.labels.values():
            self.mainLayout.addWidget(label)

        init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)

        self.cf.open_link(URI)
        if not self.cf.link:
            print('Could not connect to Crazyflie')
            sys.exit(1)

        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.5}

        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(100)
        self.hoverTimer.start()

    # Move the Crazyflie when arrow keys are pressed
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == QtCore.Qt.Key_Up:
                self.hover['x'] = 1 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_Down:
                self.hover['x'] = -1 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_Left:
                self.hover['y'] = 1 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_Right:
                self.hover['y'] = -1 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_A:
                self.hover['yaw'] = 100 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_D:
                self.hover['yaw'] = -100 * SPEED_FACTOR
            if event.key() == QtCore.Qt.Key_W:
                self.hover['height'] += 0.1
            if event.key() == QtCore.Qt.Key_S:
                self.hover['height'] += -0.1

    # Stop moving when key is released
    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Down:
                self.hover['x'] = 0
            if event.key() == QtCore.Qt.Key_Left or event.key() == QtCore.Qt.Key_Right:
                self.hover['y'] = 0
            if event.key() == QtCore.Qt.Key_A or event.key() == QtCore.Qt.Key_D:
                self.hover['yaw'] = 0
            if event.key() == QtCore.Qt.Key_W or event.key() == QtCore.Qt.Key_S:
                self.hover['height'] += 0

    # send high level control commands to the Crazyflie
    def sendHoverCommand(self):
        self.cf.commander.send_hover_setpoint(
            self.hover['x'], 
            self.hover['y'], 
            self.hover['yaw'], 
            self.hover['height'])

    # When crazyflie connects, it runs this callback function
    def connected(self, URI):
        print(f'Connected to {URI}')

        # defines logs to save
        log_config = LogConfig(name='Position', period_in_ms=100)
        log_config.add_variable('stateEstimate.x')
        log_config.add_variable('stateEstimate.y')
        log_config.add_variable('stateEstimate.z')
        log_config.add_variable('stabilizer.roll')
        log_config.add_variable('stabilizer.pitch')
        log_config.add_variable('stabilizer.yaw')
        try:
            self.cf.log.add_config(log_config)
            log_config.data_received_cb.add_callback(self.update_labels)
            log_config.start()
        except KeyError as e:
            print(f'Could not start log config: {str(e)}')

    # when crazyflie disconnects, it runs this callback function
    def disconnected(self, URI):
        print(f'Disconnected from {URI}')
        sys.exit(1)

    # update the labels of the GUI
    def update_labels(self, timestamp, data, logconf):
        for key, value in data.items():
            self.labels[key].setText(f'{key}: {value:.2f}')

    # close the link when the window is closed
    def closeEvent(self, event):
        if self.cf is not None:
            self.cf.close_link()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FlyCrazyflie(URI)
    win.show()
    sys.exit(app.exec_())
