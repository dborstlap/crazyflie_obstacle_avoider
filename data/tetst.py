import tensorflow as tf

filename = "/Users/dborstlap/work/drones/code/crazyflie_obstacle_avoider/data/images/cyberzoo_set1/img_20240123_172544_forward.png"
input_shape = [244,324,1]

image_file = tf.io.read_file(filename)
image = tf.io.decode_png(image_file, channels = 1)
image = tf.image.resize(image, input_shape[:2])
image = tf.cast(image, tf.float32)


print(image.shape)
print(image)