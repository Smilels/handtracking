import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='frozen_inference_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='../logs/'
train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
