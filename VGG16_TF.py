import tensorflow as tf
import utils
import numpy as np
import scipy.misc
import scipy.io as sio
with open("./models/vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

images = tf.placeholder("float", [None, 224, 224, 3])

tf.import_graph_def(graph_def, input_map={ "images": images })
print ("graph loaded from disk")

graph = tf.get_default_graph()

for i in graph.get_operations():
  print(i)
cat = utils.load_image("/home/abc/TiRan/WorkSpace/Python/python3/TF_NET/images/20170523221658304.jpeg")


with tf.Session() as sess:
  init = tf.initialize_all_variables()
  sess.run(init)
  print ("variables initialized")

  batch = np.expand_dims(cat, axis=0)
  # print(batch.shape)
  assert batch.shape == (1, 224, 224, 3)

  feed_dict = { images: batch }

  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  prob = sess.run(prob_tensor, feed_dict=feed_dict)

  preds = tf.nn.softmax(prob)

  predsSortIndex = np.argsort(-preds[0].eval())
  for i in range(5):  ##输出前3种分类
    nIndex = predsSortIndex
    problity = prob[0][nIndex[i]]  ##某一类型概率
    print(problity)
utils.print_prob(prob[0])