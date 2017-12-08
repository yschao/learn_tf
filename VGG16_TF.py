import tensorflow as tf
import utils
import numpy as np
from tensorflow.python.platform import gfile
import scipy.misc
import scipy.io as sio
import os
import random
TRAIN_STEPS = 3000
n_class = 5
Batch_Size = 100
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


PATH = '/home/scyang/TiRan/WorkSpace/python/python3/learn_TF/transfer_learn/flower_photos'
VALIDATAION_PERCENTAGE = 10

TEST_PERCENTAGE = 10

FileNames = []
def get_image_list():
    train = []
    val = []
    test = []

    sub_dirs = os.listdir(PATH)
    for sub_dir in sub_dirs:
        # print(sub_dir)
        # os.path.isfile()
        path = os.path.join(PATH,sub_dir)
        if os.path.isdir(path):
            FileNames.append(sub_dir)
            images = os.listdir(path)
            for image in images:
                img_label_path = sub_dir +':'+ os.path.join(path,image)
                print(img_label_path)
                chance = np.random.randint(100)
                if chance < VALIDATAION_PERCENTAGE:
                    val.append(img_label_path)
                elif chance < (TEST_PERCENTAGE + VALIDATAION_PERCENTAGE):
                    test.append(img_label_path)
                else:
                    train.append(img_label_path)
    # print(len(train),len(val),len(test))
    # print(val)
    # print(test)
    random.shuffle(train)
    # print(train)

    return train,test,val



# get_image_list()
# exit(0)
with open("/media/scyang/scyang/personal/Database/model/vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

# for i in graph.get_operations():
#   print(i)
# cat = utils.load_image("/home/abc/TiRan/WorkSpace/Python/python3/TF_NET/images/20170523221658304.jpeg")


def get_final_tensor(input_tensor,shape,n_class):
    with tf.variable_scope('final_train_ops'):
        w = tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))
        b = tf.Variable(tf.zeros([n_class]))
        logits = tf.matmul(input_tensor,w) + b
        final_tensor = tf.nn.softmax(logits)
    return final_tensor,logits


def get_img_feature(sess,image,image_input_tensor,bottleneck_tensor,img_name):
    current_file = os.path.basename(__file__)
    bottleneck_path = os.path.join('/tmp' , current_file.split('.')[0])
    if not os.path.exists(bottleneck_path):
        os.mkdir(bottleneck_path)

    bottleneck_name = os.path.join(bottleneck_path,img_name + '.txt')
    if os.path.exists(bottleneck_name):
        with open(bottleneck_name,'r') as f:
            bottleneck_string = f.read()
        bottlenect_values = [[float(x) for x in bottleneck_string.split(',')]]

    else:
        bottlenect_values = sess.run(bottleneck_tensor,feed_dict={image_input_tensor:image})
        # print( bottlenect_values.shape)
        bottleneck_string = ','.join(str(x) for x in bottlenect_values[0])
        with open(bottleneck_name, 'w') as f:
            f.write(bottleneck_string)

    return bottlenect_values


def get_val_test_tensors(sess,datas,image_input_tensor,bottleneck_tensor):
    image_tensors = []
    image_labels = []
    for data in datas:
        label_index, img_path = (data).split(':')
        label_index = FileNames.index(label_index)
        label_truth = np.zeros(n_class, np.float32)
        label_truth[(label_index)] = 1.0
        image = np.expand_dims(utils.load_image(img_path), axis=0)
        image_tensor = get_img_feature(sess, image, image_input_tensor, bottleneck_tensor,os.path.basename(img_path))
        image_labels.append(label_truth)
        image_tensors.append(image_tensor[0])

    return image_tensor,image_labels

def get_input_tensor(sess,train,image_input_tensor,bottleneck_tensor):
    data = []
    labels = []
    img_nums = len(train)
    for i in range(Batch_Size):
        image_index = random.randrange(img_nums)
        label_index,img_path = (train[image_index]).split(':')
        label_index = FileNames.index(label_index)
        # print(label_index, "sdf")
        # img_path = train[image_index][1:]
        label_truth = np.zeros(n_class, np.float32)
        label_truth[int(label_index)] = 1.0

        image = np.expand_dims(utils.load_image(img_path), axis=0)
        image = get_img_feature(sess,image,image_input_tensor,bottleneck_tensor,os.path.basename(img_path))

        labels.append(label_truth)
        data.append(image[0])
    return data,labels

def main():
    train,test,val = get_image_list()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    image_input_tensor = tf.placeholder("float", [None, 224, 224, 3])
    # bottleneck out is the last layer input
    bottleneck_input = tf.placeholder(tf.float32, shape=[None, n_class],name='BottleNeckInput')
    label_input_tensor = tf.placeholder(tf.float32,shape=[None,n_class],name='GroundTruthInput')

    tf.import_graph_def(graph_def, input_map={"images": image_input_tensor})
    graph = tf.get_default_graph()

    print ("graph loaded from disk")

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    train_nums = len(train)
    decay_steps = int(train_nums/Batch_Size)
    print(train_nums,decay_steps)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Softmax
    # prob_tensor = graph.get_tensor_by_name("import/prob:0")
    # bottleneck layer
    bottleneck_tensor = graph.get_tensor_by_name('import/Relu_1:0')



    shape = [4096, n_class]
    final_tensor, logits = get_final_tensor(bottleneck_tensor,shape,n_class)

    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_input_tensor)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    train_ops = tf.train.GradientDescentOptimizer(lr).minimize(cross_entroy_mean)

    with tf.name_scope('evaluation'):
        corrent_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(label_input_tensor,1))
        eval_step = tf.reduce_mean(tf.cast(corrent_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print ("variables initialized")
        for i in range(TRAIN_STEPS):
            # get train data and its labels
            bottleneck_input,labels_input = get_input_tensor(sess,train,image_input_tensor,bottleneck_tensor)
            # print(np.array(bottleneck_input).shape)
            sess.run(train_ops,feed_dict={bottleneck_tensor:bottleneck_input,label_input_tensor:labels_input})

            if i % 50 ==0:
                train_accuracy = sess.run(eval_step, feed_dict={bottleneck_tensor : bottleneck_input,
                                                              label_input_tensor : labels_input})
                print('After %d step,the train accuracy is %.1f%%' % (i, train_accuracy))
            if i % 100 == 0:
                bottleneck_val_input,labels_val_input = get_val_test_tensors(sess,val,image_input_tensor,bottleneck_tensor)
                val_accuracy = sess.run(eval_step,feed_dict={bottleneck_tensor:bottleneck_val_input,
                                                             label_input_tensor:labels_val_input})
                print('After %d step,the val accuracy is %.1f%%' %(i,val_accuracy))

        bottleneck_test_input, labels_test_input = get_val_test_tensors(sess, test, image_input_tensor, bottleneck_tensor)
        test_accuracy = sess.run(eval_step, feed_dict={bottleneck_tensor: bottleneck_test_input,
                                                      label_input_tensor: labels_test_input})
        print('The TEST accuracy is .1f%%' %test_accuracy)

main()