import tensorflow as tf
import utils
import numpy as np
import scipy.misc
import scipy.io as sio
import os
import random
TRAIN_STEPS = 10000
n_class = 17
Batch_Size = 100


PATH = '/home/abc/TiRan/WorkSpace/Python/python3/TF_NET/17flowers/jpg'
VALIDATAION_PERCENTAGE = 10

TEST_PERCENTAGE = 10


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
    print(len(train),len(val),len(test))
    print(val)
    print(test)
    random.shuffle(train)
    print(train)
    return train,test,val



# get_image_list()
# exit(0)
with open("./models/vgg16.tfmodel", mode='rb') as f:
  fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)


graph = tf.get_default_graph()
print ("graph loaded from disk")


# for i in graph.get_operations():
#   print(i)
# cat = utils.load_image("/home/abc/TiRan/WorkSpace/Python/python3/TF_NET/images/20170523221658304.jpeg")


def get_final_tensor(input_tensor,shape,n_class):
    with tf.variable_scope('final_train_ops'):
        w = tf.get_variable('w',shape=shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b',shape=[n_class],initializer=tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor,w),b)
        final_tensor = tf.nn.softmax(logits)
    return final_tensor,logits


def get_img_feature(sess,image,image_input_tensor,bottleneck_tensor):
    return sess.run(bottleneck_tensor,feed_dict={image_input_tensor:image})


def get_val_test_tensors(sess,datas,image_input_tensor,bottleneck_tensor):
    image_tensors = []
    image_labels = []
    for data in datas:
        label_index, img_path = (data).split(':')
        label_truth = np.zeros(n_class, np.float32)
        label_truth[int(label_index)] = 1.0
        image = np.expand_dims(utils.load_image(img_path), axis=0)
        image_tensor = get_img_feature(sess, image, image_input_tensor, bottleneck_tensor)
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
        # img_path = train[image_index][1:]
        label_truth = np.zeros(n_class, np.float32)
        label_truth[int(label_index)] = 1.0

        image = np.expand_dims(utils.load_image(img_path), axis=0)
        image = get_img_feature(sess,image,image_input_tensor,bottleneck_tensor)

        labels.append(label_truth)
        data.append(image[0])
        # print(np.array(data).shape)
    return data,labels





def main():
    image_input_tensor = tf.placeholder("float", [None, 224, 224, 3])
    # bottleneck out is the last layer input
    bottleneck_input = tf.placeholder(tf.float32, shape=[None, n_class],name='BottleNeckInput')
    label_input_tensor = tf.placeholder(tf.float32,shape=[None,n_class],name='GroundTruthInput')

    tf.import_graph_def(graph_def, input_map={"images": image_input_tensor})

    # Softmax
    # prob_tensor = graph.get_tensor_by_name("import/prob:0")
    # bottleneck layer
    bottleneck_tensor = graph.get_tensor_by_name('import/Relu_1:0')

    train,test,val = get_image_list()

    shape = [4096,n_class]
    final_tensor, logits = get_final_tensor(bottleneck_tensor,shape,n_class)

    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_input_tensor)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)

    train_ops = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entroy_mean)

    with tf.name_scope('evaluation'):
        corrent_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(label_input_tensor,1))
        eval_step = tf.reduce_mean(tf.cast(corrent_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        print ("variables initialized")
        for i in range(TRAIN_STEPS):
            # get train data and its labels
            bottleneck_input,labels_input = get_input_tensor(sess,train,image_input_tensor,bottleneck_tensor)
            # print(np.array(bottleneck_input).shape)
            sess.run(train_ops,feed_dict={bottleneck_tensor:bottleneck_input,label_input_tensor:labels_input})
            print(sess.run(cross_entroy_mean,feed_dict={bottleneck_tensor:bottleneck_input,label_input_tensor:labels_input}))

main()