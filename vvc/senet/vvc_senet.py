import keras
import math
import time
import argparse
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from mycallback import MyTensorBoard
from PIL import Image
from matplotlib import pyplot as plt
from keras.layers import Lambda, concatenate
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.datasets import cifar10, cifar100
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, multiply, Reshape
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Sequential,Model
from keras import optimizers, regularizers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve, auc, confusion_matrix, \
    average_precision_score, classification_report, f1_score, precision_score, classification_report

# set GPU memory
if ('tensorflow' == K.backend()):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8, metavar='NUMBER',
                    help='batch size(default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='NUMBER',
                    help='epochs(default: 200)') 
parser.add_argument('-d', '--dataset', type=str, default="Pathological Images", metavar='STRING',
                    help='dataset. (default: cifar10)')

args = parser.parse_args()

cardinality        = 4          # 4 or 8 or 16 or 32
base_width         = 64
inplanes           = 64
expansion          = 4
 
num_classes = 2
img_rows, img_cols = 768,768
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 68 // batch_size + 1
weight_decay = 1e-4



def scheduler(epoch):
    if epoch <50:
        return 0.001
    if epoch < 100:
        return 0.0001
    return 0.00001

'''
def load_imagedata():

    vvc_positive_number = [4, 8, 22, 51, 56, 61, 80]
    image=[]
    label=[]
    p = 0
    x = (1024-768)/2
    y = 0
    w = (1024-768)/2+768
    h = 768
    i = 0
    while i < 104:
        img = img_to_array(Image.open('/home/admin301/chengmx/images/Figure' + '{0:0>3}'.format(i + 1) + '.JPG').crop((x, y, w, h)))
        image.append(img)
        if i + 1 in vvc_positive_number:
            label.append(1)
        else:
            label.append(0)
        i += 1
    return image,label
'''
def resnext(img_input,classes_num):
    global inplanes
    def add_common_layer(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def group_conv(x,planes,stride):
        h = planes // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:,:,:, i * h : i * h + h])(x)
            groups.append(Conv2D(h,kernel_size=(3,3),strides=stride,kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),padding='same',use_bias=False)(group))
        x = concatenate(groups)
        return x

    def residual_block(x,planes,stride=(1,1)):

        D = int(math.floor(planes * (base_width/64.0)))
        C = cardinality

        shortcut = x
        
        y = Conv2D(D*C,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(shortcut)
        y = add_common_layer(y)

        y = group_conv(y,D*C,stride)
        y = add_common_layer(y)

        y = Conv2D(planes*expansion, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(y)
        y = add_common_layer(y)

        if stride != (1,1) or inplanes != planes * expansion:
            shortcut = Conv2D(planes * expansion, kernel_size=(1,1), strides=stride, padding='same', kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
            shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)

        y = squeeze_excite_block(y)

        y = add([y,shortcut])
        y = Activation('relu')(y)
        return y
    
    def residual_layer(x, blocks, planes, stride=(1,1)):
        x = residual_block(x, planes, stride)
        inplanes = planes * expansion
        for i in range(1,blocks):
            x = residual_block(x,planes)
        return x

    def squeeze_excite_block(input, ratio=16):
        init = input
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
        filters = init._keras_shape[channel_axis]  # infer input number of filters
        se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)  # determine Dense matrix shape

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        x = multiply([init, se])
        return x

    def conv3x3(x,filters):
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return add_common_layer(x)

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)


    # build the resnext model    
    x = conv3x3(img_input,64)
    x = residual_layer(x, 3, 64)
    x = residual_layer(x, 3, 128,stride=(2,2))
    x = residual_layer(x, 3, 256,stride=(2,2))
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

def load_imagedata():

    vvc_positive_number = [4, 8, 22, 51, 56, 61, 80]
    image=[]
    label=[]
    p = 0
    x = (1024-768)/2
    y = 0
    w = (1024-768)/2+768
    h = 768
    i = 0
    while i < 104:
        img = img_to_array(Image.open('/home/admin301/chengmx/images/Figure' + '{0:0>3}'.format(i + 1) + '.JPG').crop((x, y, w, h)))
        image.append(img)
        if i + 1 in vvc_positive_number:
            label.append(1)
        else:
            label.append(0)
        i += 1
    return image,label
    
def over_sampling(x_train, x_label):
    train, label, label_count, index_label = [], [], [], []
    print('x_label:',x_label)
    for i in range(0, 3):
        if i in x_label:
            for z in range(0, len(x_label)):
                if x_label[z] == i:
                    index_label.append(z)
            label_count.append(x_label.count(i))
    m = label_count.index(max(label_count))
    x_label.sort()
    for i in range(0, len(label_count)):
        n = x_label.index(i)
        if i != m:
            rannum = np.random.randint(100, size=label_count[m]) % label_count[i]
            for j in rannum:
                train.append(x_train[index_label[n + j]])
                label.append(i)
        else:
            for z in range(0, max(label_count)):
                train.append(x_train[index_label[n + z]])
                label.append(i)
    return (train, label)


class Metric(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.acc_score = []


    def on_epoch_end(self, epoch, logs={}):
        if (self.validation_data):
            # print('shape of self.validation_data:',self.validation_data[1])
            val_predict = self.model.predict(self.validation_data[0])
            # val_predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
            print('length of val_predict:', len(val_predict))
            # self.model.predict(self.validation_data[0])
            # val_predict=(np.asarray(self.model.predict(self.model.validation_data[0]))).round(),average='micro'
            val_targ = self.validation_data[1]
            print('length of val_truth:', len(val_targ))
            _val_acc_score = accuracy_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1))
            _val_f1 = f1_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1))
            _val_recall = recall_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1))
            _val_precision = precision_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1))
            self.acc_score.append(_val_acc_score)
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            print('_val_acc_score: %.4f--val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (
                _val_acc_score, _val_f1, _val_precision, _val_recall))
            print('confusion_matrix:\n', confusion_matrix(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1)))
            target_names = ['class 0', 'class 1']
            print(classification_report(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1),
                                        target_names=target_names))
            return


def customed_loss(y_true, y_pred):
    yprediction = tf.nn.softmax(y_pred)
    coe = tf.constant([1.0, 14.0])
    y_coe = y_true * coe
    loss = -tf.reduce_mean(y_coe * tf.log(yprediction))
    return loss


if __name__ == '__main__':

    print("========================================")
    print("MODEL: SeNet")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))
    print("== DONE! ==\n== BUILD MODEL... ==")
    num_classes = 2
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = resnext(img_input,num_classes)
    senet    = Model(img_input, output)
    plot_model(senet,to_file='./senet.png',show_shapes=True)
    #print(senet.summary())
    print("== LOADING DATA... ==")
    # load data

    images, labels = load_imagedata()
    print('labels:',labels,'\n')
    cvscores, label = [], []
    metrics = Metric()
    #Mytensorboard=MyTensorBoard(log_dir='./VVC_train_record/senet_cifar10_{}/'.format(args.dataset), input_images=train_loader.next(), batch_size=batch_size, update_features_freq=50, write_features=True, write_graph=True, update_freq='batch')

    sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in sfolder.split(images, labels):
        print('Train: %s \n test: %s' % (train, test))
        print(" ")
        print("over_sampling: ")
        # over_sampling
        #(train, label) = over_sampling(train, (np.array(labels)[train].tolist()))
        print('length of the train_images:', len(train), 'length of the train_labels:', len(label))
        print(" ")
        x_train = np.array(images)[train]
        y_train = np.array(labels)[train]
        x_test = np.array(images)[test]
        y_test = np.array(labels)[test]

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        print("== DONE! ==\n== COLOR PREPROCESSING... ==")
        # color preprocessing
        x_train /= 255 
        x_test /= 255
        
        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        senet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #senet.compile(loss=customed_loss, optimizer='rmsprop', metrics=['accuracy', precision, recall, f1score])
        #senet.compile(loss=customed_loss, optimizer=sgd, metrics=['accuracy'])
        #set callback ,write_images=1
        cbks = [metrics,TensorBoard(log_dir='./VVC_train_record/senet_cifar10_{}/'.format(args.dataset),histogram_freq=1), LearningRateScheduler(scheduler)]   
        #cbks = [metrics,Mytensorboard,  LearningRateScheduler(scheduler)]

        # dump checkpoint if you need.(add it to cbks)
        # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

        # set data augmentation
        print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        datagen.fit(x_train)
        iterations = len(x_train) // batch_size + 1
        # start training
        history = senet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                       steps_per_epoch=iterations,
                                       epochs=epochs,
                                       callbacks=cbks,
                                       validation_data=(x_test, y_test))
        # Predict the value from the validation dataset
        y_pred = senet.predict(x_test)
        # Convert predictions classes to one hot vectors
        y_pred_label = np.argmax(y_pred, axis=1)
        # Convert validation observations to one hot vectors
        y_true_label = np.argmax(y_test, axis=1)

        print('AUC1 :', roc_auc_score(y_true_label, y_pred_label))
        #print('AUC2 :', roc_auc_score(y_test, y_pred))
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(y_true_label, y_pred_label)
        # plot the confusion matrix
        print('y_true_label:', y_true_label)
        print('y_pred_label:', y_pred_label)
        print('confusion_matrix:\n', confusion_mtx)

        # evaluate the model
        scores = senet.evaluate(x_test, y_test, verbose=0)
        print("%s: %.2f%%" % (senet.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        senet.save('./VVC_train_record/senet_cifar10_{}.h5'.format(args.dataset))
        print('\n')
        print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        print('The current running python file is:vvc_senet.py,83:21,validation_data=(x_test, y_test),batchsize=8,epochs=100,categorical_crossentropy ,sgd')
        print('\n')
        print(history.history.keys())
        plt.plot(history.history['acc'],'b--')
        plt.plot(history.history['val_acc'],'y-')
        plt.plot(metrics.val_f1s,'r.-')
        plt.plot(metrics.val_precisions,'g-')
        plt.plot(metrics.val_recalls,'c-')
        
        plt.title('SeNet model report')
        plt.ylabel('evaluation')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy', 'val_accuracy','val_f1-score','val_precisions','val_recalls'], loc='lower right')
        plt.savefig('results/result_acc.png')
        plt.show()
        break