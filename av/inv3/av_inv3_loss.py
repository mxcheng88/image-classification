import keras
import argparse
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve, auc, confusion_matrix, \
    average_precision_score, classification_report, f1_score, precision_score, classification_report

# set GPU memory
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

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
 
num_classes = 2
img_rows, img_cols = 768, 768
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


def load_imagedata():
    none_av_number = [1, 15, 24, 29, 30, 31, 36, 41, 48, 49, 53, 54, 58, 60, 63, 64, 66, 68, 70, 90, 97, 98, 102]
    av_positive_number = [2, 25, 28, 42, 79, 81, 83, 84, 87, 88, 93, 101, 103]
    image, label = [], []
    i = 0
    p = 0
    x = (1024 - 768) / 2
    y = 0
    w = (1024 - 768) / 2 + 768
    h = 768
    while i < 104:
        if i + 1 not in none_av_number:
            img = Image.open('/home/admin301/chengmx/images/Figure' + '{0:0>3}'.format(i + 1) + '.JPG')
            img = img_to_array(img.crop((x, y, w, h)))
            image.append(img)
            if i + 1 in av_positive_number:
                label.append(1)
            else:
                label.append(0)
            #print('i+1:', i + 1, 'p:', p, 'label:', label[-1])
            p += 1
        i += 1
    print('length of av_images', len(image))
    return image, label

def over_sampling(x_train, x_label):
    train, label, label_count, index_label = [], [], [], []
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
            # history.append({'val_f1':_val_f1,'val_recall':_val_recall,'val_precision':_val_precision})
            '''
            history.append
            history.append(_val_acc_score)
            history.append(_val_f1)
            history.append(_val_recall)
            history.append(_val_precision)
            '''
            return

def customed_loss(y_true, y_pred):
    yprediction = tf.nn.softmax(y_pred)
    coe = tf.constant([1.0, 14.0])
    y_coe = y_true * coe
    loss = -tf.reduce_mean(y_coe * tf.log(yprediction))
    return loss


if __name__ == '__main__':

    print("========================================")
    print("MODEL: Inv3")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))
    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    base_model=InceptionV3(weights=None,include_top=True,input_shape=(img_rows, img_cols, img_channels),classes=2)    
    inv3 = Model(inputs=[base_model.input], outputs=[base_model.output])    

    # print model architecture if you need.
    #plot_model(inv3,to_file='./av_train_record/inv3.png',show_shapes=True)
    #print(inv3.summary())
    print("== LOADING DATA... ==")
    # load data
    num_classes = 2
    images, labels = load_imagedata()
    cvscores, label = [], []
    sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    for train, test in sfolder.split(images, labels):
        print('Train: %s \n test: %s' % (train, test))
        print(" ")
        print("over_sampling: ")
        # over_sampling
        #(train, label) = over_sampling(train, (np.array(labels)[train].tolist()))
        print('length of the train_images:', len(train), 'length of the train_labels:', len(label))
        print('Train: %s \n label: %s' % (train, label))
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

        metric = Metric()
        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        #inv3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', precision, recall, f1score])
        #inv3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        inv3.compile(loss=customed_loss, optimizer=sgd, metrics=['accuracy'])
        # set callback, write_images=1
        #cbks = [TensorBoard(log_dir='./inv3_train_record/av_inv3_{}/'.format(args.dataset),histogram_freq=1),LearningRateScheduler(scheduler)]   
        cbks = [metric, TensorBoard(log_dir='./inv3_train_record/av_inv3_customed_loss/',                            histogram_freq=1),                LearningRateScheduler(scheduler)]

        # dump checkpoint if you need.(add it to cbks)
        # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

        # set data augmentation
        print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        datagen.fit(x_train)

        # start training
        history = inv3.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                       steps_per_epoch=iterations,
                                       epochs=epochs,
                                       callbacks=cbks,
                                       validation_data=(x_test, y_test))
        # Predict the value from the validation dataset
        y_pred = inv3.predict(x_test)
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
        
        plt.plot(history.history['acc'],'b--')
        plt.plot(history.history['loss'],'k--')
        plt.plot(history.history['val_acc'],'y-')
        plt.plot(history.history['val_loss'],'m--')
        plt.plot(metric.val_f1s,'r.-')
        plt.plot(metric.val_precisions,'g-')
        plt.plot(metric.val_recalls,'c-')

        plt.title('Inv3 model report')
        plt.ylabel('evaluation')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy','train_loss', 'val_accuracy','val_loss','val_f1-score','val_precisions','val_recalls'], loc='lower right')
        plt.savefig('./customed_loss.png')
        #plt.show()
        
        print('\n')
        print (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        print('The current running python file is:av_inv3_customed_loss.py,83:21,validation_data=(x_test, y_test),batchsize=8,epochs=150,sgd,customed_loss')
        print('\n')
        # evaluate the model
        scores = inv3.evaluate(x_test, y_test, verbose=0)
        inv3.save('./inv3_train_record/av_inv3_customed_loss.h5')
        print("%s: %.2f%%" % (inv3.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        break