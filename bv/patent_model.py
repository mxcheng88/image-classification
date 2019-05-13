import keras
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.initializers import he_normal
from keras.utils.vis_utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D,MaxPooling2D,Dropout,Flatten
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Sequential
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
img_rows, img_cols = 768, 1024
img_channels = 3
batch_size = args.batch_size
epochs = args.epochs
iterations = 68 // batch_size + 1
weight_decay = 1e-4


def scheduler(epoch):
    if epoch < 80:
        return 0.001
    if epoch < 100:
        return 0.0001
    return 0.00001

#Load bv label Images
def load_imagedata():
    bv_middle_number = [2, 8, 21, 28, 37, 42, 78, 93, 101]
    bv_positive_number = [17, 20, 25, 26, 45, 69, 81]

    image = []
    label = []
    i = 0
    p = 0
    x = (1024-768)/2
    y = 0
    w = (1024-768)/2+768
    h = 768
    i = 0
    while i < 104:
        #img = img_to_array(Image.open('/home/admin301/chengmx/images/Figure' + '{0:0>3}'.format(i + 1) + '.JPG').crop((x, y, w, h)))
        img = img_to_array(Image.open('/home/admin301/chengmx/images/Figure' + '{0:0>3}'.format(i + 1) + '.JPG'))
        image.append(img)
        if i + 1 in bv_positive_number:
            label.append(2)
        elif i + 1 in bv_middle_number:
            label.append(1)
        else:
            label.append(0)
        i += 1
    return image,label


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
            _val_f1 = f1_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1), average='micro')
            _val_recall = recall_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1), average='micro')
            _val_precision = precision_score(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1), average='micro')
            self.acc_score.append(_val_acc_score)
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            print('_val_acc_score: %.4f--val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (
                _val_acc_score, _val_f1, _val_precision, _val_recall))
            print('confusion_matrix:\n', confusion_matrix(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1)))
            target_names = ['class 0', 'class 1', 'class 2']
            print(classification_report(np.argmax(val_targ, axis=1), np.argmax(val_predict, axis=1),
                                        target_names=target_names))
            return

def customed_loss(y_true, y_pred):
    yprediction = tf.nn.softmax(y_pred)
    coe = tf.constant([1.0, 10.0, 13.0])
    y_coe = y_true * coe
    loss = -tf.reduce_mean(y_coe * tf.log(yprediction))
    return loss


if __name__ == '__main__':

    print("========================================")
    print("MODEL: Patent Model")
    print("BATCH SIZE: {:3d}".format(batch_size))
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))
    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    model = Sequential()
   
    model.add(Conv2D(96, (3,3), strides=(2,2),kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), activation='relu', padding='same', input_shape=(img_rows, img_cols, img_channels)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())   
    
    model.add(Conv2D(128, (3,3), strides=(2,2),kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), activation='relu', padding='same'))    
    model.add(Conv2D(256, (3,3),kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), activation='relu', padding='same'))  
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(256, (3,3),kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
        
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(3, activation='softmax'))
    # print model architecture if you need.
    #plot_model(model,to_file='./bv_train_record/mymodel.png',show_shapes=True)
    print(model.summary())
   
    print("== LOADING DATA... ==")
    metric = Metric()
    # load data
    num_classes = 3
    images, labels = load_imagedata()
    cvscores, label = [], []
    sfolder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    for train, test in sfolder.split(images, labels):
        print('Train: %s \n test: %s' % (train, test))
        
        print(" ")
        print("over_sampling: ")
        # over_sampling
        (train, label) = over_sampling(train, (np.array(labels)[train].tolist()))
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
        
        # set optimizer
        sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        #model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # set callback
        cbks = [metric,TensorBoard(log_dir='./bv_train_record/bv_patent_model_{}/'.format(args.dataset),     histogram_freq=1),                LearningRateScheduler(scheduler)]
        #cbks = [metric,                TensorBoard(log_dir='./bv_train_record/bv_mymodel_{}/'.format(args.dataset), write_images=1,                            histogram_freq=1),                LearningRateScheduler(scheduler)]

        # dump checkpoint if you need.(add it to cbks)
        # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

        # set data augmentation
        print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

        datagen.fit(x_train)

        # start training
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                       steps_per_epoch=iterations,
                                       epochs=epochs,
                                       callbacks=cbks,
                                       validation_data=(x_test, y_test))
        # Predict the value from the validation dataset
        y_pred = model.predict(x_test)
        # Convert predictions classes to one hot vectors
        y_pred_label = np.argmax(y_pred, axis=1)
        # Convert validation observations to one hot vectors
        y_true_label = np.argmax(y_test, axis=1)

        print('AUC1 :', roc_auc_score(y_true_label, y_pred_label))
        print('AUC2 :', roc_auc_score(y_test, y_pred))
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

        plt.title('customed model report')
        plt.ylabel('evaluation')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy','train_loss', 'val_accuracy','val_loss','val_f1-score','val_precisions','val_recalls'], loc='lower right')
        plt.savefig('./patent_model.png')
        
        # evaluate the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        model.save('./bv_train_record/bv_patentmodel_{}.h5'.format(args.dataset))   
        break   