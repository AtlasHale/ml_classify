import tensorflow.keras
import numpy as np
import sklearn.metrics
import threading

class Metrics(tensorflow.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size, labels):
        self.labels = labels
        self.epoch_count = 0
        self.validation_data = val_data
        self.batch_size = batch_size
        self.val_count = 0

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1_overall = []
        self.val_recalls_overall = []
        self.val_precisions_overall = []

    def on_epoch_end(self, epoch, logs={}):
        print(f'End Of Epoch {self.epoch_count} Report:\n')
        self.epoch_count += 1
        batches = len(self.validation_data)
        total = batches * self.batch_size
        print(f'batches: {batches}')
        print(f'total number of possible validation images: {total}')
        val_predict = np.zeros(total)
        val_true = np.zeros(total)
        class_map = {}
        label_index = 0
        for label in self.labels:
            class_map[label_index] = label
            label_index += 1
        print (class_map)

        for batch in range(batches):
            thread1 = threading.Thread(target=self.parse_batch(batch, val_true, val_predict))
            thread1.start()
            thread1.join()
            # for true_value in yVal:
            #    val_true[batch] = np.argmax()
            # print(scipy.stats.mode(np.asarray(self.model.predict_classes(xVal))))
            # val_predict[batch * self.batch_size : (batch+1) * self.batch_size] = scipy.stats.mode(np.asarray(self.model.predict_classes(xVal)))
            #val_true[batch * self.batch_size : (batch+1) * self.batch_size] = yVal
        print(val_predict, val_true, sep='\n')
        val_predict = val_predict[:self.val_count]
        val_true = val_true[:self.val_count]
        print('trimmed lists', val_predict, val_true, sep='\n')
        self.val_count = 0
        # print(sklearn.metrics.classification_report(
        #     val_predict,
        #     val_true,
        #     labels=[i for i in range(len(self.labels))],
        #     target_names=self.labels))
        # TODO: fix sklearn metrics to have correct label and average params
        # _val_f1 = sklearn.metrics.f1_score(val_true, val_predict, labels=self.labels, average='weighted')
        # _val_recall = sklearn.metrics.recall_score(val_true, val_predict, labels=self.labels, average='weighted')
        # _val_precision = sklearn.metrics.precision_score(val_true, val_predict, labels=self.labels, average='weighted') # possibly something besides none (binary, etc)
        # _val_f1_overall = sklearn.metrics.f1_score(val_true, val_predict, labels=self.labels)
        # _val_recall_overall = sklearn.metrics.recall_score(val_true, val_predict, labels=self.labels,)
        # _val_precision_overall = sklearn.metrics.precision_score(val_true, val_predict, labels=self.labels,)
        #
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        # self.val_f1s_overall.append(_val_f1_overall)
        # self.val_recalls_overall.append(_val_recall_overall)
        # self.val_precisions_overall.append(_val_precision_overall)
        # print(f'Epoch: {self.epoch_count} val_f1: {_val_f1}, val_precision: {_val_precision} — val_recall {_val_recall}')
        return

    def parse_batch(self, batch, val_true, val_predict):
        try:
            xVal, yVal = next(self.validation_data)
        except StopIteration:
            return

        print(f'Actual Labels\n{yVal}')
        print(f'Printing the model predictions:\n{np.asarray(self.model.predict(xVal)).round()}')
        print(f'Printing the model predicted classes:\n{self.model.predict_classes(xVal)}\n')
        for i in range(self.batch_size):
            try:
                val_predict[batch * self.batch_size + i] = np.asarray(self.model.predict_classes(xVal))[i]
                val_true[batch * self.batch_size + i] = np.argmax(yVal, axis=1)[i]
                self.val_count += 1
            except IndexError:
                print('Caught Index Error, reached end of validation data')
                return