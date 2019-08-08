import tensorflow.keras
import numpy as np
import sklearn.metrics
import threading
import wandb


class Metrics(tensorflow.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size, labels):
        self.labels = labels
        self.epoch_count = 0
        self.validation_data = val_data
        self.batch_size = batch_size
        self.val_count = 0
        self.trim_start = 0
        self.trim_end = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        batches = len(self.validation_data)
        total = batches * self.batch_size
        val_predict = np.zeros(total)
        val_true = np.zeros(total)
        class_map = {}
        label_index = 0
        for label in self.labels:
            class_map[label_index] = label
            label_index += 1
        re_map = {label:index for index, label in class_map.items()}
        for batch in range(batches):
            thread1 = threading.Thread(target=self.parse_batch(batch, val_true, val_predict))
            thread1.start()
            thread1.join()

        val_predict = np.delete(val_predict, np.s_[self.trim_start:self.trim_end])
        val_true = np.delete(val_true, np.s_[self.trim_start:self.trim_end])
        self.val_count = 0
        label_predict = [class_map[i] for i in val_predict]
        label_true = [class_map[i] for i in val_true]

        # possibly use different average param
        _val_f1 = sklearn.metrics.f1_score(label_true, label_predict, labels=self.labels, average=None)
        _val_recall = sklearn.metrics.recall_score(label_true, label_predict, labels=self.labels, average=None)
        _val_precision = sklearn.metrics.precision_score(label_true, label_predict, labels=self.labels, average=None)
        for label in self.labels:
            filename = label + '.csv'
            csv_data = [['precision', 'recall', 'F1']]
            if not os.path.exists(os.path.join(project_home, filename)):
                with open(os.path.join(project_home, filename), 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(csv_data)
                csv_file.close()
            f1_log = {label+'_f1':_val_f1[re_map[label]]}
            precision_log = {label+'_precision':_val_precision[re_map[label]]}
            recall_log = {label+'_recall': _val_recall[re_map[label]]}
            wandb.log(f1_log, step=epoch)
            wandb.log(precision_log, step=epoch)
            wandb.log(recall_log, step=epoch)
            with open(os.path.join(project_home, filename), 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows([[precision_log, recall_log, f1_log]])
            csv_file.close()
        return

    def parse_batch(self, batch, val_true, val_predict):
        try:
            xVal, yVal = next(self.validation_data)
        except StopIteration:
            return

        for i in range(self.batch_size):
            try:
                val_predict[batch * self.batch_size + i] = np.asarray(self.model.predict_classes(xVal))[i]
                val_true[batch * self.batch_size + i] = np.argmax(yVal, axis=1)[i]
                self.val_count += 1
            except IndexError:
                self.trim_start = self.val_count
                self.trim_end = self.trim_start + self.batch_size - i
                return
