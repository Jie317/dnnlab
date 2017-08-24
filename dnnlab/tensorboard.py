import tensorflow as tf
from keras.callbacks import TensorBoard


class tensorBoard(TensorBoard):
    """Adapted tensorboard logger to track loss by sample.
    """

    def __init__(self,
                 log_dir='./logs',
                 histogram_freq=0,
                 batch_size=40960,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 track_by_samples=False):
        super(tensorBoard, self).__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads,
            write_images=write_images,
            embeddings_freq=embeddings_freq,
            embeddings_layer_names=embeddings_layer_names,
            embeddings_metadata=embeddings_metadata)
        self.track_by_samples = track_by_samples
        if self.track_by_samples:
            self.seen = 0

    def on_batch_end(self, batch, logs=None, count=True):
        if self.track_by_samples:
            batch_size = logs.get('size', 0)
            if count:
                self.seen += batch_size

            for k, v in logs.items():
                if k in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = v
                summary_value.tag = k
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        pass
