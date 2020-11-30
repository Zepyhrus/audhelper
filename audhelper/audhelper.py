__author__ = 'sherk'
__version__ = '0.2.0'
"""

"""

from os.path import join
from itertools import chain

import numpy as np
import logging

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib import slim

from audhelper.read import nread

_EPSILON = tf.keras.backend.epsilon()


def sparse_softmax_categorical_focal_loss(
  logits, labels, gamma=2.0, alpha=1.0
) -> tf.Tensor:
  r"""Focal loss function for multiclass classification with integer labels.
  This loss function generalizes multiclass softmax cross-entropy by
  introducing a hyperparameter called the *focusing parameter* that allows
  hard-to-classify examples to be penalized more heavily relative to
  easy-to-classify examples.
  See :meth:`~focal_loss.binary_focal_loss` for a description of the focal
  loss in the binary setting, as presented in the original work [1]_.
  In the multiclass setting, with integer labels :math:`y`, focal loss is
  defined as
  .. math::
    L(y, \hat{\mathbf{p}})
    = -\left(1 - \hat{p}_y\right)^\gamma \log(\hat{p}_y)
  where
  * :math:`y \in \{0, \ldots, K - 1\}` is an integer class label (:math:`K`
    denotes the number of classes),
  * :math:`\hat{\mathbf{p}} = (\hat{p}_0, \ldots, \hat{p}_{K-1})
    \in [0, 1]^K` is a vector representing an estimated probability
    distribution over the :math:`K` classes,
  * :math:`\gamma` (gamma, not :math:`y`) is the *focusing parameter* that
    specifies how much higher-confidence correct predictions contribute to
    the overall loss (the higher the :math:`\gamma`, the higher the rate at
    which easy-to-classify examples are down-weighted).
  The usual multiclass softmax cross-entropy loss is recovered by setting
  :math:`\gamma = 0`.
  Parameters
  ----------
  labels : (batch_size) tensor-like
    Integer class labels.
  logits : (bach_size, num_cls) tensor-like
    output, need to do softmax inside
  gamma : float or tensor-like of shape (K,)
    The focusing parameter :math:`\gamma`. Higher values of `gamma` make
    easy-to-classify examples contribute less to the loss relative to
    hard-to-classify examples. Must be non-negative. This can be a
    one-dimensional tensor, in which case it specifies a focusing parameter
    for each class.
  alpha :
    compensate ratio for loss reduction
  Returns
  -------
  :class:`tf.Tensor`
    The focal loss for each example.
  Examples
  --------
  This function computes the per-example focal loss between a one-dimensional
  integer label vector and a two-dimensional prediction matrix:
  >>> import numpy as np
  >>> from focal_loss import sparse_categorical_focal_loss
  >>> labels = [0, 0, 0, 1, 2]
  >>> preds = [[0.1, 0.8, 0.1], [0.5, 0.4, 0.1], [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
  >>> pass
  Warnings
  --------
  This function does not reduce its output to a scalar, so it cannot be passed
  to :meth:`tf.keras.Model.compile` as a `loss` argument. Instead, use the
  wrapper class :class:`~focal_loss.SparseCategoricalFocalLoss`.
  References
  ----------
  [1] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár. Focal loss for
    dense object detection. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 2018.
    (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
    (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)
  See Also
  --------
  :meth:`~focal_loss.SparseCategoricalFocalLoss`
    A wrapper around this function that makes it a
    :class:`tf.keras.losses.Loss`.
  """
  _probs = tf.nn.softmax(logits)
  _xloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  
  _mask = tf.one_hot(labels, depth=_probs.shape[1]) # TODO: add num_class support
  _p = tf.add(tf.reduce_max(tf.multiply(_probs, _mask), axis=1), _EPSILON)


  _focal_modulation = - alpha * tf.log(_p) * ( (1 - _p) ** gamma )
  _loss = _focal_modulation * _xloss

  return _loss

class BaseKWS(object):
  def __init__(self, cfg, training=False):
    self.initailized          = False
    self.summary_dir          = cfg['summary_dir'] if 'summary_dir' in cfg else None
    self.training_dir         = cfg['training_dir'] if 'training_dir' in cfg else None
    self.wanted_words         = ['_unknown_'] + cfg['wanted_words']
    self.num_classes          = len(self.wanted_words)
    self.samples              = cfg['samples']
    self.sample_rate          = cfg['sample_rate']
    self.loss_type            = cfg['loss_type']
    self.batch_size           = cfg['batch_size']
    self.learning_rates       = cfg['learning_rates']
    self.training_steps       = cfg['training_steps']
    self.eval_step_interval   = cfg['eval_step_interval']
    self.training             = training
    self.name                 = None

    # initialize graph and sess
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

  def initialize(self):
    with self.graph.as_default():
      # initialize self.loss
      if self.loss_type == 'Focal':
        self.loss = sparse_softmax_categorical_focal_loss
      elif self.loss_type == 'CE':
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits
      else:
        raise Exception('Unsupported loss type: %s' % self.loss_type)

      self.__audios = tf.placeholder(tf.float32, [None, self.samples])
      self.__labels = tf.placeholder(tf.int64, [None])

      self.__logits = self.module(self.__audios)
      self.__preds = tf.nn.softmax(self.__logits)

      self.__accuracy = tf.contrib.metrics.accuracy(tf.arg_max(self.__preds, 1), self.__labels)
      tf.summary.scalar('accuracy', self.__accuracy)

      if self.training:
        # we have only loss in training
        self.__loss = tf.reduce_mean(self.loss(logits=self.__logits, labels=self.__labels))
        tf.summary.scalar('loss', self.__loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope('train'), tf.control_dependencies(update_ops):
          self.__lr = tf.placeholder(tf.float32, [], name='learning_rate_input')
          self.__optimizer = tf.train.AdamOptimizer(self.__lr)
          self.__train_step = slim.learning.create_train_op(self.__loss, self.__optimizer)
        self.__global_step = tf.train.get_or_create_global_step()
        self.__saver = tf.train.Saver(tf.global_variables())

        # merge all summaries
        self.__summaries = tf.summary.merge_all()
        self.__train_writer = tf.summary.FileWriter(join(self.summary_dir, 'train'))
        self.__val_writer = tf.summary.FileWriter(join(self.summary_dir, 'val'))

      with self.sess.as_default():
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        # Parameter counts
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        print('Training parameter numbers: %d' % num_params)
    self.initailized = True
  
  def __infer(self, auds):
    return self.sess.run(self.__preds, feed_dict={self.__audios: auds})

  def infer(self, audio):
    if not self.initailized:
      logging.warning('Model not initailized!')
      self.initialize()

    aud = nread(
      data=audio,
      samples=self.samples, sample_rate=self.sample_rate,
      shuffle=False, aug=None
    ).reshape((-1, self.samples))

    return self.__infer(aud)
  
  def batch_infer(self, audios):
    if not self.initailized:
      logging.warning('Model not initailized!')
      self.initialize()
    
    # assume the audios are already normalized!
    return self.__infer(audios)



  def pred(self, audios):
    assert self.initailized, 'Model not initialized!'
    assert len(audios) <= self.batch_size, f'Input length {len(audios)} is larger than batch size{self.batch_size}'    

    auds = np.empty((len(audios), self.samples), dtype=np.float32)

    for i, audio in enumerate(audios):
      auds[i] = nread(
        data=audio,
        samples=self.samples, sample_rate=self.sample_rate,
        shuffle=False, aug=None
      )

    return self.sess.run(self.__preds, feed_dict={self.__audios: auds})

  def train(self, train_dataset, val_dataset=None):
    assert self.initailized, 'Model not initailized!'

    train_dataset.eval(self)

    best_accuracy = 0.
    lrs = list(chain.from_iterable([[lr] * ep for lr, ep in zip(self.learning_rates, self.training_steps)]))

    for epoch, learning_rate_value in enumerate(lrs):
      for _b_auds, _b_labels in train_dataset:
        train_summary, train_accuracy, train_loss, curr_step, _ = self.sess.run(
          [self.__summaries, self.__accuracy, self.__loss, self.__global_step, self.__train_step],
          feed_dict={
            self.__audios: _b_auds,
            self.__labels: _b_labels,
            self.__lr: learning_rate_value
          }
        )
        self.__train_writer.add_summary(train_summary, curr_step)
        print('%d-%d: %.2f-%.4f' % (epoch, curr_step, train_accuracy*100, train_loss))

        # validation
        if val_dataset and (curr_step + 1) % self.eval_step_interval == 0:
          total_accuracy = 0.
          total_loss = 0.

          for val_auds, val_labels in val_dataset:
            val_summary, val_accuracy, val_loss = self.sess.run(
              [self.__summaries, self.__accuracy, self.__loss],
              feed_dict={
                self.__audios: val_auds,
                self.__labels: val_labels
              }
            )
            total_accuracy += (val_accuracy * len(val_labels)) / val_dataset.size
            total_loss += (val_loss * len(val_labels)) / val_dataset.size
            self.__val_writer.add_summary(val_summary, curr_step)
          
          if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            save_dir = join(self.training_dir, '%d-%d' % (int(best_accuracy*1e4), curr_step))
            self.save_module(save_dir)
            print('Validation %d-%d: %.2f-%.4f' % (epoch, curr_step, total_accuracy*100, total_loss))
    save_dir = join(self.training_dir, 'last')
    self.save_module(save_dir)
      
  def test(self, test_dataset):
    assert self.initailized, 'Model not initailized!'

    test_dataset.eval(self)
    total_accuracy = 0.
    for test_auds, test_labels in test_dataset:
      test_accuracy = self.sess.run(
        self.__accuracy, feed_dict={self.__audios: test_auds, self.__labels: test_labels}
      )
      total_accuracy += (test_accuracy * len(test_labels))
    total_accuracy /= (test_dataset.size + _EPSILON)

    return total_accuracy

  def save_module(self, save_dir):
    raise NotImplementedError

  def get_name(self):
    return self.name

if __name__ == "__main__":
  pass






