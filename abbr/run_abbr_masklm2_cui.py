import collections
import csv
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from bert import modeling
from bert import optimization
from bert import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

## Other parameters

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_string("dataset", "train", "which split to use (train/test)")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("batch_size", 8, "Total batch size.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer(
    "max_n_masks", 16,
    "Max number of [MASKS] possible")

flags.DEFINE_float("loss_margin", 0.1, "beta in margin loss")

flags.DEFINE_float("loss_cross_entropy", 0.0, "weights on cross entropy loss")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, group, text_left, text_right, label, negs):
        self.guid = guid
        self.group = group
        self.text_left = text_left
        self.text_right = text_right
        self.label = label
        self.negs = negs


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class TestInputFeatures(object):
    def __init__(self, guid, input_ids, input_mask, segment_ids, positions, masks_mask):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.positions = positions
        self.masks_mask = masks_mask


class TrainInputFeatures(object):
    def __init__(self, guid, input_ids, input_mask, segment_ids, positions, masks_mask, label,
                 input_ids_neg, input_mask_neg, segment_ids_neg, positions_neg, masks_mask_neg, label_neg):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.positions = positions
        self.masks_mask = masks_mask
        self.label = label
        self.input_ids_neg = input_ids_neg
        self.input_mask_neg = input_mask_neg
        self.segment_ids_neg = segment_ids_neg
        self.positions_neg = positions_neg
        self.masks_mask_neg = masks_mask_neg
        self.label_neg = label_neg


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class UMNProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_train_examples2(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_check.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = int(line[0])
            group = line[1]
            text_left = tokenization.convert_to_unicode(line[2])
            text_right = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[4])
            negs = [tokenization.convert_to_unicode(neg) for neg in line[5:] if neg.strip()]
            examples.append(
                InputExample(guid=guid, group=group, text_left=text_left, text_right=text_right, label=label,
                             negs=negs))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # Truncates tokens from the left end
            # tokens_a.pop()
        else:
            tokens_b.pop()


def _make_sentence(tokens_left, tokens_right, n_mask_or_label, max_seq_length, max_n_masks, tokenizer):
    tokens = []
    segment_ids = []
    positions = []
    ind = 0
    tokens.append("[CLS]")
    segment_ids.append(0)
    ind += 1
    for token in tokens_left:
        tokens.append(token)
        segment_ids.append(0)
        ind += 1
    if isinstance(n_mask_or_label, int):
        for _ in range(n_mask_or_label):
            tokens.append("[MASK]")
            segment_ids.append(0)
            positions.append(ind)
            ind += 1
    elif isinstance(n_mask_or_label, collections.abc.Iterable):
        for token in n_mask_or_label:
            tokens.append(token)
            segment_ids.append(0)
            positions.append(ind)
            ind += 1
    else:
        raise TypeError(f"n_mask_or_label is not iterable: {n_mask_or_label}")
    for token in tokens_right:
        tokens.append(token)
        segment_ids.append(0)
        ind += 1
    tokens.append("[SEP]")
    segment_ids.append(0)
    ind += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    masks_mask = [1.0] * len(positions)
    while len(positions) < max_n_masks:
        positions.append(0)
        masks_mask.append(0.0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(positions) == max_n_masks
    assert len(masks_mask) == max_n_masks

    return input_ids, input_mask, segment_ids, positions, masks_mask


def convert_single_example_train(ex_index, example, max_seq_length, max_n_masks, tokenizer, cui_to_label_train):
    tokens_left = tokenizer.tokenize(example.text_left)
    tokens_right = tokenizer.tokenize(example.text_right)
    tokens_pos = tokenizer.tokenize(cui_to_label_train[example.label]) if (example.label != 'CUI-less') else ['[UNK]']
    tokens_negs = [
        tokenizer.tokenize(cui_to_label_train[neg]) if (neg != 'CUI-less') else ['[UNK]']
        for neg in example.negs]
    if not tokens_negs:
        random_neg = example.label
        while random_neg == example.label:
            random_neg = random.choice(list(cui_to_label_train.keys()))
        tokens_negs = [tokenizer.tokenize(cui_to_label_train[random_neg]) if (random_neg != 'CUI-less') else ['[UNK]']]

    tokens_pos = tokens_pos[:max_n_masks]
    tokens_left_pos = tokens_left.copy()
    tokens_right_pos = tokens_right.copy()
    _truncate_seq_pair(tokens_left_pos, tokens_right_pos, max_seq_length - len(tokens_pos) - 2)

    input_ids, input_mask, segment_ids, positions, masks_mask = _make_sentence(
        tokens_left_pos, tokens_right_pos, len(tokens_pos), max_seq_length, max_n_masks, tokenizer)
    label = tokenizer.convert_tokens_to_ids(tokens_pos)
    while len(label) < max_n_masks:
        label.append(0)

    features = []
    for i, tokens_neg in enumerate(tokens_negs):
        tokens_neg = tokens_neg[:max_n_masks]
        tokens_left_neg = tokens_left.copy()
        tokens_right_neg = tokens_right.copy()
        _truncate_seq_pair(tokens_left_neg, tokens_right_neg, max_seq_length - len(tokens_neg) - 2)

        input_ids_neg, input_mask_neg, segment_ids_neg, positions_neg, masks_mask_neg = _make_sentence(
            tokens_left_neg, tokens_right_neg, len(tokens_neg), max_seq_length, max_n_masks, tokenizer)
        label_neg = tokenizer.convert_tokens_to_ids(tokens_neg)
        while len(label_neg) < max_n_masks:
            label_neg.append(0)

        # input_ids_sub_abbr, input_mask_sub_abbr, segment_ids_sub_abbr, positions_sub_abbr, masks_mask_sub_abbr = _make_sentence(
            # tokens_left_abbr, tokens_right_abbr, tokens_abbr, max_seq_length, max_n_masks, tokenizer)

        # input_ids_sub_pos, input_mask_sub_pos, segment_ids_sub_pos, positions_sub_pos, masks_mask_sub_pos = _make_sentence(
            # tokens_left_pos, tokens_right_pos, tokens_pos, max_seq_length, max_n_masks, tokenizer)

        # input_ids_sub_neg, input_mask_sub_neg, segment_ids_sub_neg, positions_sub_neg, masks_mask_sub_neg = _make_sentence(
            # tokens_left_neg, tokens_right_neg, tokens_neg, max_seq_length, max_n_masks, tokenizer)

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("positions: %s" % " ".join([str(x) for x in positions]))
            tf.logging.info("masks_mask: %s" % " ".join([str(x) for x in masks_mask]))
            tf.logging.info("label: %s" % " ".join([str(x) for x in label]))
            tf.logging.info("input_ids_neg: %s" % " ".join([str(x) for x in input_ids_neg]))
            tf.logging.info("input_mask_neg: %s" % " ".join([str(x) for x in input_mask_neg]))
            tf.logging.info("segment_ids_neg: %s" % " ".join([str(x) for x in segment_ids_neg]))
            tf.logging.info("positions_neg: %s" % " ".join([str(x) for x in positions_neg]))
            tf.logging.info("masks_mask_neg: %s" % " ".join([str(x) for x in masks_mask_neg]))
            tf.logging.info("label_neg: %s" % " ".join([str(x) for x in label_neg]))

        feature = TrainInputFeatures(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            positions=positions,
            masks_mask=masks_mask,
            label=label,
            input_ids_neg=input_ids_neg,
            input_mask_neg=input_mask_neg,
            segment_ids_neg=segment_ids_neg,
            positions_neg=positions_neg,
            masks_mask_neg=masks_mask_neg,
            label_neg=label_neg,
        )
        features.append(feature)
    return features


def convert_single_example_test(ex_index, example, max_seq_length, max_n_masks, tokenizer, cui_to_label_train):
    tokens_left = tokenizer.tokenize(example.text_left)
    tokens_right = tokenizer.tokenize(example.text_right)
    # example.negs is the list of all the candidate CUIs from training set for the abbr
    tokens_trains = [
        tokenizer.tokenize(cui_to_label_train[cui]) if (cui != 'CUI-less') else ['[UNK]']
        for cui in example.negs]

    features = []
    for tokens in tokens_trains:
        tokens = tokens[:max_n_masks]
        tokens_left_2 = tokens_left.copy()
        tokens_right_2 = tokens_right.copy()
        _truncate_seq_pair(tokens_left_2, tokens_right_2, max_seq_length - len(tokens) - 2)

        input_ids, input_mask, segment_ids, positions, masks_mask = _make_sentence(
            tokens_left_2, tokens_right_2, len(tokens), max_seq_length, max_n_masks, tokenizer)

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("positions: %s" % " ".join([str(x) for x in positions]))
            tf.logging.info("masks_mask: %s" % " ".join([str(x) for x in masks_mask]))

        feature = TestInputFeatures(
            guid=example.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            positions=positions,
            masks_mask=masks_mask
        )
        features.append(feature)

    return features


def file_based_convert_examples_to_features(examples, max_seq_length, max_n_masks, tokenizer, cui_to_label_train,  output_file, is_training):
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        if is_training:
            features = convert_single_example_train(ex_index, example, max_seq_length, max_n_masks, tokenizer, cui_to_label_train)
        else:
            features = convert_single_example_test(ex_index, example, max_seq_length, max_n_masks, tokenizer, cui_to_label_train)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return feature

        if is_training:
            for feature in features:
                result = collections.OrderedDict()
                result["guid"] = create_int_feature([feature.guid])
                result["input_ids"] = create_int_feature(feature.input_ids)
                result["input_mask"] = create_int_feature(feature.input_mask)
                result["segment_ids"] = create_int_feature(feature.segment_ids)
                result["positions"] = create_int_feature(feature.positions)
                result["masks_mask"] = create_float_feature(feature.masks_mask)
                result["label"] = create_int_feature(feature.label)
                result["input_ids_neg"] = create_int_feature(feature.input_ids_neg)
                result["input_mask_neg"] = create_int_feature(feature.input_mask_neg)
                result["segment_ids_neg"] = create_int_feature(feature.segment_ids_neg)
                result["positions_neg"] = create_int_feature(feature.positions_neg)
                result["masks_mask_neg"] = create_float_feature(feature.masks_mask_neg)
                result["label_neg"] = create_int_feature(feature.label_neg)
                tf_example = tf.train.Example(features=tf.train.Features(feature=result))
                writer.write(tf_example.SerializeToString())
        else:
            for feature in features:
                result = collections.OrderedDict()
                result["guid"] = create_int_feature([feature.guid])
                result["input_ids"] = create_int_feature(feature.input_ids)
                result["input_mask"] = create_int_feature(feature.input_mask)
                result["segment_ids"] = create_int_feature(feature.segment_ids)
                result["positions"] = create_int_feature(feature.positions)
                result["masks_mask"] = create_float_feature(feature.masks_mask)
                tf_example = tf.train.Example(features=tf.train.Features(feature=result))
                writer.write(tf_example.SerializeToString())

    writer.close()


def file_based_input_fn_builder(input_file, seq_length, max_n_masks, is_training, drop_remainder, num_examples):
    if is_training:
        name_to_features = {
            "guid": tf.FixedLenFeature([1], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "positions": tf.FixedLenFeature([max_n_masks], tf.int64),
            "masks_mask": tf.FixedLenFeature([max_n_masks], tf.float32),
            "label": tf.FixedLenFeature([max_n_masks], tf.int64),
            "input_ids_neg": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask_neg": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids_neg": tf.FixedLenFeature([seq_length], tf.int64),
            "positions_neg": tf.FixedLenFeature([max_n_masks], tf.int64),
            "masks_mask_neg": tf.FixedLenFeature([max_n_masks], tf.float32),
            "label_neg": tf.FixedLenFeature([max_n_masks], tf.int64),
        }
    else:
        name_to_features = {
            "guid": tf.FixedLenFeature([1], tf.int64),
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "positions": tf.FixedLenFeature([max_n_masks], tf.int64),
            "masks_mask": tf.FixedLenFeature([max_n_masks], tf.float32),
        }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.shuffle(buffer_size=num_examples, reshuffle_each_iteration=True)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

    return log_probs


def get_masked_lm_per_example_loss(bert_config, log_probs, label_ids, label_weights):
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    return label_weights * per_example_loss


def create_model_train(bert_config, input_ids, input_mask, segment_ids, masked_lm_positions, masks_mask,
                       label, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    seq_output_all = model.get_sequence_output()
    real_bs = tf.shape(seq_output_all)[0] // 2

    log_probs = get_masked_lm_output(bert_config, seq_output_all, model.get_embedding_table(),
                                     masked_lm_positions)

    with tf.variable_scope("loss_mask"):
        per_example_loss = get_masked_lm_per_example_loss(bert_config, log_probs, label, masks_mask)
        per_example_loss = tf.reshape(per_example_loss, [2, -1, FLAGS.max_n_masks])
        per_example_loss = (tf.reduce_sum(per_example_loss, axis=-1) /
                            tf.reduce_sum(tf.reshape(masks_mask, [2, -1, FLAGS.max_n_masks]), axis=-1))
        per_example_loss = (tf.nn.relu(per_example_loss[0] - per_example_loss[1] + FLAGS.loss_margin) +
                            FLAGS.loss_cross_entropy * per_example_loss[0])
        reduced_loss_mask = tf.reduce_mean(per_example_loss)

        log_probs = tf.reshape(log_probs, [real_bs * 2, FLAGS.max_n_masks, bert_config.vocab_size])
        log_probs *= tf.expand_dims(masks_mask, axis=-1)

    return reduced_loss_mask, per_example_loss, log_probs


def create_model_test(bert_config, input_ids, input_mask, segment_ids, masked_lm_positions, masks_mask,
                      use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    seq_output_all = model.get_sequence_output()
    real_bs = tf.shape(seq_output_all)[0]

    log_probs = get_masked_lm_output(bert_config, seq_output_all, model.get_embedding_table(),
                                     masked_lm_positions)

    with tf.variable_scope("loss_mask"):
        log_probs = tf.reshape(log_probs, [real_bs, FLAGS.max_n_masks, bert_config.vocab_size])
        log_probs *= tf.expand_dims(masks_mask, axis=-1)

    return log_probs


def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        guid = features["guid"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        positions = features["positions"]
        masks_mask = features["masks_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if is_training:
            label = features["label"]
            input_ids_neg = features["input_ids_neg"]
            input_mask_neg = features["input_mask_neg"]
            segment_ids_neg = features["segment_ids_neg"]
            positions_neg = features["positions_neg"]
            masks_mask_neg = features["masks_mask_neg"]
            label_neg = features["label_neg"]

            reduced_loss, per_example_loss, log_probs = create_model_train(
                bert_config,
                tf.concat([input_ids, input_ids_neg], axis=0),
                tf.concat([input_mask, input_mask_neg], axis=0),
                tf.concat([segment_ids, segment_ids_neg], axis=0),
                tf.concat([positions, positions_neg], axis=0),
                tf.concat([masks_mask, masks_mask_neg], axis=0),
                tf.concat([label, label_neg], axis=0),
                use_one_hot_embeddings)
            tf.summary.scalar('loss', reduced_loss)
        else:
            log_probs = create_model_test(
                bert_config,
                tf.concat([input_ids], axis=0),
                tf.concat([input_mask], axis=0),
                tf.concat([segment_ids], axis=0),
                tf.concat([positions], axis=0),
                tf.concat([masks_mask], axis=0),
                use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if is_training:
            summary_op = tf.summary.merge_all()
            summary_hook = tf.estimator.SummarySaverHook(save_steps=10,
                                                         output_dir=FLAGS.output_dir,
                                                         summary_op=summary_op)
            train_op = optimization.create_optimizer(reduced_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                     use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=reduced_loss,
                train_op=train_op,
                training_hooks=[summary_hook],
                scaffold_fn=scaffold_fn)
            return output_spec
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    'log_probs': log_probs,
                    'masks_mask': masks_mask,
                    'guid': guid},
                scaffold_fn=scaffold_fn)
            return output_spec

    return model_fn


def convert_single_result(example, arrays, tokenizer, cui_to_label_train, max_n_masks):
    # tokens = [
                 # tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.label)),
                 # tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.group))
             # ] + [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(neg)) for neg in example.negs]
    tokens = [
        tokenizer.tokenize(cui_to_label_train[cui]) if (cui != 'CUI-less') else ['[UNK]']
        for cui in example.negs]
    tokens = [tokenizer.convert_tokens_to_ids(t[:max_n_masks]) for t in tokens]

    result = {}

    def _take(arrays, i, toks):
        return arrays[i][0][np.arange(len(toks)), toks]
        # return arrays[i][0][np.arange(len(toks)), toks], arrays[i][1]

    result['guid'] = example.guid
    result['group'] = example.group
    for i in range(len(example.negs)):
        result[f'log_probs_train_{i}'] = _take(arrays, i, tokens[i])

    return result

def load_cui_label_dict():
    lines = []
    tsv_fname = 'cui_label.tsv'
    with open(os.path.join(FLAGS.data_dir, tsv_fname), 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for line in reader:
            lines.append(line)
    cui_label_dict = {}
    for i in range(1, len(lines)):
        cui, label = lines[i]
        cui_label_dict[cui] = label
    return cui_label_dict


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # Juyong: can allow memory growth on GPU
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=8,
            per_host_input_for_training=is_per_host),
        session_config=session_config
        )
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                           do_lower_case=FLAGS.do_lower_case)
    if os.path.isdir(FLAGS.init_checkpoint):
        FLAGS.init_checkpoint = tf.train.latest_checkpoint(FLAGS.init_checkpoint)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    num_train_steps = None
    num_warmup_steps = None
    processor = UMNProcessor()
    if FLAGS.do_train:
        examples = processor.get_train_examples(FLAGS.data_dir)
    else:
        if FLAGS.dataset == "train":
            examples = processor.get_train_examples2(FLAGS.data_dir)
        else:
            examples = processor.get_test_examples(FLAGS.data_dir)

    cui_to_label_train = load_cui_label_dict()
    if FLAGS.do_train:
        num_examples = sum([max(len(e.negs), 1) for e in examples])
        num_train_steps = int(num_examples / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        print(f'{num_examples} train examples, {num_train_steps} train steps, {num_warmup_steps} warmup steps')
        file_based_convert_examples_to_features(
            examples, FLAGS.max_seq_length, FLAGS.max_n_masks, tokenizer, cui_to_label_train,
            os.path.join(FLAGS.output_dir, 'train.tf_record'), True)
        input_fn = file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.output_dir, 'train.tf_record'),
            seq_length=FLAGS.max_seq_length,
            max_n_masks=FLAGS.max_n_masks,
            is_training=True,
            drop_remainder=False,
            num_examples=num_examples
        )
    else:
        num_examples = sum([len(e.negs) for e in examples])
        file_based_convert_examples_to_features(
            examples, FLAGS.max_seq_length, FLAGS.max_n_masks, tokenizer, cui_to_label_train,
            os.path.join(FLAGS.output_dir, 'dev.tf_record'), False)
        input_fn = file_based_input_fn_builder(
            input_file=os.path.join(FLAGS.output_dir, 'dev.tf_record'),
            seq_length=FLAGS.max_seq_length,
            max_n_masks=FLAGS.max_n_masks,
            is_training=False,
            drop_remainder=False,
            num_examples=num_examples
        )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.batch_size)

    if FLAGS.do_train:
        estimator.train(input_fn=input_fn, max_steps=num_train_steps)
    else:
        result = estimator.predict(input_fn=input_fn)
        tf.logging.info("***** Predict results *****")
        examples_dict = {e.guid: e for e in examples}
        outputs = []
        lst = []
        last_guid = examples[0].guid
        for i, prediction in enumerate(result):
            if prediction['guid'][0] != last_guid:
                outputs.append(convert_single_result(examples_dict[last_guid], lst, tokenizer, cui_to_label_train, FLAGS.max_n_masks))
                lst = []
                last_guid = prediction['guid'][0]
            lst.append([prediction['log_probs']])
        outputs.append(convert_single_result(examples_dict[last_guid], lst, tokenizer, cui_to_label_train, FLAGS.max_n_masks))
        tf.logging.info("finished all examples")
        with open(os.path.join(FLAGS.output_dir, "output.pkl"), "wb") as f:
            pickle.dump(outputs, f)
        tf.logging.info("wrote all examples to file")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("batch_size")
    flags.mark_flag_as_required("init_checkpoint")
    tf.app.run()
