"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics, acc_and_f1
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score, average_precision_score
from copy import deepcopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="mnli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "mnli":
            return self.get_train_examples(data_dir, augmented=augmented,
                                           augmented_n=augmented_n,
                                           augmented_bias_n=augmented_bias_n)
        elif source == "hans":
            hans_processor = HansProcessor()
            return hans_processor.get_train_examples(data_dir, filter_label=None, filter_subset=None)

    def get_train_examples(self, data_dir, augmented=False, augmented_n=5, augmented_bias_n=0):
        """See base class."""
        if not augmented:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        else:
            examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "augment")

            current_seed = int(data_dir.split("/")[-1].split("-")[-1])
            augment_path = os.path.join(data_dir,
                                        "../../../veridicality",
                                        f"k-shot_ct-{augmented_n}_sup-{augmented_bias_n}_seed-{current_seed}.tsv")
            augment_examples = self._create_examples(self._read_tsv(augment_path), "train")

            label_to_ids = {}
            for label in self.get_labels():
                label_to_ids[label] = [ix for (ix, x) in enumerate(examples) if x.label == label]
                replacement_indices = [ix for (ix, x) in enumerate(augment_examples) if x.label == label]

                assert len(replacement_indices) <= len(label_to_ids[label])

                replaced_indices = random.choices(label_to_ids[label], k=len(replacement_indices))
                for replacement_ix, replaced_ix in zip(replacement_indices, replaced_indices):
                    examples[replaced_ix] = augment_examples[replacement_ix]

            return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")

class MnliSynProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "partial_0.9_test_matched.tsv")), "test_anti_biased")

class MnliSynBiasedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "partial_0.9_test_matched_all_biased.tsv")), "test_all_biased")


class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="snli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "snli":
            return self.get_train_examples(data_dir, augmented=augmented,
                                           augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif source == "counterfactuals":
            counterfactuals_processor = CounterfactualSnliProcessor()
            return counterfactuals_processor.get_train_examples(data_dir)

    def get_train_examples(self, data_dir, augmented=False, augmented_n=5, augmented_bias_n=0):
        """See base class."""
        if not augmented:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        else:
            examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

            counterfactuals_processor = CounterfactualSnliProcessor()
            augment_examples = counterfactuals_processor.get_train_examples(data_dir)

            label_to_ids = {}
            augment_label_to_ids = {}
            for label in self.get_labels():
                label_to_ids[label] = [ix for (ix, x) in enumerate(examples) if x.label == label]
                augment_label_to_ids[label] = [ix for (ix, x) in enumerate(augment_examples) if x.label == label]

                if len(augment_label_to_ids[label]) >= augmented_n:
                    replacement_indices = random.choices(augment_label_to_ids[label], k=augmented_n)
                    replaced_indices = random.choices(label_to_ids[label], k=augmented_n)
                    for replacement_ix, replaced_ix in zip(replacement_indices, replaced_indices):
                        examples[replaced_ix] = augment_examples[replacement_ix]

            return examples

    def get_train_examples(self, data_dir, augmented=False, augmented_n=5, augmented_bias_n=0):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        text_index = 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="qqp", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "qqp":
            return self.get_train_examples(data_dir, augmented=augmented,
                                           augmented_n=augmented_n,
                                           augmented_bias_n=augmented_bias_n)
        else:
            raise NotImplementedError

    def get_train_examples(self, data_dir, augmented=False, augmented_n=5, augmented_bias_n=0):
        """See base class."""
        if not augmented:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        else:
            examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

            paws_processor = PawsProcessor()
            current_seed = int(data_dir.split("/")[-1].split("-")[-1])
            augment_path = os.path.join(data_dir,
                                        "../../../paws",
                                        f"k-shot_ct-{augmented_n}_sup-{augmented_bias_n}_seed-{current_seed}.tsv")
            augment_examples = paws_processor._create_examples(self._read_tsv(augment_path), "augment")

            label_to_ids = {}
            for label in self.get_labels():
                label_to_ids[label] = [ix for (ix, x) in enumerate(examples) if x.label == label]
                replacement_indices = [ix for (ix, x) in enumerate(augment_examples) if x.label == label]

                assert len(replacement_indices) <= len(label_to_ids[label])

                replaced_indices = random.choices(label_to_ids[label], k=len(replacement_indices))
                for replacement_ix, replaced_ix in zip(replacement_indices, replaced_indices):
                    examples[replaced_ix] = augment_examples[replacement_ix]

            return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 3
        q2_index = 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="rte", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "rte":
            return self.get_train_examples(data_dir, augmented=augmented,
                                           augmented_n=augmented_n,
                                           augmented_bias_n=augmented_bias_n)
        elif source == "hans":
            hans_processor = HansProcessor()
            return hans_processor.get_train_examples(data_dir, filter_label=None, filter_subset=None)

    def get_train_examples(self, data_dir, augmented=False, augmented_n=5, augmented_bias_n=0):
        """See base class."""
        if not augmented:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        else:
            raise NotImplementedError

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

######## HANS processor ########
class HansProcessor(DataProcessor):
    """Processor for the HANS data set."""

    SUBSETS = ["lexical_overlap", "subsequence", "constituent"]
    LABELS = ["entailment", "non-entailment"]

    def get_support_examples(self, data_dir, source="mnli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if "MNLI" in data_dir:  # temporary
            # retrieve instances from the mnli train instead
            mnli_processor = MnliProcessor()
            return mnli_processor.get_train_examples(data_dir, augmented=augmented,
                                                     augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif "RTE" in data_dir:
            rte_processor = RteProcessor()
            return rte_processor.get_train_examples(data_dir, augmented=augmented,
                                                    augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif "SNLI" in data_dir:
            snli_processor = SnliProcessor()
            return snli_processor.get_train_examples(data_dir, augmented=augmented,
                                                     augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif source == "hans":
            return self.get_train_examples(data_dir, filter_label=None, filter_subset=None)

    def get_train_examples(self, data_dir, filter_subset="lexical_overlap",
                           filter_label="non-entailment"):
        """See base class."""
        # used only to get the support / demo examples
        hans_data_dir = os.path.join(data_dir, "../../../hans")
        return self._create_examples(self._read_tsv(os.path.join(hans_data_dir, "sampled_demonstration.txt")),
                                     "test", filter_subset=None, filter_label=None)

    def get_test_examples(self, data_dir, filter_subset="lexical_overlap",
                          filter_label="non-entailment"):
        """See base class."""
        hans_data_dir = os.path.join(data_dir, "../../../hans")
        return self._create_examples(self._read_tsv(os.path.join(hans_data_dir, "heuristics_evaluation_set.txt")),
                                     "test", filter_subset=filter_subset, filter_label=filter_label)

    def get_labels(self):
        # hack to handle mismatch with MNLI
        return ["contradiction", "entailment", "neutral"]

    def get_subsets(self):
        return ["lexical_overlap", "subsequence", "constituent"]

    def _create_examples(self, lines, set_type, filter_subset="lexical_overlap",
                         filter_label="non-entailment"):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s-%s" % (set_type, filter_label, filter_subset)
            text_a = line[5]
            text_b = line[6]
            pairID = line[7][2:] if line[7].startswith("ex") else line[7]
            label = line[0]
            subset = line[8]

            mnli_label = "contradiction" if label == "non-entailment" else label
            if filter_label is None and filter_subset is None:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=mnli_label))
            elif label == filter_label and subset == filter_subset:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=mnli_label))
        return examples

    # create function to return all subset names
    # return dataset containing only this particular subset


class CounterfactualSnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="snli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "snli":
            # retrieve instances from the snli train instead
            snli_processor = SnliProcessor()
            return snli_processor.get_train_examples(data_dir, augmented=augmented,
                                                     augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif source == "counterfactuals":
            return self.get_train_examples(data_dir)

    def get_train_examples(self, data_dir):
        """See base class."""
        cfct_data_dir = os.path.join(data_dir, "../../../counterfactuals/NLI/revised_combined")
        return self._create_examples(self._read_tsv(os.path.join(cfct_data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        cfct_data_dir = os.path.join(data_dir, "../../../counterfactuals/NLI/revised_combined")
        return self._create_examples(self._read_tsv(os.path.join(cfct_data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        cfct_data_dir = os.path.join(data_dir, "../../../counterfactuals/NLI/revised_combined")
        return self._create_examples(self._read_tsv(os.path.join(cfct_data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PawsProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="qqp", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "paws":
            return self.get_train_examples(data_dir)
        elif source == "qqp":
            # retrieve instances from the snli train instead
            qqp_processor = QqpProcessor()
            return qqp_processor.get_train_examples(data_dir, augmented=augmented, augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)

    def get_train_examples(self, data_dir):
        """See base class."""
        paws_data_dir = os.path.join(data_dir, "../../../paws")
        # use smaller dev/test data as augmentation for training instead
        return self._create_examples(self._read_tsv(os.path.join(paws_data_dir, "qqp_dev_and_test.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        paws_data_dir = os.path.join(data_dir, "../../../paws")
        return self._create_examples(self._read_tsv(os.path.join(paws_data_dir, "qqp_train.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        paws_data_dir = os.path.join(data_dir, "../../../paws")
        return self._create_examples(self._read_tsv(os.path.join(paws_data_dir, "qqp_train.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        q1_index = 1
        q2_index = 2
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[-1]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AnliProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self, round_name) -> None:
        super().__init__()
        self.round_name = round_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="mnli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "anli":
            return self.get_train_examples(data_dir)
        elif source == "mnli":
            # retrieve instances from the mnli train instead
            mnli_processor = MnliProcessor()
            return mnli_processor.get_train_examples(data_dir, augmented=augmented, augmented_n=augmented_n,
                                                     augmented_bias_n=augmented_bias_n)

    def get_train_examples(self, data_dir):
        """See base class."""
        anli_data_dir = os.path.join(data_dir, "../../../anli", self.round_name)
        return self._create_examples(self._read_jsonl(os.path.join(anli_data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        anli_data_dir = os.path.join(data_dir, "../../../anli", self.round_name)
        return self._create_examples(self._read_jsonl(os.path.join(anli_data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        anli_data_dir = os.path.join(data_dir, "../../../anli", self.round_name)
        return self._create_examples(self._read_jsonl(os.path.join(anli_data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        # align the labels with ones used in mnli, snli, etc.
        return ["contradiction", "entailment", "neutral"]

    def get_label_mappings(self):
        return {x[0]: x for x in self.get_labels()}

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        label_mappings = self.get_label_mappings()
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, self.round_name, line[0])
            text_a = line[1]
            text_b = line[2]
            label = label_mappings[line[-1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_jsonl(self, input_path: str):
        with open(input_path, "r") as reader:
            lines = reader.readlines()
            all_ex_lines = []
            for idx, line in enumerate(lines):
                ex_line = []
                line_dict = json.loads(line)
                ex_line.append(str(idx))
                ex_line.append(line_dict["context"])
                ex_line.append(line_dict["hypothesis"])
                ex_line.append(line_dict["label"])
                all_ex_lines.append(ex_line)
            return all_ex_lines

class ScrambleTestProcessor(DataProcessor):
    """Processor for the Scrable Test (Dasgupta et al. 2018) data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_support_examples(self, data_dir, source="snli", augmented=False, augmented_n=5, augmented_bias_n=0):
        if source == "snli":
            # retrieve instances from the snli train instead
            snli_processor = SnliProcessor()
            return snli_processor.get_train_examples(data_dir, augmented=augmented,
                                                     augmented_n=augmented_n, augmented_bias_n=augmented_bias_n)
        elif source == "scramble":
            return self.get_train_examples(data_dir)

    def get_train_examples(self, data_dir):
        """See base class."""
        # train from test
        scramble_data_dir = os.path.join(data_dir, "../../../scramble-test/test")
        return self._create_examples(self.merge_files(scramble_data_dir), "train")

    def merge_files(self, data_path):
        s1_lines = []
        s2_lines = []
        label_lines = []
        with open(os.path.join(data_path, "s1.comp_same_short"), "r") as reader:
            s1_lines.extend([x.strip() for x in reader.readlines()])
        with open(os.path.join(data_path, "s2.comp_same_short"), "r") as reader:
            s2_lines.extend([x.strip() for x in reader.readlines()])
        with open(os.path.join(data_path, "labels.comp_same_short"), "r") as reader:
            label_lines.extend([x.strip() for x in reader.readlines()])
        assert len(s1_lines) == len(s2_lines) and len(s1_lines) == len(label_lines)
        return [x for x in zip(s1_lines, s2_lines, label_lines)]

    def get_dev_examples(self, data_dir):
        """See base class."""
        scramble_data_dir = os.path.join(data_dir, "../../../scramble-test/valid")
        return self._create_examples(self.merge_files(scramble_data_dir), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # test on train
        scramble_data_dir = os.path.join(data_dir, "../../../scramble-test/train")
        return self._create_examples(self.merge_files(scramble_data_dir), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        # follow the mapping defined in https://github.com/ishita-dg/ScrambleTests/tree/training-experiment/testData
        labels_mapping = {
            "0": "contradiction",
            "1": "neutral",
            "2": "entailment"
        }
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text_a = line[0]
            text_b = line[1]
            label = labels_mapping[line[-1]]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class ScrambleTestLongProcessor(ScrambleTestProcessor):
    def merge_files(self, data_path):
        s1_lines = []
        s2_lines = []
        label_lines = []
        with open(os.path.join(data_path, "s1.comp_same_long"), "r") as reader:
            s1_lines.extend([x.strip() for x in reader.readlines()])
        with open(os.path.join(data_path, "s2.comp_same_long"), "r") as reader:
            s2_lines.extend([x.strip() for x in reader.readlines()])
        with open(os.path.join(data_path, "labels.comp_same_long"), "r") as reader:
            label_lines.extend([x.strip() for x in reader.readlines()])
        assert len(s1_lines) == len(s2_lines) and len(s1_lines) == len(label_lines)
        return [x for x in zip(s1_lines, s2_lines, label_lines)]









class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def __init__(self, task_name):
        self.task_name = task_name 

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(InputExample(guid=guid, text_a=line[1] + '. ' + line[2], short_text=line[1] + ".", label=line[0]))
            elif self.task_name == "yelp_review_full":
                examples.append(InputExample(guid=guid, text_a=line[1], short_text=line[1], label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += ' ' + line[2]
                if not pd.isna(line[3]):
                    text += ' ' + line[3]
                examples.append(InputExample(guid=guid, text_a=text, short_text=line[1], label=line[0])) 
            elif self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples
        
def text_classification_metrics(task_name, preds, labels):
    if task_name == "qqp" or task_name == "paws":
        metrics = acc_and_f1(preds, labels)
        metrics["neg_f1"] = f1_score(y_true=labels, y_pred=preds, pos_label=0)
        return metrics
    else:
        return {"acc": (preds == labels).mean()}

# Add your task to the following mappings

processors_mapping = {
    "cola": ColaProcessor(),
    "mnli": MnliProcessor(),
    "mnli-mm": MnliMismatchedProcessor(),
    "mnli-syn": MnliSynProcessor(),
    "mnli-syn-all-biased": MnliSynBiasedProcessor(),
    "hans": HansProcessor(),
    "anli_r1": AnliProcessor("R1"),
    "anli_r2": AnliProcessor("R2"),
    "anli_r3": AnliProcessor("R3"),
    "mrpc": MrpcProcessor(),
    "sst-2": Sst2Processor(),
    "sts-b": StsbProcessor(),
    "qqp": QqpProcessor(),
    "paws": PawsProcessor(),
    "qnli": QnliProcessor(),
    "rte": RteProcessor(),
    "wnli": WnliProcessor(),
    "snli": SnliProcessor(),
    "counterfactuals_snli": CounterfactualSnliProcessor(),
    "scramble-test": ScrambleTestProcessor(),
    "scramble-test-long": ScrambleTestLongProcessor(),
    "mr": TextClassificationProcessor("mr"),
    "sst-5": TextClassificationProcessor("sst-5"),
    "subj": TextClassificationProcessor("subj"),
    "trec": TextClassificationProcessor("trec"),
    "cr": TextClassificationProcessor("cr"),
    "mpqa": TextClassificationProcessor("mpqa")
}

num_labels_mapping = {
    "cola": 2,
    "mnli": 3,
    "mnli-syn": 3,
    "mnli-syn-all-biased": 3,
    "anli_r1": 3,
    "anli_r2": 3,
    "anli_r3": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "paws": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "counterfactuals_snli": 3,
    "scramble-test": 3,
    "scramble-test-long": 3,
    "mr": 2,
    "sst-5": 5,
    "subj": 2,
    "trec": 6,
    "cr": 2,
    "mpqa": 2
}

output_modes_mapping = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mnli-syn": "classification",
    "mnli-syn-all-biased": "classification",
    "anli_r1": "classification",
    "anli_r2": "classification",
    "anli_r3": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "paws": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "snli": "classification",
    "counterfactuals_snli": "classification",
    "scramble-test": "classification",
    "scramble-test-long": "classification",
    "mr": "classification",
    "sst-5": "classification",
    "subj": "classification",
    "trec": "classification",
    "cr": "classification",
    "mpqa": "classification",
    "hans": "classification"
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "cola": glue_compute_metrics,
    "mnli": glue_compute_metrics,
    "mnli-mm": glue_compute_metrics,
    "mnli-syn": text_classification_metrics,
    "mnli-syn-all-biased": text_classification_metrics,
    "anli_r1": text_classification_metrics,
    "anli_r2": text_classification_metrics,
    "anli_r3": text_classification_metrics,
    "mrpc": glue_compute_metrics,
    "sst-2": glue_compute_metrics,
    "sts-b": glue_compute_metrics,
    "qqp": text_classification_metrics,
    "paws": text_classification_metrics,
    "qnli": glue_compute_metrics,
    "rte": glue_compute_metrics,
    "wnli": glue_compute_metrics,
    "hans": glue_compute_metrics,
    "snli": text_classification_metrics,
    "counterfactuals_snli": text_classification_metrics,
    "scramble-test": text_classification_metrics,
    "scramble-test-long": text_classification_metrics,
    "mr": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
    "cr": text_classification_metrics,
    "mpqa": text_classification_metrics,
}

# For regression task only: median
median_mapping = {
    "sts-b": 2.5
}

bound_mapping = {
    "sts-b": (0, 5)
}
