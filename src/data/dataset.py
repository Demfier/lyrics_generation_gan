#!/usr/bin/env python

import csv
import logging
import pickle
import numpy as np
from overrides import overrides
from typing import Dict
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import ArrayField, TextField, LabelField, MetadataField
# from allennlp.data.tokenizers import WordTokenizer old
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("autoencoder")
class AutoencoderDatasetReader(Seq2SeqDatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(input_string)
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)
        return Instance({"source_tokens": source_field, "target_tokens": target_field})


@DatasetReader.register("lyrics-classification")
class LyricsClassifierDatasetReader(Seq2SeqDatasetReader):
    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))

                query, response, label = row
                yield self.text_to_instance(query, response, int(label))

    @overrides
    def text_to_instance(self, query_string: str, response_string: str, label: int) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._source_tokenizer.tokenize(query_string)
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        tokenized_target = self._target_tokenizer.tokenize(response_string)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)

        label_field = LabelField(label, skip_indexing=True)
        return Instance({"query_tokens": source_field, "response_tokens": target_field, "label": label_field})


@DatasetReader.register("sentence")
class SentenceReader(DatasetReader):
    """
    ``AutoencoderDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 delimiter: str = "\t",
                 max_seq_len: int = 30,
                 key: str = 'source_tokens',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_start_token = add_start_token
        self._delimiter = delimiter
        self._max_seq_len = max_seq_len
        self._key = key

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _, line in enumerate(data_file):
                sentence = line.strip("\n")
                if sentence:
                    yield self.text_to_instance(sentence)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(sentence)
        tokenized_string = tokenized_string[:self._max_seq_len - 2]
        if self._add_start_token:
            tokenized_string.insert(0, Token(START_SYMBOL))
        tokenized_string.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_string, self._token_indexers)

        return Instance({self._key: sentence_field})


@DatasetReader.register("lyrics-gan")
class LyricsGanDatasetReader(DatasetReader):
    """
    We already have pre-trained VAE vectors
    """
    def __init__(self,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.states = ["discriminator_real", "discriminator_fake", "generator"]
        self.state = self._state_generator()

    def _state_generator(self):
        while True:
            for state in self.states:
                yield state

    def _get_state(self):
        return next(self.state)

    @overrides
    def _read(self, file_path):
        """
        file_path: contains spec-VAE's mu, sigma and text-VAE's mu as a dict
        """
        with open(cached_path(file_path), "rb") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for _id, _dict in enumerate(pickle.load(data_file)):
                yield self.dict_to_instance(_dict)

    def dict_to_instance(self, array_dict: [str, np.array]) -> Instance:
        stage_field = MetadataField(self._get_state())
        # print(array_dict['spec_mu'].reshape(-1))
        return Instance({
            'source_mu': ArrayField(array_dict['spec_mu'].reshape(-1)),
            'source_std': ArrayField(array_dict['spec_std'].reshape(-1)),
            'target_mu': ArrayField(array_dict['lyrics_mu'].reshape(-1)),
            'stage': stage_field,
            })
