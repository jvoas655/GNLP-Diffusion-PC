from typing import List
import collections

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'

# Take from https://arxiv.org/pdf/1509.01626.pdf
NUMERIC_CHARACTERS = "0123456789"
ALPHA_CHARACTERS = "abcdefghijklmnopqrstuvwxyz"
SPECIAL_CHARACTERS = "-,;.!?:’’’/\\|_@#$%ˆ&*˜‘+-=<>()[]{}"


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.
    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """

    @classmethod
    def create_word_vocab(cls, samples: List[List[str]], vocab_size: int):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list.
        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.
        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """

        vocab = collections.defaultdict(int)

        # Build the vocabulary with frequency counts
        for sentence in samples:
            for word in sentence:
                vocab[word.lower()] += 1

        # Take the top words that appeared the most
        words = [
                    word for (word, _) in
                    sorted(vocab.items(), key=lambda x: x[1], reverse=True)
                ][:vocab_size]

        # Add pad and unk tokens
        return Vocabulary(words)

    @classmethod
    def create_character_vocab(cls, characters: str):
        """
        Creates a vocabulary with all the characters given.
        Args:
            characters: A string of characters to add to the vocab
        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """

        words = list(set([x.lower() for x in characters]))

        return Vocabulary(words)

    def __init__(self, words: List[str]):

        words = [PAD_TOKEN, SEP_TOKEN, UNK_TOKEN] + words
        self.words = words

        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.
    Args:
        vocabulary: An instantiated `Vocabulary` object.
    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]
        self.sep_token_id = self.vocabulary.encoding[SEP_TOKEN]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[str]:
        """
        Converts words to corresponding indices.
        Args:
            tokens: A list of words (string).
        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids: List[int]):
        """
        Converts indices to corresponding words.
        Args:
            token_ids: A list of indices (int).
        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]