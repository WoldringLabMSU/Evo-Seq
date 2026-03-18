"""
Tests for Fine-Tuning/functions.py

Coverage targets:
- pad_sequence: padding, truncation, exact-length input
- get_fasta_dict: normal input, unknown chars, multi-line sequences, empty file,
  sequence-before-header error
- token2idx: known token, unknown token fallback
- convert: normal sequence, truncation, type errors, padding
- SeqDataset: __len__, __getitem__, token tensor shape
- is_1hot_tensor: valid one-hot, non-one-hot, wrong dim
- apply_mask: masking replaces correct positions
"""

import os
import sys
import tempfile
import textwrap

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup – allow importing from Fine-Tuning without installing as a package
# ---------------------------------------------------------------------------
FINE_TUNING_DIR = os.path.join(os.path.dirname(__file__), "..", "Fine-Tuning")
sys.path.insert(0, os.path.abspath(FINE_TUNING_DIR))

# The module imports `esm` at module level; skip the whole file if it is not
# available in the test environment.
esm = pytest.importorskip("esm", reason="fair-esm not installed")

import functions  # noqa: E402  (must come after sys.path manipulation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ESM_ALPHABET = functions.esm_alphabet
TOKEN2IDX = functions.token2idx_dict

AMINO_ACIDS_STANDARD = "LACDEFGHIKLMNPQRSTVWY"


def _write_fasta(content: str) -> str:
    """Write *content* to a temporary FASTA file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# pad_sequence
# ---------------------------------------------------------------------------


class TestPadSequence:
    def test_pads_short_sequence(self):
        result = functions.pad_sequence("ACG", 6, "<pad>")
        assert result == "ACG" + "<pad>" * 3

    def test_truncates_long_sequence(self):
        result = functions.pad_sequence("ACDEFGHIK", 5, "<pad>")
        assert result == "ACDEF"

    def test_exact_length_unchanged(self):
        seq = "ACDE"
        result = functions.pad_sequence(seq, 4, "<pad>")
        assert result == seq

    def test_empty_sequence_is_all_padding(self):
        result = functions.pad_sequence("", 3, "X")
        assert result == "XXX"

    def test_zero_target_length_returns_empty(self):
        result = functions.pad_sequence("ACG", 0, "<pad>")
        assert result == ""


# ---------------------------------------------------------------------------
# token2idx
# ---------------------------------------------------------------------------


class TestToken2Idx:
    def test_known_token_returns_correct_index(self):
        for token in ["L", "A", "<cls>", "<pad>", "<mask>"]:
            expected = TOKEN2IDX[token]
            assert functions.token2idx(token) == expected

    def test_unknown_token_falls_back_to_unk(self):
        unk_idx = TOKEN2IDX["<unk>"]
        assert functions.token2idx("$$$NOT_A_TOKEN$$$") == unk_idx


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


class TestConvert:
    def test_output_length_equals_max_length_plus_two(self):
        seq = "ACDE"
        max_length = 10
        result = functions.convert(seq, max_length)
        assert len(result) == max_length + 2

    def test_first_token_is_cls(self):
        result = functions.convert("A", 10)
        assert result[0] == TOKEN2IDX["<cls>"]

    def test_last_real_token_is_eos(self):
        seq = "AC"
        max_length = 10
        result = functions.convert(seq, max_length)
        # <cls> + A + C + <eos> = indices 0,1,2,3; rest is <pad>
        assert result[3] == TOKEN2IDX["<eos>"]

    def test_padding_fills_remainder(self):
        seq = "A"
        max_length = 5
        result = functions.convert(seq, max_length)
        # indices: [cls, A, eos, pad, pad, pad, pad]  → length 7
        for i in range(3, len(result)):
            assert result[i] == TOKEN2IDX["<pad>"]

    def test_long_sequence_is_truncated(self):
        seq = "A" * 20
        max_length = 5
        result = functions.convert(seq, max_length)
        assert len(result) == max_length + 2

    def test_raises_type_error_for_invalid_seq(self):
        with pytest.raises(TypeError):
            functions.convert(12345, 10)

    def test_raises_type_error_for_invalid_max_length(self):
        with pytest.raises(TypeError):
            functions.convert("ACG", "ten")

    def test_returns_numpy_array(self):
        result = functions.convert("AC", 10)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# get_fasta_dict
# ---------------------------------------------------------------------------


class TestGetFastaDict:
    def test_reads_single_sequence(self):
        fasta = """\
            >seq1
            ACDE
        """
        path = _write_fasta(fasta)
        try:
            result = functions.get_fasta_dict(path, 10, ESM_ALPHABET)
            assert "seq1" in result
            assert result["seq1"].startswith("ACDE")
        finally:
            os.unlink(path)

    def test_pads_short_sequence_to_target_length(self):
        fasta = """\
            >seq1
            AC
        """
        path = _write_fasta(fasta)
        try:
            result = functions.get_fasta_dict(path, 10, ESM_ALPHABET)
            assert len(result["seq1"]) == 10
        finally:
            os.unlink(path)

    def test_reads_multiple_sequences(self):
        fasta = """\
            >seq1
            ACDE
            >seq2
            LMNO
        """
        path = _write_fasta(fasta)
        try:
            result = functions.get_fasta_dict(path, 10, ESM_ALPHABET)
            assert set(result.keys()) == {"seq1", "seq2"}
        finally:
            os.unlink(path)

    def test_replaces_unknown_chars_with_unk_token(self):
        fasta = """\
            >seq1
            @@@
        """
        path = _write_fasta(fasta)
        try:
            result = functions.get_fasta_dict(path, 10, ESM_ALPHABET)
            assert "<unk>" in result["seq1"]
        finally:
            os.unlink(path)

    def test_raises_on_sequence_before_header(self):
        fasta = "ACDE\n>seq1\nACDE\n"
        path = _write_fasta(fasta)
        try:
            with pytest.raises(ValueError, match="sequence data encountered before header"):
                functions.get_fasta_dict(path, 10, ESM_ALPHABET)
        finally:
            os.unlink(path)

    def test_multiline_sequence_is_concatenated(self):
        fasta = """\
            >seq1
            AC
            DE
        """
        path = _write_fasta(fasta)
        try:
            result = functions.get_fasta_dict(path, 10, ESM_ALPHABET)
            # Both lines should be present (after stripping)
            assert result["seq1"].startswith("ACDE")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# SeqDataset
# ---------------------------------------------------------------------------


class TestSeqDataset:
    def _make_dataset(self, seqs: dict, max_length: int = 20):
        return functions.SeqDataset(seqs, max_length)

    def test_len_matches_number_of_sequences(self):
        ds = self._make_dataset({"a": "ACDE", "b": "LMNO"})
        assert len(ds) == 2

    def test_getitem_returns_long_tensor(self):
        ds = self._make_dataset({"a": "ACDE"}, max_length=10)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.long

    def test_getitem_tensor_length_is_max_length_plus_two(self):
        max_length = 15
        ds = self._make_dataset({"a": "ACDE"}, max_length=max_length)
        item = ds[0]
        assert item.shape[0] == max_length + 2

    def test_getitem_out_of_range_raises(self):
        ds = self._make_dataset({"a": "ACDE"})
        with pytest.raises((IndexError, KeyError)):
            _ = ds[100]


# ---------------------------------------------------------------------------
# is_1hot_tensor
# ---------------------------------------------------------------------------


class TestIs1HotTensor:
    def test_valid_one_hot_returns_true(self):
        t = torch.eye(5).unsqueeze(0)  # shape (1, 5, 5)
        assert functions.is_1hot_tensor(t)

    def test_all_zeros_returns_false(self):
        t = torch.zeros(2, 5)
        assert not functions.is_1hot_tensor(t)

    def test_row_summing_to_two_returns_false(self):
        t = torch.zeros(1, 5)
        t[0, 0] = 1
        t[0, 1] = 1
        assert not functions.is_1hot_tensor(t)

    def test_1d_tensor_returns_false(self):
        t = torch.tensor([0.0, 1.0, 0.0])
        assert not functions.is_1hot_tensor(t)

    def test_batch_of_one_hot_vectors_returns_true(self):
        t = torch.zeros(4, 10)
        t[0, 2] = 1
        t[1, 5] = 1
        t[2, 0] = 1
        t[3, 9] = 1
        assert functions.is_1hot_tensor(t)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


class TestApplyMask:
    def _make_one_hot(self, seq_len: int, vocab_size: int) -> torch.Tensor:
        """Return a (seq_len, vocab_size) one-hot float tensor."""
        indices = torch.randint(0, vocab_size, (seq_len,))
        return torch.nn.functional.one_hot(indices, vocab_size).float()

    def test_masked_positions_contain_mask_token(self):
        vocab = 10
        seq_len = 8
        mask_idx = 7
        x = self._make_one_hot(seq_len, vocab)
        mask = torch.zeros(seq_len, 1, dtype=torch.bool)
        mask[2] = True
        mask[5] = True
        result = functions.apply_mask(x, mask, mask_idx)
        expected_row = torch.nn.functional.one_hot(torch.tensor(mask_idx), vocab).float()
        assert torch.equal(result[2], expected_row)
        assert torch.equal(result[5], expected_row)

    def test_unmasked_positions_are_unchanged(self):
        vocab = 10
        seq_len = 6
        mask_idx = 0
        x = self._make_one_hot(seq_len, vocab)
        mask = torch.zeros(seq_len, 1, dtype=torch.bool)
        mask[1] = True
        result = functions.apply_mask(x, mask, mask_idx)
        assert torch.equal(result[0], x[0])
        assert torch.equal(result[3], x[3])

    def test_original_tensor_is_not_mutated(self):
        vocab = 5
        seq_len = 4
        mask_idx = 3
        x = self._make_one_hot(seq_len, vocab)
        x_copy = x.clone()
        mask = torch.ones(seq_len, 1, dtype=torch.bool)
        functions.apply_mask(x, mask, mask_idx)
        assert torch.equal(x, x_copy)

    def test_raises_if_input_is_not_one_hot(self):
        """apply_mask uses assert is_1hot_tensor(x), so bad input raises AssertionError."""
        vocab = 5
        x = torch.zeros(4, vocab)  # all-zero rows – not one-hot
        mask = torch.zeros(4, 1, dtype=torch.bool)
        with pytest.raises(AssertionError):
            functions.apply_mask(x, mask, 0)
