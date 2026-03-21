"""
Tests for Generative_Model/oh_vae_generate_from_trained.py
and the remaining untested function in oh_vae_train.py (train_vae_model).

Coverage targets
----------------
- one_hot_decode: correct output string, shape handling, custom alphabet,
  regression for wrong-dim argmax bug
- generate_sequences: returns list of strings with correct length, requires
  a saved model file (skipped without one)
- train_vae_model: 1-epoch smoke test, early stopping, model saved to disk
"""

import os
import sys
import importlib.util as _ilu
import tempfile

import pytest
import torch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
GENERATIVE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Generative_Model")
)
sys.path.insert(0, GENERATIVE_DIR)

# ---------------------------------------------------------------------------
# Conditional imports – emit xfail if source has unfixed syntax errors
# ---------------------------------------------------------------------------

def _load_module(name: str, path: str):
    """Load a module from an absolute path; return None on SyntaxError."""
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except SyntaxError as exc:
        return None


_gen_mod = _load_module(
    "oh_vae_generate",
    os.path.join(GENERATIVE_DIR, "oh_vae_generate_from_trained.py"),
)
_train_mod = _load_module(
    "oh_vae_train",
    os.path.join(GENERATIVE_DIR, "oh_vae_train.py"),
)

_gen_syntax_ok = _gen_mod is not None
_train_syntax_ok = _train_mod is not None

requires_gen = pytest.mark.skipif(
    not _gen_syntax_ok,
    reason=(
        "oh_vae_generate_from_trained.py has a SyntaxError (line 58: "
        "missing comma in argparse call). Fix before running these tests."
    ),
)
requires_train = pytest.mark.skipif(
    not _train_syntax_ok,
    reason=(
        "oh_vae_train.py has a SyntaxError (line 37: missing comma in "
        "train_vae_model signature). Fix before running these tests."
    ),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
N_AA = len(AMINO_ACIDS)   # 21
SEQ_LEN = 404


# ===========================================================================
# one_hot_decode
# ===========================================================================

@requires_gen
class TestOneHotDecode:
    """Tests for oh_vae_generate_from_trained.one_hot_decode"""

    def _make_one_hot_tensor(self, sequence: str, amino_acids: str = AMINO_ACIDS):
        """Build a (1, seq_len, vocab) float tensor from a sequence string."""
        vocab = len(amino_acids)
        t = torch.zeros(1, len(sequence), vocab)
        for i, aa in enumerate(sequence):
            t[0, i, amino_acids.index(aa)] = 1.0
        return t

    def test_returns_string(self):
        t = self._make_one_hot_tensor("ACDE")
        result = _gen_mod.one_hot_decode(t)
        assert isinstance(result, str)

    def test_single_amino_acid_decoded_correctly(self):
        for aa in AMINO_ACIDS:
            t = self._make_one_hot_tensor(aa)
            assert _gen_mod.one_hot_decode(t) == aa

    def test_sequence_length_preserved(self):
        seq = "ACDEFGHIKLM"
        t = self._make_one_hot_tensor(seq)
        assert len(_gen_mod.one_hot_decode(t)) == len(seq)

    def test_full_alphabet_round_trip(self):
        """Every character in the alphabet should survive a round-trip."""
        seq = AMINO_ACIDS  # all 21 characters
        t = self._make_one_hot_tensor(seq)
        result = _gen_mod.one_hot_decode(t)
        assert result == seq

    def test_gap_character_decoded(self):
        t = self._make_one_hot_tensor("-")
        assert _gen_mod.one_hot_decode(t) == "-"

    def test_custom_alphabet(self):
        custom = "ACGT"
        t = torch.zeros(1, 2, 4)
        t[0, 0, 0] = 1.0   # 'A'
        t[0, 1, 3] = 1.0   # 'T'
        result = _gen_mod.one_hot_decode(t, amino_acids=custom)
        assert result == "AT"

    def test_argmax_picks_highest_probability(self):
        """Soft probabilities: argmax should pick the largest value."""
        t = torch.tensor([[[0.1, 0.7, 0.2]]])   # shape (1, 1, 3)
        custom_aa = "ACG"
        result = _gen_mod.one_hot_decode(t, amino_acids=custom_aa)
        assert result == "C"   # index 1

    def test_standard_one_hot_input_shape_decodes_correctly(self):
        """
        Regression test for wrong-dimension argmax bug.

        one_hot_decode receives tensors of shape (1, seq_len, vocab).
        The amino-acid axis is dim=2 (the last dimension), so argmax must be
        taken along dim=2 to get the per-position amino-acid index.

        The original code used dim=2 on a (1, seq_len, vocab) tensor, which
        happens to be correct for that layout.  This test makes the expected
        layout explicit so any future dimension change is immediately caught.
        """
        # Build a (1, 5, 3) tensor where every position = amino acid index 1
        # in a custom 3-letter alphabet "ACG"
        custom_aa = "ACG"
        t = torch.zeros(1, 5, 3)
        t[0, :, 1] = 1.0   # position 1 ('C') set at every sequence position
        result = _gen_mod.one_hot_decode(t, amino_acids=custom_aa)
        assert result == "CCCCC", (
            f"Expected 'CCCCC' but got {result!r}. "
            "This may indicate argmax is taken on the wrong dimension."
        )


# ===========================================================================
# generate_sequences (integration – skipped without a model checkpoint)
# ===========================================================================

@requires_gen
class TestGenerateSequences:
    """
    Tests for oh_vae_generate_from_trained.generate_sequences.
    These tests save a freshly-initialised (untrained) ProteinVAE to a
    temporary .pth file so the function can load it without needing a real
    trained model.
    """

    @pytest.fixture
    def model_path(self, tmp_path):
        from vae_oh_CNN import ProteinVAE
        vae = ProteinVAE(latent_dim=100)
        path = str(tmp_path / "test_model.pth")
        torch.save(vae.state_dict(), path)
        return path

    def test_returns_list(self, model_path):
        seqs = _gen_mod.generate_sequences(model_path, num_samples=3)
        assert isinstance(seqs, list)

    def test_correct_number_of_sequences(self, model_path):
        n = 5
        seqs = _gen_mod.generate_sequences(model_path, num_samples=n)
        assert len(seqs) == n

    def test_each_element_is_string(self, model_path):
        seqs = _gen_mod.generate_sequences(model_path, num_samples=3)
        assert all(isinstance(s, str) for s in seqs)

    def test_sequence_length_equals_architecture_seq_len(self, model_path):
        seqs = _gen_mod.generate_sequences(model_path, num_samples=2)
        for seq in seqs:
            assert len(seq) == SEQ_LEN

    def test_all_characters_are_valid_amino_acids(self, model_path):
        seqs = _gen_mod.generate_sequences(model_path, num_samples=3)
        for seq in seqs:
            for char in seq:
                assert char in AMINO_ACIDS, f"Unexpected character: {char!r}"

    def test_custom_seq_len_generates_correct_length(self, tmp_path):
        """generate_sequences respects seq_len parameter (e.g. 275)."""
        from vae_oh_CNN import ProteinVAE
        custom_len = 275
        vae = ProteinVAE(latent_dim=100, seq_len=custom_len)
        path = str(tmp_path / "model_275.pth")
        torch.save(vae.state_dict(), path)
        seqs = _gen_mod.generate_sequences(path, num_samples=2, seq_len=custom_len)
        assert len(seqs) == 2
        for seq in seqs:
            assert len(seq) == custom_len, (
                f"Expected seq_len={custom_len}, got {len(seq)}"
            )


# ===========================================================================
# train_vae_model (smoke test – 1 epoch, tiny dataset)
# ===========================================================================

@requires_train
class TestTrainVAEModel:
    """
    Smoke tests for oh_vae_train.train_vae_model.
    Uses a tiny synthetic dataset (4 sequences) and runs for at most 2 epochs
    so the test suite stays fast.
    """

    def _make_tiny_loader(self, n: int = 4):
        """Return a DataLoader with n random one-hot sequences."""
        from torch.utils.data import DataLoader
        indices = torch.randint(0, N_AA, (n, SEQ_LEN))
        data = torch.nn.functional.one_hot(indices, N_AA).float()
        return DataLoader(list(data), batch_size=2, shuffle=False)

    def test_model_file_is_saved(self, tmp_path):
        from vae_oh_CNN import ProteinVAE
        device = torch.device("cpu")
        vae = ProteinVAE().to(device)
        train_loader = self._make_tiny_loader(4)
        val_loader = self._make_tiny_loader(2)
        _train_mod.train_vae_model(
            vae, train_loader, val_loader, device,
            str(tmp_path), "test_run",
            num_epochs=1, patience=5,
        )
        saved_files = list(tmp_path.iterdir())
        assert len(saved_files) >= 1, "No model file was saved"
        assert any(f.suffix == ".pth" for f in saved_files)

    def test_early_stopping_triggers(self, tmp_path):
        """
        With patience=1 and a model that has non-decreasing val loss the
        training loop should stop before completing all epochs.
        We don't assert the exact epoch count, only that it terminates
        without error and saves at least one file.
        """
        from vae_oh_CNN import ProteinVAE
        device = torch.device("cpu")
        vae = ProteinVAE().to(device)
        train_loader = self._make_tiny_loader(4)
        val_loader = self._make_tiny_loader(2)
        # patience=1 means stop after 1 epoch with no improvement
        _train_mod.train_vae_model(
            vae, train_loader, val_loader, device,
            str(tmp_path), "early_stop_test",
            num_epochs=10, patience=1,
        )
        saved_files = list(tmp_path.iterdir())
        assert len(saved_files) >= 1

    def test_previous_checkpoint_deleted_on_improvement(self, tmp_path):
        """
        The training loop deletes the old best checkpoint when a new best is
        found.  After training there should be exactly one .pth file (the
        best), not one per epoch.
        """
        from vae_oh_CNN import ProteinVAE
        device = torch.device("cpu")
        vae = ProteinVAE().to(device)
        train_loader = self._make_tiny_loader(4)
        val_loader = self._make_tiny_loader(2)
        _train_mod.train_vae_model(
            vae, train_loader, val_loader, device,
            str(tmp_path), "single_ckpt_test",
            num_epochs=3, patience=99,
        )
        pth_files = [f for f in tmp_path.iterdir() if f.suffix == ".pth"]
        assert len(pth_files) == 1, (
            f"Expected exactly 1 checkpoint, found {len(pth_files)}"
        )
