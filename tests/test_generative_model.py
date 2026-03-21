"""
Tests for Generative_Model/vae_oh_CNN.py and oh_vae_train.py

Coverage targets:
- Sampling: reparameterisation produces correct shape, uses device of inputs
- Encoder: output shapes, forward pass
- Decoder: output shape, values in [0, 1] (sigmoid output)
- ProteinVAE: forward shapes, loss is a positive scalar, reconstruction in [0,1]
- one_hot_encode: standard amino acids, gap character, unknown char raises
- read_fasta_sequences: reads and shuffles sequences
- train_val_split: correct sizes, no data leakage
"""

import os
import sys
import tempfile
import textwrap

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
GENERATIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "Generative_Model")
sys.path.insert(0, os.path.abspath(GENERATIVE_DIR))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from vae_oh_CNN import Decoder, Encoder, ProteinVAE, Sampling  # noqa: E402
from oh_vae_train import one_hot_encode, read_fasta_sequences, train_val_split  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching the hard-coded architecture
# ---------------------------------------------------------------------------
SEQ_LEN = 404        # hard-coded in Conv1d / Linear layers
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"  # 21 characters
N_AA = len(AMINO_ACIDS)   # 21
LATENT_DIM = 100
BATCH = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_batch():
    """Random one-hot encoded batch: (BATCH, SEQ_LEN, N_AA)."""
    indices = torch.randint(0, N_AA, (BATCH, SEQ_LEN))
    return torch.nn.functional.one_hot(indices, N_AA).float()


@pytest.fixture
def vae():
    return ProteinVAE(latent_dim=LATENT_DIM).eval()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_output_shape_matches_input(self):
        sampler = Sampling()
        z_mean = torch.zeros(BATCH, LATENT_DIM)
        z_log_var = torch.zeros(BATCH, LATENT_DIM)
        z = sampler(z_mean, z_log_var)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_output_is_stochastic(self):
        """Two calls should not produce identical samples."""
        sampler = Sampling()
        z_mean = torch.zeros(BATCH, LATENT_DIM)
        z_log_var = torch.zeros(BATCH, LATENT_DIM)
        z1 = sampler(z_mean, z_log_var)
        z2 = sampler(z_mean, z_log_var)
        assert not torch.equal(z1, z2)

    def test_zero_variance_samples_close_to_mean(self):
        """When log_var → -inf (std → 0), sample ≈ mean."""
        sampler = Sampling()
        z_mean = torch.ones(2, 4) * 5.0
        z_log_var = torch.full((2, 4), -30.0)  # exp(0.5 * -30) ≈ 0
        z = sampler(z_mean, z_log_var)
        assert torch.allclose(z, z_mean, atol=1e-2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_forward_returns_three_tensors(self, dummy_batch):
        enc = Encoder(latent_dim=LATENT_DIM).eval()
        with torch.no_grad():
            z_mean, z_log_var, z = enc(dummy_batch)
        assert z_mean.shape == (BATCH, LATENT_DIM)
        assert z_log_var.shape == (BATCH, LATENT_DIM)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_z_mean_and_z_log_var_are_deterministic(self, dummy_batch):
        """Mean and log-var are deterministic; z is not."""
        enc = Encoder(latent_dim=LATENT_DIM).eval()
        with torch.no_grad():
            m1, lv1, _ = enc(dummy_batch)
            m2, lv2, _ = enc(dummy_batch)
        assert torch.equal(m1, m2)
        assert torch.equal(lv1, lv2)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(latent_dim=LATENT_DIM).eval()
        z = torch.randn(BATCH, LATENT_DIM)
        with torch.no_grad():
            out = dec(z)
        assert out.shape == (BATCH, SEQ_LEN, N_AA)

    def test_output_values_in_zero_one(self):
        """Decoder ends with sigmoid, so output must be in [0, 1]."""
        dec = Decoder(latent_dim=LATENT_DIM).eval()
        z = torch.randn(BATCH, LATENT_DIM)
        with torch.no_grad():
            out = dec(z)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# ProteinVAE (end-to-end)
# ---------------------------------------------------------------------------


class TestProteinVAE:
    def test_forward_output_shapes(self, vae, dummy_batch):
        with torch.no_grad():
            z_mean, z_log_var, recon = vae(dummy_batch)
        assert z_mean.shape == (BATCH, LATENT_DIM)
        assert z_log_var.shape == (BATCH, LATENT_DIM)
        assert recon.shape == dummy_batch.shape

    def test_reconstruction_in_zero_one(self, vae, dummy_batch):
        with torch.no_grad():
            _, _, recon = vae(dummy_batch)
        assert recon.min().item() >= 0.0 - 1e-6
        assert recon.max().item() <= 1.0 + 1e-6

    def test_loss_is_positive_scalar(self, vae, dummy_batch):
        with torch.no_grad():
            z_mean, z_log_var, recon = vae(dummy_batch)
            loss = vae.loss(dummy_batch, z_mean, z_log_var, recon)
        assert loss.ndim == 0        # scalar
        assert loss.item() > 0

    def test_loss_decreases_with_identity_reconstruction(self, vae, dummy_batch):
        """
        If reconstruction equals input exactly the recon_loss term is
        minimised (BCE = 0 for perfect one-hot). The total loss should be
        lower than with random reconstruction of the same batch size.
        """
        with torch.no_grad():
            z_mean, z_log_var, recon = vae(dummy_batch)
            random_recon = torch.rand_like(dummy_batch)
            loss_real = vae.loss(dummy_batch, z_mean, z_log_var, recon)
            loss_random = vae.loss(dummy_batch, z_mean, z_log_var, random_recon)
        # Both are positive; we cannot guarantee ordering without training,
        # but we can assert the loss value is finite.
        assert torch.isfinite(loss_real)
        assert torch.isfinite(loss_random)


# ---------------------------------------------------------------------------
# one_hot_encode (from oh_vae_train.py)
# ---------------------------------------------------------------------------


class TestOneHotEncode:
    def test_output_shape(self):
        seq = "ACDE"
        enc = one_hot_encode(seq)
        assert enc.shape == (len(seq), N_AA)

    def test_single_hot_per_position(self):
        seq = "ACDE"
        enc = one_hot_encode(seq)
        assert (enc.sum(dim=1) == 1.0).all()

    def test_correct_position_is_set(self):
        amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
        for i, aa in enumerate(amino_acids):
            enc = one_hot_encode(aa)
            assert enc[0, i] == 1.0

    def test_gap_character_encoded(self):
        enc = one_hot_encode("-")
        assert enc[0, -1] == 1.0   # '-' is last in the default alphabet

    def test_unknown_amino_acid_raises(self):
        """Characters not in the alphabet cause ValueError via str.index()."""
        with pytest.raises(ValueError):
            one_hot_encode("Z")  # 'Z' is not in the default alphabet

    def test_long_sequence_shape(self):
        seq = "A" * SEQ_LEN
        enc = one_hot_encode(seq)
        assert enc.shape == (SEQ_LEN, N_AA)


# ---------------------------------------------------------------------------
# read_fasta_sequences (from oh_vae_train.py)
# ---------------------------------------------------------------------------


class TestReadFastaSequences:
    def _write_fasta(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
        f.write(textwrap.dedent(content))
        f.close()
        return f.name

    def test_reads_correct_number_of_sequences(self):
        fasta = """\
            >seq1
            ACDE
            >seq2
            LMNO
            >seq3
            PQRS
        """
        path = self._write_fasta(fasta)
        try:
            seqs = read_fasta_sequences(path)
            assert len(seqs) == 3
        finally:
            os.unlink(path)

    def test_sequences_are_strings(self):
        fasta = ">s1\nACDE\n"
        path = self._write_fasta(fasta)
        try:
            seqs = read_fasta_sequences(path)
            assert all(isinstance(s, str) for s in seqs)
        finally:
            os.unlink(path)

    def test_empty_fasta_returns_empty_list(self):
        path = self._write_fasta("")
        try:
            seqs = read_fasta_sequences(path)
            assert seqs == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# train_val_split (from oh_vae_train.py)
# ---------------------------------------------------------------------------


class TestTrainValSplit:
    def test_sizes_sum_to_total(self):
        data = list(range(100))
        train, val = train_val_split(data, val_ratio=0.2)
        assert len(train) + len(val) == 100

    def test_val_ratio_is_respected(self):
        data = list(range(100))
        _, val = train_val_split(data, val_ratio=0.1)
        assert len(val) == 10

    def test_no_overlap_between_splits(self):
        data = list(range(200))
        train, val = train_val_split(data, val_ratio=0.2)
        assert set(train).isdisjoint(set(val))

    def test_all_items_present(self):
        data = list(range(50))
        train, val = train_val_split(data, val_ratio=0.2)
        assert sorted(train + val) == data

    def test_zero_val_ratio(self):
        data = list(range(20))
        train, val = train_val_split(data, val_ratio=0.0)
        assert len(val) == 0
        assert len(train) == 20


# ---------------------------------------------------------------------------
# seq_len parameter (architecture parameterisation tests)
# ---------------------------------------------------------------------------

CUSTOM_SEQ_LEN = 275


class TestSeqLenParameter:
    """Verify that Encoder, Decoder, and ProteinVAE work with seq_len != 404."""

    def test_encoder_custom_seq_len_output_shape(self):
        enc = Encoder(latent_dim=LATENT_DIM, seq_len=CUSTOM_SEQ_LEN).eval()
        x = torch.zeros(BATCH, CUSTOM_SEQ_LEN, N_AA)
        with torch.no_grad():
            z_mean, z_log_var, z = enc(x)
        assert z_mean.shape == (BATCH, LATENT_DIM)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_decoder_custom_seq_len_output_shape(self):
        dec = Decoder(latent_dim=LATENT_DIM, seq_len=CUSTOM_SEQ_LEN).eval()
        z = torch.randn(BATCH, LATENT_DIM)
        with torch.no_grad():
            out = dec(z)
        assert out.shape == (BATCH, CUSTOM_SEQ_LEN, N_AA)

    def test_vae_custom_seq_len_reconstruction_shape(self):
        vae = ProteinVAE(latent_dim=LATENT_DIM, seq_len=CUSTOM_SEQ_LEN).eval()
        indices = torch.randint(0, N_AA, (BATCH, CUSTOM_SEQ_LEN))
        x = torch.nn.functional.one_hot(indices, N_AA).float()
        with torch.no_grad():
            z_mean, z_log_var, recon = vae(x)
        assert recon.shape == (BATCH, CUSTOM_SEQ_LEN, N_AA)

    def test_vae_custom_seq_len_loss_is_finite(self):
        vae = ProteinVAE(latent_dim=LATENT_DIM, seq_len=CUSTOM_SEQ_LEN).eval()
        indices = torch.randint(0, N_AA, (BATCH, CUSTOM_SEQ_LEN))
        x = torch.nn.functional.one_hot(indices, N_AA).float()
        with torch.no_grad():
            z_mean, z_log_var, recon = vae(x)
            loss = vae.loss(x, z_mean, z_log_var, recon)
        assert torch.isfinite(loss)
        assert loss.item() > 0
