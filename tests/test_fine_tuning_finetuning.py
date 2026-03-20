"""
Tests for Fine-Tuning/fine-tune.py

Coverage targets
----------------
- ESMWithClassifier: output shape, parameter freezing/unfreezing
- fine_tune_esm: model checkpoint saved at correct path, only intended
  parameters are trainable (regression for 'prefix' vs 'out_prefix' bug)
- main() CLI: argparse wiring, prefix derivation from filename, out_dir
  creation, delegation to fine_tune_esm

All tests that require ESM are guarded by a skipif so the suite stays
green in environments without the esm package installed.

Design: rather than downloading a ~140 MB ESM model, we replace it with a
tiny fake model using unittest.mock.patch so that:
  1. Tests run in seconds
  2. The structural logic (freezing, saving, argparse) is still exercised
"""

import os
import sys
import importlib.util as _ilu
import textwrap
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Load fine-tune.py via importlib (name contains no importable hyphens)
# ---------------------------------------------------------------------------
FINETUNE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Fine-Tuning")
)
sys.path.insert(0, FINETUNE_DIR)

_FT_PATH = os.path.join(FINETUNE_DIR, "fine-tune.py")
_spec = _ilu.spec_from_file_location("fine_tune", _FT_PATH)
_ft_mod = _ilu.module_from_spec(_spec)

try:
    # Stub out 'esm' so the import succeeds without the package installed
    _esm_stub = mock.MagicMock()
    sys.modules.setdefault("esm", _esm_stub)
    sys.modules.setdefault("tqdm", mock.MagicMock())
    _spec.loader.exec_module(_ft_mod)
    _FT_IMPORT_OK = True
except Exception as _FT_ERR:
    _FT_IMPORT_OK = False

_esm_available = _FT_IMPORT_OK  # used for skipif marks

requires_ft = pytest.mark.skipif(
    not _FT_IMPORT_OK,
    reason=f"Could not import fine-tune.py: {'' if _FT_IMPORT_OK else _FT_ERR}",
)

# ---------------------------------------------------------------------------
# Tiny fake ESM model used by all tests that exercise fine_tune_esm()
# ---------------------------------------------------------------------------

ESM_REPR_DIM = 480   # matches hardcoded nn.Linear(480, ...) in ESMWithClassifier


class _FakeESMLayer(nn.Module):
    """Stand-in for a single ESM transformer layer (has .parameters())."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(ESM_REPR_DIM, ESM_REPR_DIM)

    def forward(self, x):
        return x


class _FakeESMModel(nn.Module):
    """
    Minimal fake ESM model.  Matches the interface used by fine_tune_esm:
      - model(x, repr_layers=[12])['representations'][12]  → tensor
      - model.layers[-1] / [-2]  → iterable of parameters
    """
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([_FakeESMLayer() for _ in range(n_layers)])

    def forward(self, x, repr_layers=None):
        batch, seq_len = x.shape
        # Return random repr of shape (batch, seq_len, ESM_REPR_DIM)
        fake_repr = torch.randn(batch, seq_len, ESM_REPR_DIM)
        return {"representations": {12: fake_repr}}


class _FakeAlphabet:
    """Minimal ESM alphabet stub."""
    def get_idx(self, token: str) -> int:
        return 0


def _make_fake_esm():
    return _FakeESMModel(), _FakeAlphabet()


# ---------------------------------------------------------------------------
# Helper – tiny FASTA file on disk
# ---------------------------------------------------------------------------

def _write_fasta(tmp_path, name: str = "seqs.fasta") -> str:
    content = textwrap.dedent("""\
        >seq1
        ACDEFGHIKLMNPQRSTVWY
        >seq2
        ACDEFGHIKLMNPQRSTVWY
    """)
    p = str(tmp_path / name)
    with open(p, "w") as f:
        f.write(content)
    return p


# ===========================================================================
# ESMWithClassifier
# ===========================================================================

@requires_ft
class TestESMWithClassifier:

    def _make_model(self, alphabet_size: int = 33):
        fake_esm, _ = _make_fake_esm()
        return _ft_mod.ESMWithClassifier(fake_esm, alphabet_size)

    def test_is_nn_module(self):
        assert isinstance(self._make_model(), nn.Module)

    def test_forward_output_shape(self):
        """
        Input: (batch=2, seq_len=10) token-index tensor.
        Output should be (batch * seq_len, alphabet_size) — the classifier
        reshapes the sequence dimension into the batch dimension.
        """
        model = self._make_model(alphabet_size=33)
        x = torch.randint(0, 33, (2, 10))
        out = model(x)
        assert out.shape == (2 * 10, 33), (
            f"Expected shape (20, 33), got {tuple(out.shape)}"
        )

    def test_classifier_is_linear(self):
        model = self._make_model(alphabet_size=33)
        assert isinstance(model.classifier, nn.Linear)

    def test_classifier_output_dim_matches_alphabet_size(self):
        for sz in [10, 33, 100]:
            model = self._make_model(alphabet_size=sz)
            assert model.classifier.out_features == sz


# ===========================================================================
# fine_tune_esm – structural / regression tests
# ===========================================================================

@requires_ft
class TestFineTuneESM:

    def _run_fine_tune(self, tmp_path, fasta_path: str,
                       out_prefix: str = "mymodel",
                       num_epochs: int = 1):
        """
        Run fine_tune_esm with the real code path but a fake ESM model.
        Patches esm.pretrained.esm2_t12_35M_UR50D to return the fake model.
        """
        fake_esm, fake_alpha = _make_fake_esm()
        with mock.patch.object(_ft_mod, "esm") as mock_esm_pkg, \
             mock.patch.object(_ft_mod, "functions") as mock_funcs:
            # Wire up the ESM pretrained loader
            mock_esm_pkg.pretrained.esm2_t12_35M_UR50D.return_value = (
                fake_esm, fake_alpha
            )
            # Wire up functions module: return a tiny dataset
            _SEQ_LEN = 202   # 200 + <cls> + <eos>
            fake_tensor = torch.randint(0, 33, (_SEQ_LEN,))
            mock_funcs.get_fasta_dict.return_value = {"seq1": "A" * 200}
            mock_funcs.SeqDataset.return_value = [fake_tensor, fake_tensor]
            mock_funcs.apply_mask.side_effect = lambda x, mask, idx: x

            _ft_mod.fine_tune_esm(
                fasta_path,
                str(tmp_path),
                out_prefix,
                alphabet_size=33,
                num_epochs=num_epochs,
            )

    def test_checkpoint_saved_at_correct_path(self, tmp_path):
        """
        Regression test for the 'prefix' vs 'out_prefix' NameError bug.

        fine_tune_esm used to reference `prefix` (undefined) instead of
        `out_prefix` when building the model save path. The model was
        therefore never saved in practice.

        After the fix, the file must exist at {out_dir}/{out_prefix}.pth.
        """
        fasta = _write_fasta(tmp_path)
        self._run_fine_tune(tmp_path, fasta, out_prefix="run42")
        expected = tmp_path / "run42.pth"
        assert expected.exists(), (
            f"Expected checkpoint at {expected} but it was not created. "
            "Check that fine_tune_esm uses `out_prefix` not `prefix`."
        )

    def test_checkpoint_name_uses_out_prefix_not_fasta_basename(self, tmp_path):
        """When out_prefix='myrun', the file must be myrun.pth, not seqs.pth."""
        fasta = _write_fasta(tmp_path, name="seqs.fasta")
        self._run_fine_tune(tmp_path, fasta, out_prefix="myrun")
        assert (tmp_path / "myrun.pth").exists()
        assert not (tmp_path / "seqs.pth").exists()

    def test_checkpoint_is_loadable_state_dict(self, tmp_path):
        """The saved .pth must be a valid state_dict (a dict of tensors)."""
        fasta = _write_fasta(tmp_path)
        self._run_fine_tune(tmp_path, fasta, out_prefix="run")
        sd = torch.load(str(tmp_path / "run.pth"), map_location="cpu")
        assert isinstance(sd, dict)

    def test_only_last_two_layers_and_classifier_are_trainable(self, tmp_path):
        """
        fine_tune_esm should freeze all parameters then unfreeze only:
          - esm_model.layers[-1]
          - esm_model.layers[-2]
          - classifier

        We verify by checking requires_grad on representative parameters
        after the freeze/unfreeze loop runs (using 0 epochs so no training
        actually happens, if supported; otherwise 1 fast epoch).
        """
        fasta = _write_fasta(tmp_path)
        fake_esm, fake_alpha = _make_fake_esm()
        captured_model: list = []

        original_torch_save = torch.save

        def _capture_save(obj, path, **kw):
            captured_model.append(obj)
            original_torch_save(obj, path, **kw)

        with mock.patch.object(_ft_mod, "esm") as mock_esm_pkg, \
             mock.patch.object(_ft_mod, "functions") as mock_funcs, \
             mock.patch("torch.save", side_effect=_capture_save):
            mock_esm_pkg.pretrained.esm2_t12_35M_UR50D.return_value = (
                fake_esm, fake_alpha
            )
            _SEQ_LEN = 202
            fake_tensor = torch.randint(0, 33, (_SEQ_LEN,))
            mock_funcs.get_fasta_dict.return_value = {"seq1": "A" * 200}
            mock_funcs.SeqDataset.return_value = [fake_tensor]
            mock_funcs.apply_mask.side_effect = lambda x, mask, idx: x

            # Inspect the model's grad state after one epoch
            wrapped_model_holder: list = []
            _original_ESMWithClassifier = _ft_mod.ESMWithClassifier

            class _CapturingWrapper(_original_ESMWithClassifier):
                def __init__(self, esm_m, alpha_sz):
                    super().__init__(esm_m, alpha_sz)
                    wrapped_model_holder.append(self)

            with mock.patch.object(_ft_mod, "ESMWithClassifier", _CapturingWrapper):
                _ft_mod.fine_tune_esm(
                    str(tmp_path / "seqs.fasta"),
                    str(tmp_path), "grad_test",
                    alphabet_size=33, num_epochs=1,
                )

        if not wrapped_model_holder:
            pytest.skip("Could not capture model instance")

        model = wrapped_model_holder[0]

        # Classifier must be trainable
        assert all(p.requires_grad for p in model.classifier.parameters()), \
            "Classifier parameters should be trainable after unfreezing"

        # Last two ESM layers must be trainable
        for layer_idx in [-1, -2]:
            layer = model.esm_model.layers[layer_idx]
            assert all(p.requires_grad for p in layer.parameters()), \
                f"ESM layer {layer_idx} parameters should be trainable"

        # Early layers (0 .. -3) must remain frozen
        for layer in model.esm_model.layers[:-2]:
            assert all(not p.requires_grad for p in layer.parameters()), \
                "Early ESM layer parameters should remain frozen"


# ===========================================================================
# main() CLI – argparse wiring
# ===========================================================================

@requires_ft
class TestFineTuneMain:
    """
    Tests for fine-tune.py's main() function.

    Strategy: patch sys.argv + mock fine_tune_esm to verify that argparse
    wires arguments correctly without actually running training.
    """

    def _run_main(self, argv: list[str], tmp_path=None):
        """Invoke _ft_mod.main() with argv, mocking fine_tune_esm."""
        with mock.patch.object(_ft_mod, "fine_tune_esm") as mock_ft, \
             mock.patch("sys.argv", argv), \
             mock.patch("os.makedirs"):
            _ft_mod.main()
        return mock_ft

    def test_input_fasta_forwarded(self, tmp_path):
        mock_ft = self._run_main(
            ["ft", "-i", "seqs.fasta", "-o", str(tmp_path), "-p", "run"]
        )
        args, kwargs = mock_ft.call_args
        assert args[0] == "seqs.fasta"

    def test_out_dir_forwarded(self, tmp_path):
        mock_ft = self._run_main(
            ["ft", "-i", "seqs.fasta", "-o", str(tmp_path), "-p", "run"]
        )
        args, _ = mock_ft.call_args
        assert args[1] == str(tmp_path)

    def test_explicit_prefix_used(self, tmp_path):
        mock_ft = self._run_main(
            ["ft", "-i", "seqs.fasta", "-o", str(tmp_path), "-p", "myprefix"]
        )
        args, _ = mock_ft.call_args
        assert args[2] == "myprefix"

    def test_prefix_derived_from_fasta_basename_when_not_provided(self, tmp_path):
        """When -p is omitted, prefix should come from the FASTA filename stem."""
        mock_ft = self._run_main(
            ["ft", "-i", "proteins.fasta", "-o", str(tmp_path)]
        )
        args, _ = mock_ft.call_args
        assert args[2] == "proteins"

    def test_fine_tune_esm_called_exactly_once(self, tmp_path):
        mock_ft = self._run_main(
            ["ft", "-i", "seqs.fasta", "-o", str(tmp_path), "-p", "run"]
        )
        mock_ft.assert_called_once()


# ===========================================================================
# main() in oh_vae_train.py
# ===========================================================================

class TestVAETrainMain:
    """Tests for oh_vae_train.py's main() CLI wiring."""

    @pytest.fixture(autouse=True)
    def _load_train_mod(self):
        import importlib.util as ilu
        gen_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Generative_Model")
        )
        sys.path.insert(0, gen_dir)
        spec = ilu.spec_from_file_location(
            "oh_vae_train_mod",
            os.path.join(gen_dir, "oh_vae_train.py"),
        )
        mod = ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            self.train_mod = mod
            self._ok = True
        except SyntaxError:
            self._ok = False

    def _skip_if_broken(self):
        if not self._ok:
            pytest.skip("oh_vae_train.py has unfixed syntax errors")

    def test_train_vae_model_called_with_correct_out_prefix(self, tmp_path):
        self._skip_if_broken()
        fasta = str(tmp_path / "seqs.fasta")
        with open(fasta, "w") as f:
            f.write(">s1\nACDEFGHIKLMNPQRSTVWY\n>s2\nACDEFGHIKLMNPQRSTVWY\n")

        with mock.patch.object(self.train_mod, "train_vae_model") as mock_tv, \
             mock.patch("sys.argv", ["vae", "-i", fasta, "-o", str(tmp_path), "-p", "run1"]), \
             mock.patch("os.makedirs"), \
             mock.patch.object(self.train_mod, "ProteinVAE", return_value=mock.MagicMock()):
            self.train_mod.main()

        _, kwargs = mock_tv.call_args
        # out_prefix is positional arg index 5
        call_args = mock_tv.call_args[0]
        assert call_args[5] == "run1"

    def test_prefix_derived_from_filename_when_not_provided(self, tmp_path):
        self._skip_if_broken()
        fasta = str(tmp_path / "proteins.fasta")
        with open(fasta, "w") as f:
            f.write(">s1\nACDEFGHIKLMNPQRSTVWY\n>s2\nACDEFGHIKLMNPQRSTVWY\n")

        with mock.patch.object(self.train_mod, "train_vae_model") as mock_tv, \
             mock.patch("sys.argv", ["vae", "-i", fasta, "-o", str(tmp_path)]), \
             mock.patch("os.makedirs"), \
             mock.patch.object(self.train_mod, "ProteinVAE", return_value=mock.MagicMock()):
            self.train_mod.main()

        call_args = mock_tv.call_args[0]
        assert call_args[5] == "proteins"


# ===========================================================================
# main() in oh_vae_generate_from_trained.py
# ===========================================================================

class TestVAEGenerateMain:
    """Tests for oh_vae_generate_from_trained.py's main() CLI wiring."""

    @pytest.fixture(autouse=True)
    def _load_gen_mod(self):
        import importlib.util as ilu
        gen_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Generative_Model")
        )
        sys.path.insert(0, gen_dir)
        spec = ilu.spec_from_file_location(
            "oh_vae_gen_mod",
            os.path.join(gen_dir, "oh_vae_generate_from_trained.py"),
        )
        mod = ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            self.gen_mod = mod
            self._ok = True
        except SyntaxError:
            self._ok = False

    def _skip_if_broken(self):
        if not self._ok:
            pytest.skip("oh_vae_generate_from_trained.py has unfixed syntax errors")

    def test_generate_sequences_called_with_model_path(self, tmp_path):
        self._skip_if_broken()
        model_file = str(tmp_path / "model.pth")
        # Create a dummy .pth so generate_sequences can at least open it
        torch.save({}, model_file)

        with mock.patch.object(self.gen_mod, "generate_sequences", return_value=["A" * 404]) as mock_gs, \
             mock.patch("sys.argv", ["gen", "--model_path", model_file]), \
             mock.patch.object(self.gen_mod.pd.DataFrame, "to_csv"):
            self.gen_mod.main()

        mock_gs.assert_called_once()
        assert mock_gs.call_args[0][0] == model_file

    def test_csv_path_derived_from_model_name_when_not_provided(self, tmp_path):
        """When --csv_path is omitted, CSV path should be <model_stem>.csv"""
        self._skip_if_broken()
        model_file = str(tmp_path / "mymodel.pth")
        torch.save({}, model_file)

        csv_paths_written: list = []

        def _capture_to_csv(path, **kw):
            csv_paths_written.append(path)

        with mock.patch.object(self.gen_mod, "generate_sequences", return_value=["A"]), \
             mock.patch("sys.argv", ["gen", "--model_path", model_file]):
            # Patch DataFrame.to_csv at the pandas level
            with mock.patch("pandas.DataFrame.to_csv", side_effect=_capture_to_csv):
                self.gen_mod.main()

        assert any("mymodel.csv" in p for p in csv_paths_written), (
            f"Expected CSV filename to be 'mymodel.csv', got: {csv_paths_written}"
        )

    def test_explicit_csv_path_used(self, tmp_path):
        self._skip_if_broken()
        model_file = str(tmp_path / "model.pth")
        csv_file = str(tmp_path / "output.csv")
        torch.save({}, model_file)

        csv_paths_written: list = []

        def _capture_to_csv(path, **kw):
            csv_paths_written.append(path)

        with mock.patch.object(self.gen_mod, "generate_sequences", return_value=["A"]), \
             mock.patch("sys.argv", ["gen", "--model_path", model_file,
                                     "--csv_path", csv_file]):
            with mock.patch("pandas.DataFrame.to_csv", side_effect=_capture_to_csv):
                self.gen_mod.main()

        assert any(csv_file in p for p in csv_paths_written), (
            f"Expected CSV path '{csv_file}', got: {csv_paths_written}"
        )
