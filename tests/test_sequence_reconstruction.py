"""
Tests for Sequence_Reconstruction/generate-sequence-from-IQ-state.py
and Sequence_Reconstruction/AP-LASR/Species_Name_Truncator_For_PAML.py

Coverage targets (generate-sequence-from-IQ-state):
- generate_sequence: threshold logic, gap on empty numeric row, valid amino-acid choice
- process_sequence_block: gap insertion at correct positions, out-of-range positions ignored
- stream_sequences_to_file: correct number of sequences per node, FASTA format
- insert_gaps: gap characters inserted in right places

Coverage targets (Species_Name_Truncator_For_PAML):
- shorten_species_name: header shortened to 10 chars, short header handled
- process_fasta: output file written with shortened headers, Excel created
"""

import os
import sys
import tempfile
import textwrap

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup – generate-sequence-from-IQ-state.py
# ---------------------------------------------------------------------------
SEQ_RECON_DIR = os.path.join(os.path.dirname(__file__), "..", "Sequence_Reconstruction")
sys.path.insert(0, os.path.abspath(SEQ_RECON_DIR))

# The script has no importable module name (hyphenated filename); import via importlib.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "iq_state",
    os.path.join(SEQ_RECON_DIR, "generate-sequence-from-IQ-state.py"),
)
iq_state = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(iq_state)

# ---------------------------------------------------------------------------
# Path setup – Species_Name_Truncator_For_PAML.py
# Note: the script runs process_fasta at import time with hard-coded Windows paths,
# so we import only the functions we need by exec-ing them selectively.
# ---------------------------------------------------------------------------
APLASR_DIR = os.path.join(SEQ_RECON_DIR, "AP-LASR")
sys.path.insert(0, os.path.abspath(APLASR_DIR))

_truncator_path = os.path.join(APLASR_DIR, "Species_Name_Truncator_For_PAML.py")

# Manually exec only the function definitions to avoid the top-level calls.
_truncator_ns: dict = {}
with open(_truncator_path) as _fh:
    _src = _fh.read()

# Replace the module-level call with a no-op so we can import functions safely.
_safe_src = _src.replace(
    "process_fasta(input_file, output_file, excel_output_file)",
    "pass  # patched out by test suite",
).replace(
    'print("Processing complete',
    'pass  # patched out  #',
).replace(
    'print("Excel sheet',
    'pass  # patched out  #',
)
exec(compile(_safe_src, _truncator_path, "exec"), _truncator_ns)

shorten_species_name = _truncator_ns["shorten_species_name"]
process_fasta = _truncator_ns["process_fasta"]


# ===========================================================================
# Tests for generate-sequence-from-IQ-state.py
# ===========================================================================


class TestGenerateSequence:
    """Unit tests for the generate_sequence() function."""

    def _make_node_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a minimal node DataFrame as expected by generate_sequence."""
        # Columns: Node, Site, State, then amino-acid probability columns
        return pd.DataFrame(rows)

    def test_returns_string(self):
        df = pd.DataFrame([{"Node": "N1", "Site": 1, "State": "A", "A": 0.9, "C": 0.05}])
        node_df = df.iloc[:, 3:]  # drop first 3 cols as the function does
        # Reconstruct a minimal df that generate_sequence expects (it slices [:,3:] itself)
        full_df = pd.DataFrame([{"col0": "N1", "col1": 1, "col2": "A", "A": 0.9, "C": 0.05}])
        result = iq_state.generate_sequence(full_df)
        assert isinstance(result, str)

    def test_length_equals_number_of_rows(self):
        rows = [
            {"col0": "N1", "col1": i, "col2": "X", "A": 0.8, "C": 0.1}
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        result = iq_state.generate_sequence(df)
        assert len(result) == 5

    def test_gap_inserted_when_no_numeric_data(self):
        """A row with no numeric values should yield a '-' placeholder."""
        rows = [
            {"col0": "N1", "col1": 1, "col2": "X", "A": "not_a_number"},
        ]
        df = pd.DataFrame(rows)
        result = iq_state.generate_sequence(df)
        assert result == "-"

    def test_chosen_residue_is_above_threshold(self):
        """The chosen amino acid must be one of the residues above the threshold."""
        # With max_likelihood=0.9 > 0.2, valid = residues with p >= 0.2
        rows = [
            {"col0": "N1", "col1": 1, "col2": "X", "A": 0.9, "C": 0.05, "D": 0.3},
        ]
        df = pd.DataFrame(rows)
        # Run many times to be statistically certain
        chosen = {iq_state.generate_sequence(df) for _ in range(100)}
        assert chosen.issubset({"A", "D"})  # C=0.05 is below threshold 0.2

    def test_fallback_threshold_when_max_below_threshold(self):
        """When max < 0.2, threshold becomes max - 0.1."""
        rows = [
            # max = 0.15 < 0.2, so new_threshold = 0.05; C=0.06 should also be valid
            {"col0": "N1", "col1": 1, "col2": "X", "A": 0.15, "C": 0.06, "D": 0.01},
        ]
        df = pd.DataFrame(rows)
        chosen = {iq_state.generate_sequence(df) for _ in range(100)}
        # D (0.01) is below 0.05, A and C are above
        assert chosen.issubset({"A", "C"})


class TestProcessSequenceBlock:
    def test_gap_inserted_at_correct_position(self):
        gap_positions = {"nodeA": [2]}   # 1-based position 2 → index 1
        seq_block = "nodeA\nACDE\n"
        node, sequence = iq_state.process_sequence_block(seq_block, gap_positions)
        assert node == "nodeA"
        assert sequence[1] == "-"

    def test_non_gap_positions_unchanged(self):
        gap_positions = {"nodeA": [3]}
        seq_block = "nodeA\nACDE\n"
        _, sequence = iq_state.process_sequence_block(seq_block, gap_positions)
        assert sequence[0] == "A"
        assert sequence[1] == "C"
        assert sequence[3] == "E"

    def test_out_of_range_position_ignored(self):
        gap_positions = {"nodeA": [999]}
        seq_block = "nodeA\nACDE\n"
        _, sequence = iq_state.process_sequence_block(seq_block, gap_positions)
        assert "".join(sequence) == "ACDE"

    def test_node_with_no_gap_positions(self):
        gap_positions = {}
        seq_block = "nodeB\nLMNO\n"
        node, sequence = iq_state.process_sequence_block(seq_block, gap_positions)
        assert node == "nodeB"
        assert "".join(sequence) == "LMNO"


class TestStreamSequencesToFile:
    def _make_state_file(self, rows: list[dict]) -> str:
        df = pd.DataFrame(rows)
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False
        )
        df.to_csv(f.name, index=False, sep=" ")
        f.close()
        return f.name

    def test_correct_number_of_sequences_written(self):
        rows = [
            {"Node": "N1", "Site": 1, "State": "A", "A": 0.9, "C": 0.1},
            {"Node": "N1", "Site": 2, "State": "A", "A": 0.8, "C": 0.2},
        ]
        state_file = self._make_state_file(rows)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        try:
            iq_state.stream_sequences_to_file(
                state_file, out_f.name, nodes=["N1"], num_sequences=5
            )
            with open(out_f.name) as fh:
                content = fh.read()
            # Each sequence block has a header (>) line + sequence line
            headers = [l for l in content.splitlines() if l.startswith(">")]
            assert len(headers) == 5
        finally:
            os.unlink(state_file)
            os.unlink(out_f.name)

    def test_fasta_format_headers(self):
        rows = [{"Node": "N2", "Site": 1, "State": "A", "A": 1.0}]
        state_file = self._make_state_file(rows)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        try:
            iq_state.stream_sequences_to_file(
                state_file, out_f.name, nodes=["N2"], num_sequences=3
            )
            with open(out_f.name) as fh:
                lines = fh.read().splitlines()
            assert all(l.startswith(">N2") for l in lines if l.startswith(">"))
        finally:
            os.unlink(state_file)
            os.unlink(out_f.name)


# ===========================================================================
# Tests for Species_Name_Truncator_For_PAML.py
# ===========================================================================


class TestShortenSpeciesName:
    def test_long_header_truncated_to_10_chars(self):
        header = ">Homo_sapiens_12345_extra"
        result = shorten_species_name(header)
        assert result == "Homo_sapie"
        assert len(result) == 10

    def test_short_header_returned_in_full(self):
        header = ">AB"   # only 2 chars after '>'
        result = shorten_species_name(header)
        assert result == "AB"

    def test_exactly_10_chars_after_gt_unchanged(self):
        header = ">1234567890"
        result = shorten_species_name(header)
        assert result == "1234567890"

    def test_gt_is_stripped(self):
        result = shorten_species_name(">ABCDE")
        assert not result.startswith(">")


class TestProcessFasta:
    def _write_fasta(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
        f.write(textwrap.dedent(content))
        f.close()
        return f.name

    def test_output_fasta_has_shortened_headers(self):
        fasta = ">Homo_sapiens_12345\nACDE\n>Mus_musculus_999\nLMNO\n"
        in_f = self._write_fasta(fasta)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        xl_f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        xl_f.close()
        try:
            process_fasta(in_f, out_f.name, xl_f.name)
            with open(out_f.name) as fh:
                lines = fh.read().splitlines()
            headers = [l for l in lines if l.startswith(">")]
            for h in headers:
                # header is ">"+10chars = 11 chars total
                assert len(h) <= 11
        finally:
            for p in (in_f, out_f.name, xl_f.name):
                if os.path.exists(p):
                    os.unlink(p)

    def test_sequences_are_preserved_unchanged(self):
        fasta = ">LongSpeciesName\nACDEFGHI\n"
        in_f = self._write_fasta(fasta)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        xl_f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        xl_f.close()
        try:
            process_fasta(in_f, out_f.name, xl_f.name)
            with open(out_f.name) as fh:
                content = fh.read()
            assert "ACDEFGHI" in content
        finally:
            for p in (in_f, out_f.name, xl_f.name):
                if os.path.exists(p):
                    os.unlink(p)

    def test_excel_file_is_created(self):
        fasta = ">SpeciesA\nACDE\n"
        in_f = self._write_fasta(fasta)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        xl_f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        xl_f.close()
        try:
            process_fasta(in_f, out_f.name, xl_f.name)
            assert os.path.exists(xl_f.name)
            df = pd.read_excel(xl_f.name)
            assert "Original Species Names" in df.columns
            assert "Truncated Species Names" in df.columns
        finally:
            for p in (in_f, out_f.name, xl_f.name):
                if os.path.exists(p):
                    os.unlink(p)

    def test_both_names_stored_in_excel(self):
        fasta = ">Homo_sapiens_12345\nACDE\n"
        in_f = self._write_fasta(fasta)
        out_f = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
        out_f.close()
        xl_f = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        xl_f.close()
        try:
            process_fasta(in_f, out_f.name, xl_f.name)
            df = pd.read_excel(xl_f.name)
            assert ">Homo_sapiens_12345" in df["Original Species Names"].values
        finally:
            for p in (in_f, out_f.name, xl_f.name):
                if os.path.exists(p):
                    os.unlink(p)
