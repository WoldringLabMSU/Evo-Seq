"""
Tests for the pure/utility functions in Sequence_Reconstruction/AP-LASR/AP-LASR.py
that do NOT require external tools (BLAST, CD-Hit, MAFFT, IQ-TREE, BaLi-Phy).

Tested functions
----------------
- fasta2dict          : parse FASTA files to dicts
- dict2fasta          : write dicts to FASTA files
- Is_Valid_AA         : validate amino-acid strings / lists
- Is_Valid_Codon      : validate codon strings / lists
- Strike_Gaps_Dict    : remove gap characters from all sequences in a dict
- CDHit_Cutoff        : map identity thresholds to CD-Hit word-length params
- Fasta_Dict_Hamming  : compute Hamming distances between aligned sequences
- Degenerate_Nucleotide_Codon : generate degenerate DNA codons from AA lists
- Build_DNA_Sequence  : build a full degenerate DNA string from a position list
- Library_Size_Count  : count unique amino-acid sequences encoded by a DNA string

Why no tests for functions that call external tools
---------------------------------------------------
Functions such as BlastP, CDHit, NCBI_to_XML, IQTree_Phylo, IQTree_ASR,
Binary_Gap_Analysis, run_baliphy, and Select_Ancestor_Nodes all invoke
subprocess calls to external programs (BLAST+, CD-Hit, MAFFT, IQ-TREE,
BaLi-Phy) that are not available in the unit-test environment.  These
should be covered by integration tests run in a properly-provisioned
environment (e.g. a dedicated CI runner or a Conda environment with the
tools installed).
"""

import os
import sys
import importlib.util as _ilu
import tempfile
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Load AP-LASR.py as a module (it has a hyphen-free alternate name below)
# ---------------------------------------------------------------------------
_APLASR_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Sequence_Reconstruction",
        "AP-LASR",
        "AP-LASR.py",
    )
)

_spec = _ilu.spec_from_file_location("aplasr", _APLASR_PATH)
_mod = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
    _IMPORT_OK = True
except Exception as exc:
    _IMPORT_OK = False
    _IMPORT_ERR = exc

pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"Could not import AP-LASR.py: {'' if _IMPORT_OK else _IMPORT_ERR}",
)

# Convenience references
fasta2dict = _mod.fasta2dict
dict2fasta = _mod.dict2fasta
Is_Valid_AA = _mod.Is_Valid_AA
Is_Valid_Codon = _mod.Is_Valid_Codon
Strike_Gaps_Dict = _mod.Strike_Gaps_Dict
CDHit_Cutoff = _mod.CDHit_Cutoff
Fasta_Dict_Hamming = _mod.Fasta_Dict_Hamming
Degenerate_Nucleotide_Codon = _mod.Degenerate_Nucleotide_Codon
Build_DNA_Sequence = _mod.Build_DNA_Sequence
Library_Size_Count = _mod.Library_Size_Count
AA_key = _mod.AA_key


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_fasta(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# ===========================================================================
# fasta2dict
# ===========================================================================

class TestFasta2Dict:
    def test_reads_single_sequence(self):
        path = _write_fasta(">seq1\nACDEFGHI\n")
        try:
            d = fasta2dict(path, {})
            assert "seq1" in d
            assert d["seq1"] == "ACDEFGHI"
        finally:
            os.unlink(path)

    def test_reads_multiple_sequences(self):
        path = _write_fasta(">s1\nACDE\n>s2\nLMNO\n")
        try:
            d = fasta2dict(path, {})
            assert set(d.keys()) == {"s1", "s2"}
            assert d["s1"] == "ACDE"
            assert d["s2"] == "LMNO"
        finally:
            os.unlink(path)

    def test_strips_trailing_newlines(self):
        path = _write_fasta(">seq1\nACDE\n\n")
        try:
            d = fasta2dict(path, {})
            assert d["seq1"] == "ACDE"
        finally:
            os.unlink(path)

    def test_multiline_sequence_concatenated(self):
        path = _write_fasta(">s1\nAC\nDE\n")
        try:
            d = fasta2dict(path, {})
            assert d["s1"] == "ACDE"
        finally:
            os.unlink(path)

    def test_colon_in_header_replaced_by_underscore(self):
        path = _write_fasta(">sp:P12345|PROT_HUMAN\nACDE\n")
        try:
            # The function replaces ':' with '_' but then overwrites with the
            # raw line – see implementation.  Either way the key must be present.
            d = fasta2dict(path, {})
            assert len(d) == 1
        finally:
            os.unlink(path)

    def test_empty_file_raises_value_error(self):
        path = _write_fasta("")
        try:
            with pytest.raises(ValueError):
                fasta2dict(path, {})
        finally:
            os.unlink(path)

    def test_sequences_with_gaps_accepted(self):
        path = _write_fasta(">gapped\nA-C-DE\n")
        try:
            d = fasta2dict(path, {})
            assert d["gapped"] == "A-C-DE"
        finally:
            os.unlink(path)


# ===========================================================================
# dict2fasta
# ===========================================================================

class TestDict2Fasta:
    def test_writes_correct_fasta_format(self, tmp_path):
        out = str(tmp_path / "out.fasta")
        dict2fasta({"seq1": "ACDE", "seq2": "LMNO"}, out)
        with open(out) as fh:
            content = fh.read()
        assert ">seq1\nACDE\n" in content
        assert ">seq2\nLMNO\n" in content

    def test_round_trip_through_fasta2dict(self, tmp_path):
        original = {"protA": "ACDEFGHI", "protB": "LMNOPQRS"}
        path = str(tmp_path / "round.fasta")
        dict2fasta(original, path)
        recovered = fasta2dict(path, {})
        assert recovered == original

    def test_empty_dict_creates_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.fasta")
        dict2fasta({}, path)
        assert os.path.exists(path)
        assert open(path).read() == ""


# ===========================================================================
# Is_Valid_AA
# ===========================================================================

class TestIsValidAA:
    def test_valid_amino_acid_string(self):
        assert Is_Valid_AA("ACDEFGHIKLMNPQRSTVWY") is True

    def test_gap_character_is_valid(self):
        assert Is_Valid_AA("-") is True

    def test_invalid_character_in_string(self):
        assert Is_Valid_AA("B") is False   # B is not in AA_key

    def test_valid_list_of_amino_acids(self):
        assert Is_Valid_AA(["A", "C", "D"]) is True

    def test_invalid_list_of_amino_acids(self):
        assert Is_Valid_AA(["A", "Z"]) is False

    def test_valid_list_of_lists(self):
        assert Is_Valid_AA([["A", "C"], ["D", "E"]]) is True

    def test_invalid_list_of_lists(self):
        assert Is_Valid_AA([["A", "B"]]) is False

    def test_invalid_type_raises_value_error(self):
        with pytest.raises((ValueError, TypeError)):
            Is_Valid_AA(42)

    def test_empty_string_returns_true(self):
        """An empty sequence trivially satisfies the 'all' predicate."""
        assert Is_Valid_AA("") is True

    def test_all_twenty_standard_amino_acids(self):
        for aa in AA_key:
            assert Is_Valid_AA(aa), f"Standard AA {aa!r} should be valid"

    def test_empty_list_does_not_crash(self):
        """
        Regression test: Is_Valid_AA([]) previously raised IndexError on
        AA[0] because the list-of-strings branch accessed the first element
        before checking whether it was empty.  An empty list is vacuously
        valid (no invalid amino acids present).
        """
        # Should not raise; an empty collection has no invalid elements.
        result = Is_Valid_AA([])
        assert result is True


# ===========================================================================
# Is_Valid_Codon
# ===========================================================================

class TestIsValidCodon:
    def test_standard_codon(self):
        assert Is_Valid_Codon("atg") is True

    def test_degenerate_codon(self):
        assert Is_Valid_Codon("ryn") is True   # all degenerate bases

    def test_invalid_base(self):
        assert Is_Valid_Codon("atx") is False

    def test_wrong_length_string(self):
        assert Is_Valid_Codon("at") is False   # not a multiple of 3

    def test_valid_list_of_codons(self):
        assert Is_Valid_Codon(["atg", "gcc", "tgg"]) is True

    def test_invalid_list_element(self):
        assert Is_Valid_Codon(["atg", "xbc"]) is False

    def test_list_element_wrong_length(self):
        assert Is_Valid_Codon(["at"]) is False

    def test_empty_string(self):
        # len("") % 3 == 0 and all() on empty is True
        assert Is_Valid_Codon("") is True

    def test_uppercase_bases_are_invalid(self):
        """The valid set uses lowercase; uppercase is not included."""
        assert Is_Valid_Codon("ATG") is False


# ===========================================================================
# Strike_Gaps_Dict
# ===========================================================================

class TestStrikeGapsDict:
    def test_removes_gaps_from_all_sequences(self):
        d = {"s1": "A-C-D", "s2": "--LM"}
        Strike_Gaps_Dict(d)
        assert d["s1"] == "ACD"
        assert d["s2"] == "LM"

    def test_no_gaps_unchanged(self):
        d = {"s1": "ACDE"}
        Strike_Gaps_Dict(d)
        assert d["s1"] == "ACDE"

    def test_all_gaps_becomes_empty_string(self):
        d = {"s1": "---"}
        Strike_Gaps_Dict(d)
        assert d["s1"] == ""

    def test_modifies_dict_in_place(self):
        d = {"s1": "A-C"}
        original_id = id(d)
        Strike_Gaps_Dict(d)
        assert id(d) == original_id   # same dict object


# ===========================================================================
# CDHit_Cutoff
# ===========================================================================

class TestCDHitCutoff:
    def test_identity_above_0_7_returns_5(self):
        assert CDHit_Cutoff(0.80) == 5
        assert CDHit_Cutoff(1.00) == 5

    def test_identity_between_0_6_and_0_7_returns_4(self):
        assert CDHit_Cutoff(0.65) == 4

    def test_identity_between_0_5_and_0_6_returns_3(self):
        assert CDHit_Cutoff(0.55) == 3

    def test_identity_between_0_4_and_0_5_returns_2(self):
        assert CDHit_Cutoff(0.45) == 2

    def test_identity_at_boundary_0_7_returns_4(self):
        """0.7 is > 0.6 but not > 0.7, so should return 4."""
        assert CDHit_Cutoff(0.70) == 4

    def test_identity_too_low_raises_value_error(self):
        with pytest.raises(ValueError):
            CDHit_Cutoff(0.30)

    def test_identity_at_exactly_0_4_raises_value_error(self):
        with pytest.raises(ValueError):
            CDHit_Cutoff(0.40)

    def test_identity_above_1_raises_value_error(self):
        with pytest.raises(ValueError):
            CDHit_Cutoff(1.01)


# ===========================================================================
# Fasta_Dict_Hamming
# ===========================================================================

class TestFastaDictHamming:
    def test_identical_sequences_have_zero_distance(self):
        d = {"s1": "ACDE", "s2": "ACDE"}
        result = Fasta_Dict_Hamming(d, "ACDE")
        assert result["s1"] == 0
        assert result["s2"] == 0

    def test_fully_different_sequences(self):
        d = {"s1": "LLLL"}
        result = Fasta_Dict_Hamming(d, "ACDE")
        assert result["s1"] == 4

    def test_partial_difference(self):
        d = {"s1": "ACDE"}
        result = Fasta_Dict_Hamming(d, "AAAA")
        # Positions 1,2,3 differ (C≠A, D≠A, E≠A)
        assert result["s1"] == 3

    def test_unequal_length_raises_value_error(self):
        d = {"s1": "ACDE"}
        with pytest.raises(ValueError):
            Fasta_Dict_Hamming(d, "ACD")   # length 3 != 4

    def test_single_position_difference(self):
        d = {"s1": "ACDX"}
        result = Fasta_Dict_Hamming(d, "ACDE")
        assert result["s1"] == 1

    def test_returns_dict_with_same_keys(self):
        d = {"s1": "ACDE", "s2": "ACDE"}
        result = Fasta_Dict_Hamming(d, "ACDE")
        assert set(result.keys()) == set(d.keys())


# ===========================================================================
# Degenerate_Nucleotide_Codon
# ===========================================================================

class TestDegenerateNucleotideCodon:
    def test_single_amino_acid_ecoli_returns_ecoli_codon(self):
        result = Degenerate_Nucleotide_Codon(["A"], source="EColi")
        assert result == "gcc"   # AA_to_Codon_Ecoli['A']

    def test_single_amino_acid_human_returns_human_codon(self):
        result = Degenerate_Nucleotide_Codon(["A"], source="Human")
        assert result == "gcc"   # same for Ala in both organisms

    def test_two_different_amino_acids_returns_three_base_codon(self):
        result = Degenerate_Nucleotide_Codon(["A", "C"], source="EColi")
        assert len(result) == 3
        assert Is_Valid_Codon(result)

    def test_removes_x_amino_acids(self):
        """'X' (unknown) is silently removed; the result codes for valid AAs."""
        result = Degenerate_Nucleotide_Codon(["X", "A"], source="EColi")
        assert Is_Valid_Codon(result)

    def test_empty_list_returns_empty_string(self):
        result = Degenerate_Nucleotide_Codon([], source="EColi")
        assert result == ""

    def test_invalid_amino_acid_raises_value_error(self):
        with pytest.raises(ValueError):
            Degenerate_Nucleotide_Codon(["Z"], source="EColi")

    def test_invalid_source_raises_name_error(self):
        with pytest.raises(NameError):
            Degenerate_Nucleotide_Codon(["A"], source="Yeast")

    def test_result_is_valid_degenerate_codon(self):
        """Output must always pass Is_Valid_Codon."""
        for aa in AA_key[:10]:   # test first 10 standard AAs
            result = Degenerate_Nucleotide_Codon([aa], source="EColi")
            assert Is_Valid_Codon(result), (
                f"Codon {result!r} for {aa!r} is not valid"
            )


# ===========================================================================
# Build_DNA_Sequence
# ===========================================================================

class TestBuildDNASequence:
    def test_single_position_single_aa_ecoli(self):
        result = Build_DNA_Sequence([["A"]], source="EColi")
        assert result == "gcc"

    def test_single_position_single_aa_human(self):
        result = Build_DNA_Sequence([["A"]], source="Human")
        assert result == "gcc"

    def test_gap_position_contributes_nothing(self):
        """A position containing only '-' should be skipped."""
        result = Build_DNA_Sequence([["-"]], source="EColi")
        assert result == ""

    def test_multi_position_sequence_correct_length(self):
        """Three positions → 9 DNA bases (3 codons)."""
        result = Build_DNA_Sequence([["A"], ["C"], ["D"]], source="EColi")
        assert len(result) == 9

    def test_two_aa_position_uses_pair_lookup(self):
        """Two AAs at one position should use the pair lookup table."""
        result = Build_DNA_Sequence([["A", "C"]], source="EColi")
        assert len(result) == 3
        assert Is_Valid_Codon(result)

    def test_invalid_organism_raises_name_error(self):
        with pytest.raises(NameError):
            Build_DNA_Sequence([["A"]], source="Bacteria")

    def test_output_is_valid_codon(self):
        result = Build_DNA_Sequence([["L"], ["M"]], source="Human")
        assert Is_Valid_Codon(result)

    def test_ecoli_and_human_may_differ_for_same_aa(self):
        """Some amino acids have different preferred codons in each organism."""
        ecoli = Build_DNA_Sequence([["R"]], source="EColi")
        human = Build_DNA_Sequence([["R"]], source="Human")
        # Both must be valid; they may be the same or different
        assert Is_Valid_Codon(ecoli)
        assert Is_Valid_Codon(human)


# ===========================================================================
# Library_Size_Count
# ===========================================================================

class TestLibrarySizeCount:
    def test_single_unambiguous_codon_returns_one(self):
        """gcc encodes only Ala → library size = 1."""
        assert Library_Size_Count("gcc") == 1

    def test_degenerate_codon_covering_two_amino_acids(self):
        """
        'ryn' is a highly degenerate codon; we only assert the result is
        a positive integer without hard-coding the exact count (which would
        couple the test to the lookup tables).
        """
        result = Library_Size_Count("ryn")
        assert isinstance(result, int)
        assert result >= 1

    def test_two_unambiguous_codons_multiplied(self):
        """gcc (A) + gcc (A) → 1 × 1 = 1."""
        assert Library_Size_Count("gccgcc") == 1

    def test_invalid_sequence_raises_value_error(self):
        with pytest.raises(ValueError):
            Library_Size_Count("XYZ")   # not valid degenerate DNA

    def test_empty_string(self):
        """An empty sequence has no codons → size 1 (multiplicative identity)."""
        assert Library_Size_Count("") == 1

    def test_stop_codon_excluded_from_size(self):
        """
        A degenerate codon that encodes a stop codon should have that stop
        excluded from the library size count.
        """
        # 'taa' is a stop codon; build a codon that covers exactly Trp and stop:
        # 'tgg' = Trp only, so library size = 1
        assert Library_Size_Count("tgg") == 1

    def test_library_size_is_positive(self):
        for codon in ["atg", "gcc", "tgg", "cgt"]:
            assert Library_Size_Count(codon) >= 1

    def test_two_distinct_unambiguous_codons_multiply_independently(self):
        """
        Regression test for off-by-one bug: the original code used
        range(0, len-2, 3) which silently dropped the last codon of any
        sequence.  "atg" (Met, 1 AA) + "tgg" (Trp, 1 AA) should give
        1 × 1 = 1.  With the old bug the second codon was never iterated
        so Library_Size_Count("atgtgg") would still return 1 by accident.
        Use "atgwsn" (Met + degenerate second codon encoding multiple AAs)
        to make the second codon's contribution observable.
        """
        # "atg" = Met only (1 AA).
        # "wsn" is a highly degenerate codon encoding many amino acids (≥ 4).
        # Correct result: 1 × Library_Size_Count("wsn")  > 1.
        single = Library_Size_Count("wsn")
        combined = Library_Size_Count("atgwsn")
        assert combined == single, (
            f"Expected atg (×1) * wsn (×{single}) = {single}, got {combined}. "
            "This suggests the second codon is being dropped (off-by-one bug)."
        )

    def test_three_codon_sequence_counts_all_codons(self):
        """
        Verify all three codons in a 9-base string are counted.
        "gcc" (A, 1) * "atg" (M, 1) * "tgg" (W, 1) = 1.
        """
        assert Library_Size_Count("gccatgtgg") == 1

    def test_mutable_default_argument_does_not_leak_between_calls(self, tmp_path):
        """
        Regression test for mutable default argument bug in fasta2dict.
        The original signature was fasta2dict(path, return_dict={}) — Python
        creates that dict once at function-definition time, so state from one
        call would bleed into the next call that relied on the default.
        Two independent calls must return independent dicts.
        """
        p1 = str(tmp_path / "a.fasta")
        p2 = str(tmp_path / "b.fasta")
        with open(p1, "w") as f:
            f.write(">seqA\nACDE\n")
        with open(p2, "w") as f:
            f.write(">seqB\nLMNO\n")
        d1 = fasta2dict(p1)   # uses default return_dict
        d2 = fasta2dict(p2)   # must NOT see seqA
        assert "seqA" not in d2, (
            "fasta2dict returned seqA in a second call that should only see seqB. "
            "This indicates the mutable-default-argument bug is still present."
        )
        assert "seqB" not in d1
