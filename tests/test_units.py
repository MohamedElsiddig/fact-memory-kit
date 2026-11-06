"""Tests for exact unit canonicalisation (factkit.units)."""
from __future__ import annotations

import pytest

from factkit.units import canonical, conflicts, negated, ports_overlap

GB = 1024 ** 3
MB = 1024 ** 2
TB = 1024 ** 4


# --------------------------------------------------------------- capacities
def test_capacity_units_normalise_to_same_bytes():
    assert canonical("unit", "16GB") == (16 * GB,)
    assert canonical("unit", "16 GB") == canonical("unit", "16GB")
    assert canonical("unit", "16 gb") == canonical("unit", "16GB")
    assert canonical("unit", "16384MB") == canonical("unit", "16GB")
    assert conflicts("unit", "16GB", "16384MB") is False


def test_capacity_tb_and_attr_aliases():
    assert canonical("storage", "1TB") == (TB,)
    assert canonical("disk", "1024GB") == canonical("ssd", "1TB")
    assert canonical(" RAM ", "16 GB") == canonical("memory", "16gb")
    assert conflicts("storage", "1TB", "1024GB") is False


def test_capacity_differences_are_real_conflicts():
    assert conflicts("unit", "16GB", "32GB") is True
    assert conflicts("ram", "512MB", "1GB") is True
    assert canonical("unit", "1.5TB") == (int(1.5 * TB),)


# -------------------------------------------------------------------- clock
def test_clock_ghz_and_mhz_are_equal():
    assert canonical("clock", "3.6GHz") == (3600.0,)
    assert canonical("clock", "3600 MHz") == (3600.0,)
    assert conflicts("clock", "3.6GHz", "3600 MHz") is False


def test_clock_difference_is_a_conflict():
    assert conflicts("clock", "3.6GHz", "3.2GHz") is True


# -------------------------------------------------------------------- cores
def test_cores_parse_int_and_suffix():
    assert canonical("cores", "16") == (16,)
    assert canonical("cores", "16 cores") == (16,)
    assert conflicts("cores", "8 cores", "8") is False
    assert conflicts("cores", "8", "16") is True


# -------------------------------------------------------------------- ports
def test_port_single_and_range_shapes():
    assert canonical("port", "8080") == ("single", 8080)
    assert canonical("port", " 8080 ") == ("single", 8080)
    assert canonical("port", "8000-8100") == ("range", 8000, 8100)
    assert canonical("port", "8000 - 8100") == ("range", 8000, 8100)
    assert canonical("port", "8080-8080") == ("single", 8080)


def test_ports_overlap_vs_disjoint():
    assert ports_overlap("8000-8100", "8080") is True
    assert ports_overlap("8000-8100", "8100-8200") is True
    assert ports_overlap("8000-8100", "9000-9100") is False
    assert ports_overlap("8080", "8081") is False
    assert ports_overlap("8080", "8080") is True


def test_port_conflicts_only_on_overlap():
    assert conflicts("port", "8080", "8080") is False
    assert conflicts("port", "8000-8100", "8080") is True
    assert conflicts("port", "8000-8100", "9000-9100") is False
    assert conflicts("port", "8080", "8081") is False
    assert conflicts("port", "8000-8100", "8100-8200") is True


# ----------------------------------------------------------------- versions
def test_version_prefix_and_short_forms_are_equal():
    assert canonical("version", "v1.2.3") == (1, 2, 3)
    assert canonical("version", "1.2.3") == (1, 2, 3)
    assert conflicts("version", "v1.2.3", "1.2.3") is False
    assert conflicts("version", "1.2", "1.2.0") is False
    assert conflicts("version", "v2.0", "2.0.0") is False


def test_version_difference_is_a_conflict():
    assert conflicts("version", "v1.2.3", "1.2.4") is True
    assert conflicts("version", "1.2.3", "1.2") is True
    assert conflicts("version", "2.0", "2.1.0") is True


# -------------------------------------------------------- unknown attributes
def test_unknown_attr_falls_back_to_case_insensitive_string():
    assert canonical("owner", " Atlas Server ") == ("atlas server",)
    assert conflicts("owner", "Atlas Server", " atlas server ") is False
    assert conflicts("owner", "atlas", "beacon") is True


# ----------------------------------------------------------------- negation
def test_negation_cues_detected():
    assert negated("Node no longer listens on port 7000") is True
    assert negated("Backup is not enabled") is True
    assert negated("Atlas doesn't run Docker") is True
    assert negated("Atlas does not run Docker") is True
    assert negated("Uses port 9000 instead of 8000") is True
    assert negated("The service stopped listening") is True


def test_negation_is_word_boundary_safe():
    assert negated("Notification service is healthy") is False
    assert negated("This is a notable notebook") is False
    assert negated("Atlas runs Docker") is False
    assert negated("") is False
    assert negated(None) is False


def test_negation_does_not_confuse_substrings():
    assert negated("The knot is tied") is False
    assert negated("An unstopped process") is False


# --------------------------------------------------------------------- junk
def test_junk_inputs_never_raise_and_fall_back():
    junk = ["", "   ", None, "n/a", "many", "1e9 GB", "not-a-port", "-", "8080-"]
    for value in junk:
        for attr in ("unit", "ram", "clock", "cores", "port", "version", "owner", None):
            result = canonical(attr, value)
            assert isinstance(result, tuple)
            assert result

    assert canonical("unit", None) == ("",)
    assert canonical("ram", "many") == ("many",)
    assert canonical("port", "not-a-port") == ("not-a-port",)
    assert canonical("clock", "fast") == ("fast",)
    assert canonical("cores", "many") == ("many",)
    assert canonical("version", "latest") == ("latest",)


def test_junk_conflicts_do_not_raise():
    assert conflicts("unit", "", "") is False
    assert conflicts("unit", None, "") is False
    assert conflicts("port", "not-a-port", "8080") is False
    assert conflicts("port", "not-a-port", "also-junk") is False
    assert ports_overlap("not-a-port", "8080") is False
    assert ports_overlap(None, "") is False


def test_known_attr_unparseable_values_compare_as_strings():
    assert canonical("unit", "lots") == canonical("unit", "lots ")
    assert conflicts("unit", "lots", "lots") is False
    assert conflicts("unit", "lots", "more") is True


@pytest.mark.parametrize(
    "attr,a,b,expected",
    [
        ("unit", "16GB", "16384MB", False),
        ("unit", "16GB", "32GB", True),
        ("port", "8000-8100", "8080", True),
        ("port", "8000-8100", "9000-9100", False),
        ("version", "1.2", "1.2.0", False),
        ("version", "1.2", "1.3", True),
        ("clock", "3.6GHz", "3600MHz", False),
        ("cores", "4", "4 cores", False),
    ],
)
def test_conflict_matrix(attr, a, b, expected):
    assert conflicts(attr, a, b) is expected
