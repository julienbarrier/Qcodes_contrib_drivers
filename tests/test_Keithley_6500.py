"""
Tests for the Keithley DMM6500 driver, focusing on the timetrace (digitize)
functionality added via Keithley6500Buffer, Keithley6500DigitizeSense, and
the _Keithley6500TimeTrace / _Keithley6500TimeAxis parameters.

Uses pyvisa-sim for instrument initialization and pytest monkeypatch for
commands that embed buffer names or require complex sequencing.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcodes_contrib_drivers.drivers.Tektronix.Keithley_6500 import (
    Keithley6500Buffer,
    Keithley6500DigitizeSense,
    Keithley_6500,
)


@pytest.fixture(scope="function")
def dmm():
    driver = Keithley_6500(
        "Keithley_6500",
        address="GPIB0::1::INSTR",
        pyvisa_sim_file="qcodes_contrib_drivers.sims:Keithley_6500.yaml",
    )
    yield driver
    driver.close()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init(dmm: Keithley_6500) -> None:
    idn = dmm.get_idn()
    assert idn["vendor"] == "KEITHLEY INSTRUMENTS INC."
    assert idn["model"] == "DMM6500"


def test_default_buffer_registered(dmm: Keithley_6500) -> None:
    assert dmm.buffer_name() == "defbuffer1"
    buf = dmm.submodules["_buffer_defbuffer1"]
    assert isinstance(buf, Keithley6500Buffer)


def test_digitize_sense_submodules_registered(dmm: Keithley_6500) -> None:
    for func in Keithley6500DigitizeSense.function_modes:
        assert f"_digi_sense_{func}" in dmm.submodules


# ---------------------------------------------------------------------------
# digi_sense_function parameter
# ---------------------------------------------------------------------------


def test_digi_sense_function_set_get(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("voltage")
    assert dmm.digi_sense_function() == "voltage"

    dmm.digi_sense_function("current")
    assert dmm.digi_sense_function() == "current"


def test_digi_sense_function_none(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("None")
    assert dmm.digi_sense_function() == "None"


def test_digi_sense_property_returns_correct_submodule(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("voltage")
    assert isinstance(dmm.digi_sense, Keithley6500DigitizeSense)

    dmm.digi_sense_function("current")
    assert isinstance(dmm.digi_sense, Keithley6500DigitizeSense)


def test_digi_sense_raises_when_none_selected(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("None")
    with pytest.raises(RuntimeError, match="No digitize function selected"):
        _ = dmm.digi_sense


# ---------------------------------------------------------------------------
# digitize_trigger parameter
# ---------------------------------------------------------------------------


def test_digitize_trigger_set_get(dmm: Keithley_6500) -> None:
    dmm.digitize_trigger("NONE")
    assert dmm.digitize_trigger() == "NONE"

    dmm.digitize_trigger("EXT")
    assert dmm.digitize_trigger() == "EXT"


# ---------------------------------------------------------------------------
# Keithley6500DigitizeSense parameters (voltage channel)
# ---------------------------------------------------------------------------


def test_digi_sense_voltage_range(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("voltage")
    dmm.digi_sense.range(10.0)
    assert dmm.digi_sense.range() == pytest.approx(10.0)


def test_digi_sense_voltage_acq_rate(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("voltage")
    dmm.digi_sense.acq_rate(50_000)
    assert dmm.digi_sense.acq_rate() == 50_000


def test_digi_sense_count(dmm: Keithley_6500) -> None:
    dmm.digi_sense_function("voltage")
    dmm.digi_sense.count(500)
    assert dmm.digi_sense.count() == 500


# ---------------------------------------------------------------------------
# Keithley6500Buffer helpers
# ---------------------------------------------------------------------------


def test_buffer_method_returns_existing_buffer(dmm: Keithley_6500) -> None:
    buf1 = dmm.buffer(name="defbuffer1")
    buf2 = dmm.buffer(name="defbuffer1")
    assert buf1 is buf2


def test_buffer_data_start_end_manual_params(dmm: Keithley_6500) -> None:
    buf = dmm.buffer(name="defbuffer1")
    buf.data_start(1)
    buf.data_end(100)
    assert buf.data_start() == 1
    assert buf.data_end() == 100


def test_clear_buffer_sends_abort_then_clear(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clear_buffer() must abort before clearing to avoid hanging the bus."""
    written: list[str] = []
    monkeypatch.setattr(dmm, "write", lambda cmd: written.append(cmd))

    buf = dmm.buffer(name="defbuffer1")
    buf.clear_buffer()

    assert any(":ABORt" in c for c in written), "abort not sent"
    assert any(":TRACe:CLEar" in c for c in written), "clear not sent"

    abort_idx = next(i for i, c in enumerate(written) if ":ABORt" in c)
    clear_idx = next(i for i, c in enumerate(written) if ":TRACe:CLEar" in c)
    assert abort_idx < clear_idx, "abort must precede clear"


# ---------------------------------------------------------------------------
# _Keithley6500TimeTrace.get_raw()
# ---------------------------------------------------------------------------


def _make_ask_mock(n_samples: int, data_str: str):
    """Return an ask() mock that responds to the timetrace command sequence."""

    def mock_ask(cmd: str) -> str:
        if ":MEASure:DIGitize?" in cmd:
            return data_str
        if ":TRACe:ACTual?" in cmd:
            return str(n_samples)
        if ":TRACe:DATA?" in cmd:
            return data_str
        return "0"

    return mock_ask


def test_timetrace_get_returns_array(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 10
    data_values = [float(i) * 0.1 for i in range(n_samples)]
    data_str = ",".join(str(v) for v in data_values)

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")

    monkeypatch.setattr(dmm, "ask", _make_ask_mock(n_samples, data_str))
    monkeypatch.setattr(dmm, "write", lambda cmd: None)

    result = buf.timetrace.get()

    assert isinstance(result, np.ndarray)
    assert len(result) == n_samples
    np.testing.assert_allclose(result, data_values)


def test_timetrace_get_sends_abort_and_clear(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 5
    data_str = ",".join(["1.0"] * n_samples)
    written: list[str] = []

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")

    monkeypatch.setattr(dmm, "ask", _make_ask_mock(n_samples, data_str))
    monkeypatch.setattr(dmm, "write", lambda cmd: written.append(cmd))

    buf.timetrace.get()

    assert any(":ABORt" in c for c in written)
    assert any(":TRACe:CLEar" in c for c in written)
    abort_idx = next(i for i, c in enumerate(written) if ":ABORt" in c)
    clear_idx = next(i for i, c in enumerate(written) if ":TRACe:CLEar" in c)
    assert abort_idx < clear_idx


def test_timetrace_unit_set_to_volts_for_voltage_function(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 5
    data_str = ",".join(["1.0"] * n_samples)

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")

    monkeypatch.setattr(dmm, "ask", _make_ask_mock(n_samples, data_str))
    monkeypatch.setattr(dmm, "write", lambda cmd: None)

    buf.timetrace.get()

    assert buf.timetrace.unit == "V"


def test_timetrace_unit_set_to_amps_for_current_function(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 5
    data_str = ",".join(["0.001"] * n_samples)

    dmm.digi_sense_function("current")
    buf = dmm.buffer(name="defbuffer1")

    monkeypatch.setattr(dmm, "ask", _make_ask_mock(n_samples, data_str))
    monkeypatch.setattr(dmm, "write", lambda cmd: None)

    buf.timetrace.get()

    assert buf.timetrace.unit == "A"


def test_timetrace_trigger_set_to_none_before_acquire(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_raw() must reset digitize_trigger to 'NONE' (immediate trigger)."""
    n_samples = 5
    data_str = ",".join(["0.0"] * n_samples)

    dmm.digi_sense_function("voltage")
    # Start with EXT trigger so we can detect it being reset to NONE
    dmm.digitize_trigger("EXT")
    buf = dmm.buffer(name="defbuffer1")

    monkeypatch.setattr(dmm, "ask", _make_ask_mock(n_samples, data_str))
    monkeypatch.setattr(dmm, "write", lambda cmd: None)

    buf.timetrace.get()

    # Parameter setters update the cache; verify digitize_trigger was reset to NONE
    assert dmm.digitize_trigger.get_latest() == "NONE", (
        "digitize_trigger was not reset to 'NONE' before acquisition"
    )


# ---------------------------------------------------------------------------
# _Keithley6500TimeAxis.get_raw()
# ---------------------------------------------------------------------------


def test_time_axis_length_matches_data_end(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 100
    sample_rate = 10_000

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")
    buf.data_start(1)
    buf.data_end(n_samples)

    monkeypatch.setattr(
        dmm, "ask", lambda cmd: str(sample_rate) if "SRATE" in cmd else "0"
    )

    t_axis = buf.time_axis.get()

    assert len(t_axis) == n_samples


def test_time_axis_starts_at_zero(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 50
    sample_rate = 10_000

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")
    buf.data_start(1)
    buf.data_end(n_samples)

    monkeypatch.setattr(
        dmm, "ask", lambda cmd: str(sample_rate) if "SRATE" in cmd else "0"
    )

    t_axis = buf.time_axis.get()

    assert t_axis[0] == pytest.approx(0.0)


def test_time_axis_end_matches_sample_rate(
    dmm: Keithley_6500, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_samples = 50
    sample_rate = 10_000

    dmm.digi_sense_function("voltage")
    buf = dmm.buffer(name="defbuffer1")
    buf.data_start(1)
    buf.data_end(n_samples)

    monkeypatch.setattr(
        dmm, "ask", lambda cmd: str(sample_rate) if "SRATE" in cmd else "0"
    )

    t_axis = buf.time_axis.get()

    assert t_axis[-1] == pytest.approx((n_samples - 1) / sample_rate)
