"""Driver for Rigol DSA800 series spectrum analyzers.

Programming reference:
https://m.testlink.co.kr/download/rigol/DSA800_ProgrammingGuide_EN.pdf

I have tested this driver on a DSA815-TG, but it should be compatible with other DSA800 models as well.

See `docs/examples/` for example notebooks.

Written by Edward Laird (http://wp.lancs.ac.uk/laird-group/).
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from qcodes.instrument import InstrumentChannel, VisaInstrument
from qcodes.parameters import (
    ParameterWithSetpoints,
    create_on_off_val_mapping,
)
from qcodes.validators import Arrays, Enum, Ints, Numbers

log = logging.getLogger(__name__)


_MODEL_MAX_FREQUENCY_HZ = {
    815: 1.5e9,
    832: 3.2e9,
    875: 7.5e9,
}

_ON_OFF_MAPPING = create_on_off_val_mapping(on_val=1, off_val=0)


def _strip_ieee_block_header(payload: str) -> str:
    """Remove a definite-length IEEE 488.2 block header when present."""
    text = payload.strip()
    if not text.startswith("#") or len(text) < 2 or not text[1].isdigit():
        return text

    digits = int(text[1])
    if len(text) < 2 + digits:
        raise ValueError(f"Malformed IEEE block header: {payload!r}")

    header_end = 2 + digits
    data_length = int(text[2:header_end])
    data_end = header_end + data_length
    return text[header_end:data_end].strip()


def _parse_ascii_trace_data(payload: str) -> npt.NDArray[np.float64]:
    """Parse an ASCII trace returned by ``:TRACe:DATA?``."""
    body = _strip_ieee_block_header(payload)
    if not body:
        return np.array([], dtype=float)
    return np.fromstring(body, sep=",", dtype=float)


def _infer_max_frequency_from_model(model: str) -> float:
    """Infer the analyzer frequency limit from the model string."""
    match = re.search(r"DSA(\d{3})", model.upper())
    if match is None:
        log.warning(
            "Could not infer maximum frequency from model %r. "
            "Falling back to 7.5 GHz.",
            model,
        )
        return 7.5e9

    model_code = int(match.group(1))
    return _MODEL_MAX_FREQUENCY_HZ.get(model_code, 7.5e9)


def _power_unit_label(power_unit: str) -> str:
    """Return a display-friendly unit label for trace-like parameters."""
    return power_unit


class RigolDSA800Trace(ParameterWithSetpoints):
    """Trace parameter with analyzer axis setpoints."""

    def __init__(
        self,
        *args: Any,
        trace_index: int,
        **kwargs: Any,
    ) -> None:
        self._trace_index = trace_index
        super().__init__(*args, **kwargs)

    def get_raw(self) -> npt.NDArray[np.float64]:
        instrument = cast(RigolDSA800, self.root_instrument)
        self.unit = _power_unit_label(str(instrument.power_unit()))
        return instrument._query_trace_data(self._trace_index)


class RigolDSA800Marker(InstrumentChannel):
    """Marker control and readout for a single analyzer marker."""

    def __init__(self, parent: "RigolDSA800", name: str, marker_index: int) -> None:
        super().__init__(parent, name)
        self._marker_index = marker_index

        self.enabled = self.add_parameter(
            "enabled",
            label=f"Marker {marker_index} enabled",
            set_cmd=f":CALCulate:MARKer{marker_index}:STATe {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:STATe?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.mode = self.add_parameter(
            "mode",
            label=f"Marker {marker_index} mode",
            set_cmd=f":CALCulate:MARKer{marker_index}:MODE {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:MODE?",
            vals=Enum("POS", "DELT", "BAND", "SPAN"),
        )

        self.trace = self.add_parameter(
            "trace",
            label=f"Marker {marker_index} trace",
            set_cmd=f":CALCulate:MARKer{marker_index}:TRACe {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:TRACe?",
            get_parser=int,
            vals=Ints(1, 3),
        )

        self.trace_auto = self.add_parameter(
            "trace_auto",
            label=f"Marker {marker_index} auto trace assignment",
            set_cmd=f":CALCulate:MARKer{marker_index}:TRACe:AUTO {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:TRACe:AUTO?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.x = self.add_parameter(
            "x",
            label=f"Marker {marker_index} x-value",
            set_cmd=f":CALCulate:MARKer{marker_index}:X {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:X?",
            get_parser=float,
        )

        self.y = self.add_parameter(
            "y",
            label=f"Marker {marker_index} y-value",
            get_cmd=f":CALCulate:MARKer{marker_index}:Y?",
            get_parser=float,
        )

        self.peak_excursion = self.add_parameter(
            "peak_excursion",
            label=f"Marker {marker_index} peak excursion",
            unit="dB",
            set_cmd=f":CALCulate:MARKer{marker_index}:PEAK:EXCursion {{}}",
            get_cmd=f":CALCulate:MARKer{marker_index}:PEAK:EXCursion?",
            get_parser=float,
            vals=Numbers(0, 200),
        )

    def peak_search(self) -> None:
        """Move the marker to the strongest peak."""
        self.write(f":CALCulate:MARKer{self._marker_index}:MAXimum:MAX")

    def next_peak(self) -> None:
        """Move the marker to the next peak."""
        self.write(f":CALCulate:MARKer{self._marker_index}:MAXimum:NEXT")

    def peak_left(self) -> None:
        """Move the marker to the nearest peak on the left."""
        self.write(f":CALCulate:MARKer{self._marker_index}:MAXimum:LEFT")

    def peak_right(self) -> None:
        """Move the marker to the nearest peak on the right."""
        self.write(f":CALCulate:MARKer{self._marker_index}:MAXimum:RIGHt")

    def minimum_search(self) -> None:
        """Move the marker to the minimum-amplitude point."""
        self.write(f":CALCulate:MARKer{self._marker_index}:MINimum")


class RigolDSA800TraceChannel(InstrumentChannel):
    """Trace configuration and readout for a single displayed trace."""

    def __init__(self, parent: "RigolDSA800", name: str, trace_index: int) -> None:
        super().__init__(parent, name)
        self._trace_index = trace_index

        self.mode = self.add_parameter(
            "mode",
            label=f"Trace {trace_index} mode",
            set_cmd=f":TRACe{trace_index}:MODE {{}}",
            get_cmd=f":TRACe{trace_index}:MODE?",
            vals=Enum("WRIT", "MAXH", "MINH", "VIEW", "BLANK", "VID", "POW"),
        )

        self.average_type = self.add_parameter(
            "average_type",
            label=f"Trace {trace_index} average type",
            set_cmd=f":TRACe{trace_index}:AVERage:TYPE {{}}",
            get_cmd=f":TRACe{trace_index}:AVERage:TYPE?",
            vals=Enum("VID", "RMS"),
        )

        self.data = self.add_parameter(
            "data",
            label=f"Trace {trace_index} data",
            unit="dBm",
            parameter_class=RigolDSA800Trace,
            trace_index=trace_index,
            setpoints=(parent.trace_axis,),
            vals=Arrays(shape=(parent._point_count,)),
        )


class RigolDSA800TrackingGenerator(InstrumentChannel):
    """Tracking-generator control for TG-equipped DSA800 analyzers."""

    def __init__(self, parent: "RigolDSA800", name: str) -> None:
        super().__init__(parent, name)

        self.enabled = self.add_parameter(
            "enabled",
            label="Tracking generator output",
            set_cmd=":OUTPut:STATe {}",
            get_cmd=":OUTPut:STATe?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.amplitude_offset = self.add_parameter(
            "amplitude_offset",
            label="Tracking generator amplitude offset",
            unit="dB",
            set_cmd=":SOURce:CORRection:OFFSet {}",
            get_cmd=":SOURce:CORRection:OFFSet?",
            get_parser=float,
            vals=Numbers(-200, 200),
        )

        self.fixed_power_amplitude = self.add_parameter(
            "fixed_power_amplitude",
            label="Tracking generator fixed power amplitude",
            unit="dBm",
            set_cmd=":SOURce:POWer:LEVel:IMMediate:AMPLitude {}",
            get_cmd=":SOURce:POWer:LEVel:IMMediate:AMPLitude?",
            get_parser=float,
            vals=Numbers(-40, 0),
        )

        self.power_mode = self.add_parameter(
            "power_mode",
            label="Tracking generator power mode",
            set_cmd=":SOURce:POWer:MODE {}",
            get_cmd=":SOURce:POWer:MODE?",
            vals=Enum("FIX", "SWE"),
        )

        self.fixed_power_span = self.add_parameter(
            "fixed_power_span",
            label="Tracking generator fixed power span",
            unit="dB",
            set_cmd=":SOURce:POWer:SPAN {}",
            get_cmd=":SOURce:POWer:SPAN?",
            get_parser=float,
            vals=Numbers(0, 20),
        )

        self.sweep_start_amplitude = self.add_parameter(
            "sweep_start_amplitude",
            label="Tracking generator sweep start amplitude",
            unit="dBm",
            set_cmd=":SOURce:POWer:STARt {}",
            get_cmd=":SOURce:POWer:STARt?",
            get_parser=float,
            vals=Numbers(-40, 0),
        )

        self.sweep_amplitude_range = self.add_parameter(
            "sweep_amplitude_range",
            label="Tracking generator sweep amplitude range",
            unit="dB",
            set_cmd=":SOURce:POWer:SWEep {}",
            get_cmd=":SOURce:POWer:SWEep?",
            get_parser=float,
            vals=Numbers(0, 20),
        )

        self.reference_trace_enabled = self.add_parameter(
            "reference_trace_enabled",
            label="Tracking generator normalization reference trace",
            set_cmd=":SOURce:TRACe:REF:STATe {}",
            get_cmd=":SOURce:TRACe:REF:STATe?",
            val_mapping=_ON_OFF_MAPPING,
        )

    def enable(self) -> None:
        """Enable the tracking-generator output."""
        self.enabled("ON")

    def disable(self) -> None:
        """Disable the tracking-generator output."""
        self.enabled("OFF")

    def store_reference_trace(self) -> None:
        """Store the normalization reference trace."""
        self.write(":SOURce:TRACe:STORref")


class RigolDSA800(VisaInstrument):
    """QCoDeS driver for Rigol DSA800 series spectrum analyzers."""

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        kwargs.setdefault("terminator", "\n")
        kwargs.setdefault("device_clear", False)
        super().__init__(name, address, **kwargs)

        idn = self.get_idn()
        self._model = idn.get("model") or "DSA800"
        self.max_frequency_hz = _infer_max_frequency_from_model(self._model)

        self.frequency_start = self.add_parameter(
            "frequency_start",
            label="Start frequency",
            unit="Hz",
            set_cmd=":SENSe:FREQuency:STARt {}",
            get_cmd=":SENSe:FREQuency:STARt?",
            get_parser=float,
            vals=Numbers(0, self.max_frequency_hz),
        )

        self.frequency_stop = self.add_parameter(
            "frequency_stop",
            label="Stop frequency",
            unit="Hz",
            set_cmd=":SENSe:FREQuency:STOP {}",
            get_cmd=":SENSe:FREQuency:STOP?",
            get_parser=float,
            vals=Numbers(0, self.max_frequency_hz),
        )

        self.frequency_center = self.add_parameter(
            "frequency_center",
            label="Center frequency",
            unit="Hz",
            set_cmd=":SENSe:FREQuency:CENTer {}",
            get_cmd=":SENSe:FREQuency:CENTer?",
            get_parser=float,
            vals=Numbers(0, self.max_frequency_hz),
        )

        self.frequency_span = self.add_parameter(
            "frequency_span",
            label="Frequency span",
            unit="Hz",
            set_cmd=":SENSe:FREQuency:SPAN {}",
            get_cmd=":SENSe:FREQuency:SPAN?",
            get_parser=float,
            vals=Numbers(0, self.max_frequency_hz),
        )

        self.resolution_bandwidth = self.add_parameter(
            "resolution_bandwidth",
            label="Resolution bandwidth",
            unit="Hz",
            set_cmd=":SENSe:BANDwidth:RESolution {}",
            get_cmd=":SENSe:BANDwidth:RESolution?",
            get_parser=float,
            vals=Numbers(10, 1e6),
        )

        self.resolution_bandwidth_auto = self.add_parameter(
            "resolution_bandwidth_auto",
            label="Auto RBW",
            set_cmd=":SENSe:BANDwidth:RESolution:AUTO {}",
            get_cmd=":SENSe:BANDwidth:RESolution:AUTO?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.video_bandwidth = self.add_parameter(
            "video_bandwidth",
            label="Video bandwidth",
            unit="Hz",
            set_cmd=":SENSe:BANDwidth:VIDeo {}",
            get_cmd=":SENSe:BANDwidth:VIDeo?",
            get_parser=float,
            vals=Numbers(1, 3e6),
        )

        self.video_bandwidth_auto = self.add_parameter(
            "video_bandwidth_auto",
            label="Auto VBW",
            set_cmd=":SENSe:BANDwidth:VIDeo:AUTO {}",
            get_cmd=":SENSe:BANDwidth:VIDeo:AUTO?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.detector = self.add_parameter(
            "detector",
            label="Detector",
            set_cmd=":SENSe:DETector:FUNCtion {}",
            get_cmd=":SENSe:DETector:FUNCtion?",
            vals=Enum("NEG", "NORM", "POS", "RMS", "SAMP", "VAV", "QPEAK"),
        )

        self.attenuation = self.add_parameter(
            "attenuation",
            label="Input attenuation",
            unit="dB",
            set_cmd=":SENSe:POWer:RF:ATTenuation {}",
            get_cmd=":SENSe:POWer:RF:ATTenuation?",
            get_parser=float,
            vals=Numbers(0, 30),
        )

        self.attenuation_auto = self.add_parameter(
            "attenuation_auto",
            label="Auto attenuation",
            set_cmd=":SENSe:POWer:RF:ATTenuation:AUTO {}",
            get_cmd=":SENSe:POWer:RF:ATTenuation:AUTO?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.preamplifier_enabled = self.add_parameter(
            "preamplifier_enabled",
            label="Preamplifier",
            set_cmd=":SENSe:POWer:RF:GAIN:STATe {}",
            get_cmd=":SENSe:POWer:RF:GAIN:STATe?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.reference_level = self.add_parameter(
            "reference_level",
            label="Reference level",
            unit="dBm",
            set_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:RLEVel {}",
            get_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:RLEVel?",
            get_parser=float,
            vals=Numbers(-100, 20),
        )

        self.reference_level_offset = self.add_parameter(
            "reference_level_offset",
            label="Reference level offset",
            unit="dB",
            set_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:RLEVel:OFFSet {}",
            get_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:RLEVel:OFFSet?",
            get_parser=float,
            vals=Numbers(-300, 300),
        )

        self.scale_per_division = self.add_parameter(
            "scale_per_division",
            label="Y scale per division",
            unit="dB",
            set_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:PDIVision {}",
            get_cmd=":DISPlay:WINdow:TRACe:Y:SCALe:PDIVision?",
            get_parser=float,
            vals=Numbers(0.1, 20),
        )

        self.power_unit = self.add_parameter(
            "power_unit",
            label="Y-axis unit",
            set_cmd=":UNIT:POWer {}",
            get_cmd=":UNIT:POWer?",
            vals=Enum("DBM", "DBMV", "DBUV", "V", "W"),
        )

        self.sweep_points = self.add_parameter(
            "sweep_points",
            label="Sweep points",
            set_cmd=":SENSe:SWEep:POINts {}",
            get_cmd=":SENSe:SWEep:POINts?",
            get_parser=int,
            vals=Ints(101, 3001),
        )

        self.sweep_count = self.add_parameter(
            "sweep_count",
            label="Sweeps per single acquisition",
            set_cmd=":SENSe:SWEep:COUNt {}",
            get_cmd=":SENSe:SWEep:COUNt?",
            get_parser=int,
            vals=Ints(1, 9999),
        )

        self.sweep_time = self.add_parameter(
            "sweep_time",
            label="Sweep time",
            unit="s",
            set_cmd=":SENSe:SWEep:TIME {}",
            get_cmd=":SENSe:SWEep:TIME?",
            get_parser=float,
            vals=Numbers(20e-6, 7500),
        )

        self.sweep_time_auto = self.add_parameter(
            "sweep_time_auto",
            label="Auto sweep time",
            set_cmd=":SENSe:SWEep:TIME:AUTO {}",
            get_cmd=":SENSe:SWEep:TIME:AUTO?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.continuous_sweep = self.add_parameter(
            "continuous_sweep",
            label="Continuous sweep",
            set_cmd=":INITiate:CONTinuous {}",
            get_cmd=":INITiate:CONTinuous?",
            val_mapping=_ON_OFF_MAPPING,
        )

        self.trace_average_count = self.add_parameter(
            "trace_average_count",
            label="Trace average count",
            set_cmd=":TRACe:AVERage:COUNt {}",
            get_cmd=":TRACe:AVERage:COUNt?",
            get_parser=int,
            vals=Ints(1, 1000),
        )

        self.trigger_source = self.add_parameter(
            "trigger_source",
            label="Trigger source",
            set_cmd=":TRIGger:SEQuence:SOURce {}",
            get_cmd=":TRIGger:SEQuence:SOURce?",
            vals=Enum("IMM", "VID", "EXT"),
        )

        self.external_trigger_slope = self.add_parameter(
            "external_trigger_slope",
            label="External trigger slope",
            set_cmd=":TRIGger:SEQuence:EXTernal:SLOPe {}",
            get_cmd=":TRIGger:SEQuence:EXTernal:SLOPe?",
            vals=Enum("POS", "NEG"),
        )

        self.video_trigger_level = self.add_parameter(
            "video_trigger_level",
            label="Video trigger level",
            unit="dBm",
            set_cmd=":TRIGger:SEQuence:VIDeo:LEVel {}",
            get_cmd=":TRIGger:SEQuence:VIDeo:LEVel?",
            get_parser=float,
            vals=Numbers(-300, 50),
        )

        self.trace_data_format = self.add_parameter(
            "trace_data_format",
            label="Trace data format",
            set_cmd=":FORMat:TRACe:DATA {}",
            get_cmd=":FORMat:TRACe:DATA?",
            vals=Enum("ASCII", "REAL,32"),
        )

        self.binary_byte_order = self.add_parameter(
            "binary_byte_order",
            label="Binary byte order",
            set_cmd=":FORMat:BORDer {}",
            get_cmd=":FORMat:BORDer?",
            vals=Enum("NORM", "SWAP"),
        )

        self.trace_axis = self.add_parameter(
            "trace_axis",
            label="Trace axis",
            unit="Hz",
            get_cmd=self._get_trace_axis,
            vals=Arrays(shape=(self._point_count,)),
        )

        for trace_index in range(1, 4):
            trace_channel = RigolDSA800TraceChannel(
                self,
                f"trace{trace_index}",
                trace_index,
            )
            self.add_submodule(f"trace{trace_index}", trace_channel)

        tracking_generator = RigolDSA800TrackingGenerator(
            self,
            "tracking_generator",
        )
        self.add_submodule("tracking_generator", tracking_generator)

        for marker_index in range(1, 5):
            marker = RigolDSA800Marker(self, f"marker{marker_index}", marker_index)
            self.add_submodule(f"marker{marker_index}", marker)

        self.connect_message()

    def _point_count(self) -> int:
        """Return the current trace length for dynamic array validators."""
        return int(self.sweep_points())

    def _get_trace_axis(self) -> npt.NDArray[np.float64]:
        """Return the current setpoint axis for trace acquisition."""
        span_hz = float(self.frequency_span())
        points = int(self.sweep_points())

        if span_hz == 0:
            self.trace_axis.label = "Time axis"
            self.trace_axis.unit = "s"
            sweep_time = float(self.sweep_time())
            return np.linspace(0, sweep_time, num=points, dtype=float)

        self.trace_axis.label = "Frequency axis"
        self.trace_axis.unit = "Hz"
        start = float(self.frequency_start())
        stop = float(self.frequency_stop())
        return np.linspace(start, stop, num=points, dtype=float)

    def _query_trace_data(self, trace_index: int) -> npt.NDArray[np.float64]:
        """Query trace data using the instrument's currently selected transfer format."""
        trace_format = str(self.trace_data_format()).upper()
        command = f":TRACe:DATA? TRACE{trace_index}"

        if trace_format == "ASCII":
            return _parse_ascii_trace_data(self.ask_raw(command))

        is_big_endian = str(self.binary_byte_order()).upper() == "NORM"
        data = self.visa_handle.query_binary_values(
            command,
            datatype="f",
            is_big_endian=is_big_endian,
            container=list,
        )
        return np.asarray(data, dtype=float)

    def abort(self) -> None:
        """Abort the current operation."""
        self.write(":ABORt")

    def reset(self) -> None:
        """Reset the analyzer to its default state."""
        self.write("*RST")

    def autoscale(self) -> None:
        """Adjust reference level and scale for an easy-to-read display."""
        self.write(":SENSe:POWer:ASCale")

    def autorange(self) -> None:
        """Adjust amplitude-related parameters within the current span."""
        self.write(":SENSe:POWer:ARANge")

    def autotune(self) -> None:
        """Search the full frequency range and optimize the display."""
        self.write(":SENSe:POWer:ATUNe")

    def clear_all_markers(self) -> None:
        """Disable all active markers."""
        self.write(":CALCulate:MARKer:AOFF")

    def restart_trace_average(self) -> None:
        """Restart the averaging accumulation for averaged traces."""
        self.write(":TRACe:AVERage:RESet")

    def run_continuous(self) -> None:
        """Enable continuous sweeping."""
        self.continuous_sweep("ON")

    def hold(self) -> None:
        """Disable continuous sweeping."""
        self.continuous_sweep("OFF")

    def _recommended_sweep_timeout(self) -> float:
        """Estimate a safe timeout for one initiated acquisition.

        The estimate is based on the instrument-reported sweep time and sweep
        count, with extra headroom for analyzer overhead and command latency.
        """
        sweep_duration = float(self.sweep_time()) * int(self.sweep_count())
        return max(5.0, sweep_duration * 1.5 + 1.0)

    def wait_for_operation_complete(self, timeout: float | None = None) -> None:
        """Block until the current operation is complete.

        Args:
            timeout: Optional timeout in seconds for the wait.
        """
        visa_timeout = self.visa_handle.timeout
        resolved_timeout = timeout
        if resolved_timeout is None and visa_timeout is not None:
            resolved_timeout = max(
                self._recommended_sweep_timeout(),
                float(visa_timeout) / 1000,
            )
        if resolved_timeout is not None:
            self.visa_handle.timeout = int(resolved_timeout * 1000)
        try:
            self.ask("*OPC?")
        finally:
            self.visa_handle.timeout = visa_timeout

    def trigger_sweep(self, timeout: float | None = None) -> None:
        """Initiate a sweep without changing the current sweep mode.

        Args:
            timeout: Optional timeout in seconds for the sweep.
        """
        self.write(":INITiate:IMMediate")
        self.wait_for_operation_complete(timeout=timeout)

    def single_sweep(self, timeout: float | None = None) -> None:
        """Run a single sweep and wait for completion.

        Args:
            timeout: Optional timeout in seconds for the sweep.
        """
        self.hold()
        self.trigger_sweep(timeout=timeout)

    def acquire_trace(
        self,
        trace_index: int = 1,
        timeout: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Run a single sweep and return the requested trace.

        Args:
            trace_index: Trace number to read back.
            timeout: Optional timeout in seconds for the sweep.

        Returns:
            The acquired trace data.
        """
        if trace_index not in (1, 2, 3):
            raise ValueError("trace_index must be 1, 2, or 3")

        self.single_sweep(timeout=timeout)
        return self._query_trace_data(trace_index)
