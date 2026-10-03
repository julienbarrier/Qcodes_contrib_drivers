from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import numpy.typing as npt

from qcodes.instrument import InstrumentChannel, VisaInstrument
from qcodes.parameters import Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays, Enum, Ints, Numbers

from .Keithley_2000_Scan import Keithley_2000_Scan_Channel

if TYPE_CHECKING:
    pass


class _Keithley6500TimeTrace(ParameterWithSetpoints):
    """1-D timetrace parameter that lives on a Keithley6500Buffer.

    Calling get() on this parameter performs the full acquisition sequence:
    abort any in-progress trigger model, clear the buffer, issue an internal
    software trigger (blocking until the buffer is full), then read back the
    data.  No separate trigger_start() call is needed.
    """

    def get_raw(self) -> npt.NDArray:
        buf: Keithley6500Buffer = cast("Keithley6500Buffer", self.instrument)
        dmm: Keithley_6500 = cast("Keithley_6500", buf.parent)

        # Ensure the digitize trigger fires immediately (no external stimulus)
        dmm.digitize_trigger("NONE")

        # Abort any in-progress acquisition, then clear stale data
        buf.clear_buffer()  # calls abort() internally

        # :MEASure:DIGitize? is a *blocking query* for digitize mode: it
        # fills the buffer with digi_sense.count() readings and only returns
        # once acquisition is complete.  :TRACe:TRIGger is non-blocking in
        # digitize mode and must not be used here.
        dmm.ask(f":MEASure:DIGitize? '{buf.short_name}'")

        n_actual = buf.number_of_readings()
        buf.data_end(n_actual)

        # Keep the unit metadata in sync with the active function
        func = dmm.digi_sense_function.get_latest() or dmm.digi_sense_function()
        if func != "None":
            unit = Keithley6500DigitizeSense.function_modes[func]["unit"]
            buf.timetrace.unit = unit

        start = buf.data_start()
        raw = dmm.ask(
            f":TRACe:DATA? {start}, {n_actual}, '{buf.short_name}'"
        )
        return np.array([float(v) for v in raw.split(",")])


class _Keithley6500TimeAxis(Parameter):
    """Setpoints (time axis) derived from acq_rate and sample count."""

    def get_raw(self) -> npt.NDArray:
        buf: Keithley6500Buffer = cast("Keithley6500Buffer", self.instrument)
        n = buf.data_end() - buf.data_start() + 1
        # acq_rate is on the active digitize-sense channel
        dmm: Keithley_6500 = cast("Keithley_6500", buf.parent)
        rate = dmm.digi_sense.acq_rate()
        return np.linspace(0, (n - 1) / rate, n)


class Keithley6500Buffer(InstrumentChannel):
    """
    Reading buffer submodule for the Keithley DMM6500.

    Buffers store measurement data and provide a timetrace parameter for
    retrieving digitized waveforms as a NumPy array with time setpoints.
    """

    default_buffer: ClassVar[set[str]] = {"defbuffer1", "defbuffer2"}

    def __init__(
        self,
        parent: "Keithley_6500",
        name: str,
        size: int | None = None,
        style: str = "",
    ) -> None:
        super().__init__(parent, name)

        if name not in self.default_buffer:
            if style:
                self.write(f":TRACe:MAKE '{name}', {size}, {style}")
            elif size is not None:
                self.write(f":TRACe:MAKE '{name}', {size}")
        elif size is not None:
            self.log.warning(
                f"Use size() to resize default buffer '{name}' to {size}."
            )

        self.size: Parameter = self.add_parameter(
            "size",
            get_cmd=f":TRACe:POINts? '{name}'",
            set_cmd=f":TRACe:POINts {{}}, '{name}'",
            get_parser=int,
            docstring="Number of readings the buffer can store.",
        )

        self.number_of_readings: Parameter = self.add_parameter(
            "number_of_readings",
            get_cmd=f":TRACe:ACTual? '{name}'",
            get_parser=int,
            docstring="Number of readings currently stored in the buffer.",
        )

        self.data_start: Parameter = self.add_parameter(
            "data_start",
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
            vals=Ints(1),
            docstring="First (1-based) index of data to retrieve.",
        )

        self.data_end: Parameter = self.add_parameter(
            "data_end",
            initial_value=1,
            get_cmd=None,
            set_cmd=None,
            vals=Ints(1),
            docstring="Last (1-based) index of data to retrieve.",
        )

        self.t_start: Parameter = self.add_parameter(
            "t_start",
            get_cmd=f":TRACe:ACTual:STARt? '{name}'",
            get_parser=float,
            unit="s",
            docstring="Timestamp of the first reading in the buffer.",
        )

        self.t_stop: Parameter = self.add_parameter(
            "t_stop",
            get_cmd=f":TRACe:ACTual:END? '{name}'",
            get_parser=float,
            unit="s",
            docstring="Timestamp of the last reading in the buffer.",
        )

        self.time_axis: _Keithley6500TimeAxis = self.add_parameter(
            "time_axis",
            parameter_class=_Keithley6500TimeAxis,
            unit="s",
            label="Time",
            vals=Arrays(shape=(self._get_n_pts,)),
            snapshot_value=False,
        )

        self.timetrace: _Keithley6500TimeTrace = self.add_parameter(
            "timetrace",
            parameter_class=_Keithley6500TimeTrace,
            setpoints=(self.time_axis,),
            unit="",  # set correctly by Keithley6500DigitizeSense._update_timetrace_unit
            label="Digitized signal",
            vals=Arrays(shape=(self._get_n_pts,)),
            snapshot_value=False,
        )

    def _get_n_pts(self) -> int:
        return self.data_end() - self.data_start() + 1

    def clear_buffer(self) -> None:
        """Abort any active trigger model, then clear all readings from the buffer.

        Sending :TRACe:CLEar while a trigger model or :TRACe:TRIGger acquisition
        is still running will cause the instrument to hold the VISA bus until the
        acquisition completes, making the write() call appear to hang.  Aborting
        first guarantees the instrument is idle before the clear is sent.
        """
        self.parent.abort()
        self.write(f":TRACe:CLEar '{self.short_name}'")

    def trigger_start(self) -> None:
        """Trigger acquisition into this buffer using the active measure function."""
        self.write(f":TRACe:TRIGger '{self.short_name}'")

    def delete(self) -> None:
        """Delete this buffer (no-op for default buffers)."""
        if self.short_name not in self.default_buffer:
            self.parent.submodules.pop(f"_buffer_{self.short_name}")
            self.parent.buffer_name("defbuffer1")
            self.write(f":TRACe:DELete '{self.short_name}'")


class Keithley6500DigitizeSense(InstrumentChannel):
    """
    Digitize-sense submodule for the Keithley DMM6500.

    Exposes the :SENSe:DIGitize namespace which enables high-speed
    time-domain (timetrace) measurements at up to 100 kS/s for voltage.
    """

    function_modes: ClassVar[dict[str, dict]] = {
        "voltage": {
            "name": '"VOLT"',
            "unit": "V",
            "range_vals": Numbers(0.1, 1000),
            "rate_vals": Ints(1000, 100_000),
        },
        "current": {
            "name": '"CURR"',
            "unit": "A",
            "range_vals": Numbers(10e-6, 10),
            "rate_vals": Ints(1000, 10_000),
        },
    }

    def __init__(
        self,
        parent: "Keithley_6500",
        name: str,
        proper_function: str,
    ) -> None:
        super().__init__(parent, name)
        self._proper_function = proper_function
        mode = self.function_modes[proper_function]
        unit: str = mode["unit"]
        range_vals: Numbers = mode["range_vals"]
        rate_vals: Ints = mode["rate_vals"]

        self.range: Parameter = self.add_parameter(
            "range",
            get_cmd=f":SENSe:DIGitize:{proper_function}:RANGe?",
            set_cmd=f":SENSe:DIGitize:{proper_function}:RANGe {{}}",
            vals=range_vals,
            get_parser=float,
            unit=unit,
            docstring="Full-scale input range for the digitize function.",
        )

        self.acq_rate: Parameter = self.add_parameter(
            "acq_rate",
            get_cmd=f":SENSe:DIGitize:{proper_function}:SRATE?",
            set_cmd=f":SENSe:DIGitize:{proper_function}:SRATE {{}}",
            vals=rate_vals,
            get_parser=int,
            unit="S/s",
            docstring="Sample rate for digitized measurements.",
        )

        self.aperture: Parameter = self.add_parameter(
            "aperture",
            get_cmd=f":SENSe:DIGitize:{proper_function}:APERture?",
            set_cmd=f":SENSe:DIGitize:{proper_function}:APERture {{}}",
            get_parser=float,
            unit="us",
            docstring="Acquisition aperture (integration window) in microseconds.",
        )

        self.count: Parameter = self.add_parameter(
            "count",
            get_cmd="SENSe:DIGitize:COUNt?",
            set_cmd="SENSe:DIGitize:COUNt {}",
            vals=Ints(1, 55_000_000),
            get_parser=int,
            docstring="Number of samples to acquire per trigger.",
        )

    def _measure(self) -> float:
        buffer_name = self.parent.buffer_name()
        return float(self.ask(f":MEASure:DIGitize? '{buffer_name}'"))


class Keithley_Sense(InstrumentChannel):
    """
    This is the class for a measurement channel, i.e. the quantity to be measured (e.g. resistance, voltage).
    """
    def __init__(self, parent: VisaInstrument, name: str, channel: str) -> None:
        """

        Args:
            parent: VisaInstrument instance of the Keithley Digital Multimeter
            name: Channel name (e.g. 'CH1')
            channel: Name of the quantity to measure (e.g. 'VOLT' for DC voltage measurement)
        """
        valid_channels = ['VOLT', 'CURR', 'RES', 'FRES', 'TEMP']
        if channel.upper() not in valid_channels:
            raise ValueError(f"Channel must be one of the following: {', '.join(valid_channels)}")
        super().__init__(parent, name)

        self.add_parameter('measure',
                           unit=self._get_unit(channel),
                           label=self._get_label(channel),
                           get_parser=float,
                           get_cmd=partial(self.parent._measure, channel),
                           docstring="Measure value of chosen quantity (Current/Voltage/Resistance/Temperature)."
                           )

        self.add_parameter('nplc',
                           label='NPLC',
                           get_parser=float,
                           get_cmd=f"SENS:{channel}:NPLC?",
                           set_cmd=f"SENS:{channel}:NPLC {{:.4f}}",
                           vals=Numbers(0.0005, 12),
                           docstring="Integration rate (Number of Power Line Cycles)"
                           )

    @staticmethod
    def _get_unit(quantity: str) -> str:
        """

        Args:
            quantity: Quantity to be measured

        Returns: Corresponding unit string

        """
        channel_units = {'VOLT': 'V', 'CURR': 'A', 'RES': 'Ohm', 'FRES': 'Ohm', 'TEMP': 'C'}
        return channel_units[quantity]

    @staticmethod
    def _get_label(quantity: str) -> str:
        """

        Args:
            quantity: Quantity to be measured

        Returns: Corresponding parameter label

        """
        channel_labels = {'VOLT': 'Measured voltage.',
                          'CURR': 'Measured current.',
                          'RES': 'Measured resistance',
                          'FRES': 'Measured resistance (4w)',
                          'TEMP': 'Measured temperature'}
        return channel_labels[quantity]


class Keithley_6500(VisaInstrument):
    """
    This is the qcodes driver for a Keithley DMM6500 digital multimeter.
    """
    def __init__(self, name: str,
                 address: str,
                 terminator="\n",
                 **kwargs):
        """
        Initialize instance of digital multimeter Keithley6500. Check if scanner card is inserted.
        Args:
            name: Name of instrument
            address: Address of instrument
            terminator: Termination character for SCPI commands
            **kwargs: Keyword arguments to pass to __init__ function of VisaInstrument class
        """
        super().__init__(name, address, terminator=terminator, **kwargs)
        for quantity in ['VOLT', 'CURR', 'RES', 'FRES', 'TEMP']:
            channel = Keithley_Sense(self, quantity.lower(), quantity)
            self.add_submodule(quantity.lower(), channel)

        self.add_parameter(
            "buffer_name",
            get_cmd=None,
            set_cmd=None,
            docstring="Name of the reading buffer currently in use.",
        )

        self.add_parameter(
            "digi_sense_function",
            set_cmd=":SENSe:DIGitize:FUNCtion {}",
            get_cmd=":SENSe:DIGitize:FUNCtion?",
            val_mapping={
                "voltage": '"VOLT"',
                "current": '"CURR"',
                "None": '"NONE"',
            },
            docstring="Active digitize-sense function.",
        )

        self.add_parameter(
            "digitize_trigger",
            get_cmd=":TRIGger:DIGitize:STIMulus?",
            set_cmd=":TRIGger:DIGitize:STIMulus {}",
            vals=Enum("NONE", "EXT"),
            docstring=(
                "Stimulus that triggers a digitize measurement. "
                "Use 'NONE' for an immediate internal (software) trigger, "
                "or 'EXT' to wait for a signal on the external trigger input."
            ),
        )

        self.add_parameter('active_terminal',
                           label='active terminal',
                           get_cmd="ROUTe:TERMinals?",
                           docstring="Active terminal of instrument. Can only be switched via knob on front panel.")

        self.add_parameter('resistance',
                           unit='Ohm',
                           label='Measured resistance',
                           get_parser=float,
                           get_cmd=partial(self._measure, 'RES'),
                           )

        self.add_parameter('resistance_4w',
                           unit='Ohm',
                           label='Measured resistance',
                           get_parser=float,
                           get_cmd=partial(self._measure, 'FRES')
                           )

        self.add_parameter('voltage_dc',
                           unit='V',
                           label='Measured DC voltage',
                           get_parser=float,
                           get_cmd=partial(self._measure, 'VOLT')
                           )

        self.add_parameter('current_dc',
                           unit='A',
                           label='Measured DC current',
                           get_parser=float,
                           get_cmd=partial(self._measure, 'CURR')
                           )

        self.add_parameter('temperature',
                           unit='C',
                           label='Measured temperature',
                           get_parser=float,
                           get_cmd=partial(self._measure, 'TEMP')
                           )

        self.connect_message()

        # Register digitize-sense submodules for voltage and current
        for func in Keithley6500DigitizeSense.function_modes:
            self.add_submodule(
                f"_digi_sense_{func}",
                Keithley6500DigitizeSense(self, "digi_sense", func),
            )

        # Set up default buffer
        self.buffer_name("defbuffer1")
        self.buffer(name=self.buffer_name())

        # check if scanner card is connected
        # If no scanner card is connected, the query below returns "Empty Slot".
        # For the Scanner Card 2000-SCAN used for development of this driver the output was
        # "2000,10-Chan Mux,0.0.0a,00000000".
        scan_idn_msg = self.ask(":SYSTem:CARD1:IDN?")
        if scan_idn_msg != "Empty Slot":
            msg_parts = scan_idn_msg.split(",")
            print(f"Scanner card {msg_parts[0]}-SCAN detected.")
            for ch_number in range(1, 11):
                scan_channel = Keithley_2000_Scan_Channel(self, ch_number)
                self.add_submodule(f"ch{ch_number:d}", scan_channel)

    # only measure if front terminal is active
    def _measure(self, quantity: str) -> str:
        """
        Measure given quantity at front terminal of the instrument. Only perform measurement if front terminal is
        active. Send SCPI command to measure and read out given quantity.
        Args:
            quantity: Quantity to be measured

        Returns: Measurement result

        """
        if self.active_terminal.get() == 'FRON':
            return self.ask(f"MEAS:{quantity}?")
        else:
            raise RuntimeError("Rear terminal is active instead of front terminal.")

    def buffer(
        self,
        name: str,
        size: int | None = None,
        style: str = "",
    ) -> Keithley6500Buffer:
        """
        Return the buffer submodule for *name*, creating it if necessary.

        Args:
            name: Buffer name (e.g. ``"defbuffer1"`` or a custom name).
            size: Number of readings the buffer can store (ignored for
                default buffers if the buffer already exists).
            style: Buffer style string passed to ``:TRACe:MAKE`` (e.g.
                ``"FULL"``).  Ignored for default and existing buffers.

        Returns:
            The :class:`Keithley6500Buffer` submodule.
        """
        self.buffer_name(name)
        key = f"_buffer_{name}"
        if key in self.submodules:
            return cast("Keithley6500Buffer", self.submodules[key])
        new_buffer = Keithley6500Buffer(parent=self, name=name, size=size, style=style)
        self.add_submodule(key, new_buffer)
        return new_buffer

    @property
    def digi_sense(self) -> Keithley6500DigitizeSense:
        """
        Return the active :class:`Keithley6500DigitizeSense` submodule
        (selected by :attr:`digi_sense_function`).
        """
        func = self.digi_sense_function.get_latest() or self.digi_sense_function()
        if func == "None":
            raise RuntimeError(
                "No digitize function selected. "
                "Set digi_sense_function to 'voltage' or 'current' first."
            )
        submodule = self.submodules[f"_digi_sense_{func}"]
        return cast("Keithley6500DigitizeSense", submodule)

    def abort(self) -> None:
        r"""Stop any running trigger model or active acquisition (\*ABORt).

        Call this before :meth:`Keithley6500Buffer.clear_buffer` or any other
        command that would conflict with an in-progress acquisition.
        """
        self.write(":ABORt")

    def initiate(self) -> None:
        """Start the trigger model."""
        self.write(":INITiate")

    def wait(self) -> None:
        """Wait for all pending operations to complete"""
        self.write("*WAI")
