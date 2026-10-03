import logging
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from qcodes.instrument import InstrumentChannel
from qcodes.parameters import (
    ParamRawDataType, Parameter, ParameterWithSetpoints
)
from qcodes.validators import Arrays, ComplexNumbers, Enum, MultiType, Numbers, Strings
from zhinst.qcodes import MFLI as mfli

log = logging.getLogger(__name__)


class ComplexSampleParameter(Parameter):
    """
    This defines a Complex Sample Parameter for use in the MFLI class.
    """
    def __init__(
        self,
        *args: Any,
        dict_parameter: Optional[Parameter] = None,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        if dict_parameter is None:
            raise TypeError("ComplexSampleParameter requires a dict_parameter")
        self._dict_parameter = dict_parameter

    def get_raw(self) -> ParamRawDataType:
        values_dict = self._dict_parameter.get()
        x = values_dict["x"]
        y = values_dict["y"]
        if hasattr(x, "__len__"):
            return complex(x[0], y[0])
        return complex(x,y)

class DAQTrace(ParameterWithSetpoints):
    """
    A trace acquired with the LabOne DAQ module. Reading it starts (or
    serves) an acquisition, see `MFLIDAQ`.
    """

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.instrument, MFLIDAQ)
        return self.instrument._serve(self.name)


class MFLIDAQ(InstrumentChannel):
    """
    Wrapper around the LabOne DAQ module of an MFLI. One acquisition (one
    long gapless trace, made of ``count`` bursts of ``burst_duration``
    seconds) serves all the signals registered with `add_signal`: reading
    the traces one after the other gives one acquisition, reading the same
    trace twice acquires twice.

    Args:
        parent: the MFLI instrument
        name: name of the channel
        daq_module: optional callable returning the (configured by this
            class) LabOne DAQ module; defaults to
            ``parent.session.modules.daq``. Mostly useful for tests.
        clockbase: clock rate [Hz] used for the timestamps, defaults to
            the value read from the instrument (60 MHz if not available).
    """

    def __init__(
        self,
        parent: Any,
        name: str,
        daq_module: Optional[Callable[[], Any]] = None,
        clockbase: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(parent, name, **kwargs)
        self._daq_module = daq_module
        self._clockbase = clockbase
        self._signals: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self._cache: Optional[Dict[str, np.ndarray]] = None
        self._served: Set[str] = set()

        self.duration = Parameter(
            'duration',
            label='Total acquisition duration',
            unit='s',
            get_cmd=None,
            set_cmd=None,
            vals=Numbers(min_value=1e-3),
            initial_value=10.0,
            instrument=self
        )

        self.sampling_rate = Parameter(
            'sampling_rate',
            label='Sampling rate',
            unit='Hz',
            get_cmd=None,
            set_cmd=None,
            vals=Numbers(min_value=1e-3),
            initial_value=100.0,
            instrument=self
        )

        self.burst_duration = Parameter(
            'burst_duration',
            label='Burst duration (None = min(duration, 10 s))',
            unit='s',
            get_cmd=None,
            set_cmd=None,
            vals=MultiType(Enum(None), Numbers(min_value=1e-3)),
            initial_value=None,
            instrument=self
        )

        self.save_directory = Parameter(
            'save_directory',
            label='LabOne save directory (empty = no file)',
            get_cmd=None,
            set_cmd=None,
            vals=Strings(),
            initial_value='',
            instrument=self
        )

        self.n_points = Parameter(
            'n_points',
            label='Number of points of a trace',
            get_cmd=self._get_n_points,
            snapshot_value=False,
            instrument=self
        )

        self.time = Parameter(
            'time',
            label='Time',
            unit='s',
            get_cmd=self._get_time,
            vals=Arrays(shape=(self._get_n_points,)),
            snapshot_value=False,
            instrument=self
        )

    # Acquisition settings
    def _get_n_points(self) -> int:
        return int(round(self.sampling_rate() * self.duration()))

    def _get_time(self) -> np.ndarray:
        return np.arange(self._get_n_points()) / self.sampling_rate()

    def _burst(self) -> float:
        burst = self.burst_duration()
        if burst is None:
            burst = min(self.duration(), 10.0)
        return float(min(burst, self.duration()))

    def _cols(self) -> int:
        return max(1, int(round(self.sampling_rate() * self._burst())))

    def _count(self) -> int:
        return max(1, int(math.ceil(self.duration() / self._burst() - 1e-9)))

    # Signals
    def _demod_node(self, demod: int) -> str:
        sample = self.parent.demods[demod - 1].sample
        try:
            path = sample.zi_node
        except AttributeError:
            path = sample.node_info.path
        return str(path)

    def add_signal(
        self,
        name: str,
        demod: Optional[int] = None,
        field: str = 'y',
        node: Optional[str] = None,
        unit: str = '',
        label: str = ''
    ) -> DAQTrace:
        """
        Add a trace that is acquired by the DAQ module.

        Args:
            name: name of the trace parameter
            demod: demodulator number (starting at 1); the node is then
                ``demods/<demod - 1>/sample.<field>``
            field: field of the demodulator sample ('x', 'y', 'r', 'theta',
                'frequency', ...)
            node: explicit node path, instead of ``demod``/``field``
            unit: unit of the trace
            label: label of the trace
        """
        if (demod is None) == (node is None):
            raise ValueError('Give either demod or node')
        if node is None:
            assert demod is not None
            node = f'{self._demod_node(demod)}.{field}'
        trace = DAQTrace(
            name,
            label=label or name,
            unit=unit,
            setpoints=(self.time,),
            vals=Arrays(shape=(self._get_n_points,)),
            snapshot_get=False,
            snapshot_value=False,
            instrument=self
        )
        self.parameters[name] = trace
        self._signals[name] = node.lower()
        return trace

    # Acquisition
    def abort_acquisition(self) -> None:
        """Stop a running acquisition. The trace that is being acquired is
        returned with the data read so far, padded with NaN."""
        self._abort.set()

    def acquire(self) -> Dict[str, np.ndarray]:
        """Acquire all registered signals and return {name: trace}."""
        with self._lock:
            return self._acquire_locked()

    def _serve(self, name: str) -> np.ndarray:
        with self._lock:
            if self._cache is None or name in self._served:
                self._cache = self._acquire_locked()
                self._served = set()
            self._served.add(name)
            return self._cache[name]

    def _get_clockbase(self) -> float:
        if self._clockbase is not None:
            return self._clockbase
        try:
            return float(self.parent.system.clockbase())
        except Exception:
            return 60e6

    def _module(self) -> Any:
        if self._daq_module is not None:
            return self._daq_module()
        module = self.parent.session.modules.daq
        module.device(self.parent)
        return module

    def _configure(self, module: Any) -> None:
        module.type(0)  # continuous
        module.grid.mode(2)
        module.grid.rows(1)
        module.count(self._count())
        module.duration(self._burst())
        module.grid.cols(self._cols())
        directory = self.save_directory()
        if directory:
            module.save.directory(directory)
            module.save.saveonread(1)

    def _acquire_locked(self) -> Dict[str, np.ndarray]:
        if not self._signals:
            raise RuntimeError('No signal registered, use add_signal()')
        self._abort.clear()
        self._cache = None
        module = self._module()
        clockbase = self._get_clockbase()
        duration = self.duration()
        poll = max(self._burst() / 10, 1e-3)
        deadline = time.monotonic() + 1.5 * duration + 10.0
        bursts: Dict[str, List[Any]] = {n: [] for n in self._signals}
        nodes = {node: name for name, node in self._signals.items()}

        self._configure(module)
        for node in nodes:
            module.subscribe(node)
        module.execute()
        try:
            while True:
                time.sleep(poll)
                self._collect(module, clockbase, nodes, bursts)
                if module.finished():
                    self._collect(module, clockbase, nodes, bursts)
                    break
                if self._abort.is_set():
                    self._collect(module, clockbase, nodes, bursts)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'DAQ acquisition not finished after '
                        f'{1.5 * duration + 10.0:.1f} s')
        finally:
            try:
                module.finish()
            finally:
                for node in nodes:
                    try:
                        module.unsubscribe(node)
                    except Exception:
                        log.exception('Could not unsubscribe %s', node)
        return {name: self._assemble(name, bursts[name], clockbase)
                for name in self._signals}

    @staticmethod
    def _collect(module: Any, clockbase: float, nodes: Dict[str, str],
                 bursts: Dict[str, List[Any]]) -> None:
        result = module.read(raw=False, clk_rate=clockbase)
        for key, data in result.items():
            name = nodes.get(str(key).lower())
            if name is not None:
                bursts[name].extend(data)

    def _assemble(self, name: str, bursts: List[Any],
                  clockbase: float) -> np.ndarray:
        n_points = self._get_n_points()
        if bursts:
            self._check_timestamps(name, bursts, clockbase)
            data = np.concatenate(
                [np.ravel(np.asarray(b.value, dtype=float)) for b in bursts])
        else:
            data = np.empty(0)
        if len(data) < n_points:
            if not self._abort.is_set():
                raise RuntimeError(
                    f'Signal {name}: received {len(data)} points, expected '
                    f'{n_points}')
            data = np.concatenate([data, np.full(n_points - len(data),
                                                 np.nan)])
        return data[:n_points]

    def _check_timestamps(self, name: str, bursts: List[Any],
                          clockbase: float) -> None:
        stamps = [np.ravel(b.header['createdtimestamp'])[0] / clockbase
                  for b in bursts]
        expected = self._cols() / self.sampling_rate()
        tolerance = max(2 / self.sampling_rate(), 1e-3 * expected)
        for i, delta in enumerate(np.diff(stamps), 1):
            if abs(delta - expected) > tolerance:
                raise RuntimeError(
                    f'Signal {name}: gap between bursts {i - 1} and {i} '
                    f'({delta:.6f} s instead of {expected:.6f} s)')


class MFLI(mfli):
    """
    This wrapper adds back a "complex sample" parameter to the demodulators
    such that we can use them in the way that it was done with "sample"
    parameter in version 0.2 of ZHINST-qcodes
    written by jenshnielsen: https://github.com/zhinst/zhinst-qcodes/issues/41

    It also adds a ``daq`` submodule (`MFLIDAQ`) that acquires long gapless
    traces with the LabOne DAQ module, as ParameterWithSetpoints.

    Args:
        name: name of the instrument
        serial: serial number of the device
        host: host of the LabOne data server
        daq_signals: optional {trace name: {"demod": 1, "field": "y",
            "unit": "V"}} dictionary, passed to ``daq.add_signal`` (the
            dictionary can also contain "node" and "label").
    """

    def __init__(self, name: str, serial: str, host: str,
                 daq_signals: Optional[Dict[str, Dict[str, Any]]] = None,
                 **kwargs: Any):
        super().__init__(
            name=name, serial=serial, host=host, **kwargs
        )
        self.add_submodule("daq", MFLIDAQ(self, "daq"))
        for signal_name, options in (daq_signals or {}).items():
            self.daq.add_signal(signal_name, **options)
        for demod in self.demods:
            demod.add_parameter(
                "complex_sample",
                label="Vrms",
                vals=ComplexNumbers(),
                parameter_class=ComplexSampleParameter,
                dict_parameter=demod.sample,
                snapshot_value=False,
            )
