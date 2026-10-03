# -*- coding: utf-8 -*-
"""QCoDeS-Driver for Stanford Research Systems DC215 voltage/current source:
https://www.thinksrs.com/products/dc215.html

Authors:
    Julien Barrier, <julien@julienbarrier.eu>, 2026
"""

import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from qcodes.instrument import VisaInstrument
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes import validators as vals

log = logging.getLogger(__name__)

# Value returned by measurement queries when the measurement is not available
_OVERLOAD = 9.999999e3


def _enum(*labels: str) -> Dict[str, str]:
    """Map labels to the string index used by the instrument (0, 1, ...)."""
    return {label: str(i) for i, label in enumerate(labels)}


class StanfordResearchSystems_DC215(VisaInstrument):
    """
    Class to represent a Stanford Research Systems DC215 voltage/current source

    status: beta-version

    Args:
        name: name for the instrument
        address: Visa resource name to connect
        terminator: Visa terminator
    """

    # Source range codes used by QDAT: code -> (source type, nominal range)
    _sequence_ranges: Dict[int, Tuple[str, float]] = {
        0: ('voltage', 30.0), 1: ('voltage', 10.0), 2: ('voltage', 1.0),
        3: ('voltage', 0.1), 4: ('voltage', 0.01),
        5: ('current', 0.2), 6: ('current', 0.1), 7: ('current', 0.01),
        8: ('current', 0.001)
    }

    _key_codes: Dict[str, int] = {
        'none': 0, '3': 1, 'enter_v_a': 2, '6': 3, 'enter_mv_ma': 4,
        'prgm': 5, 'shift': 6, '9': 7, 'clr': 8, '1': 9, '2': 10, '4': 11,
        '5': 12, 'pause': 13, 'step': 14, '8': 15, '7': 16,
        'range_down': 17, 'range_up': 18, 'setting_down': 19,
        'setting_up': 20, 'run': 21, '+/-': 22, '.': 23, '0': 24,
        'cursor_down': 25, 'cursor_right': 26, 'cursor_up': 27,
        'cursor_left': 28, 'v/i': 29, 'on/off': 30
    }

    _ramp_actions: Dict[str, int] = {
        'cancel_revert': 0,  # cancel and return to the start value
        'cancel_keep': 1,  # cancel and keep the present value
        'cancel_final': 2,  # cancel and jump to the final value
        'pause': 3,
        'resume': 4
    }

    _display_sources: Dict[str, int] = {
        'source': 0, 'monitor': 1, 'sequence': 2, 'freeze': 3
    }

    def __init__(
            self,
            name: str,
            address: str,
            terminator: str = '\r\n',
            **kwargs: Any) -> None:
        super().__init__(name, address=address, terminator=terminator,
                         **kwargs)

        # Output
        self.output = Parameter(
            'output',
            label='Output',
            get_cmd='OUTP?',
            set_cmd='OUTP {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.source_type = Parameter(
            'source_type',
            label='Source type',
            get_cmd='VORI?',
            set_cmd='VORI {}',
            val_mapping=_enum('voltage', 'current'),
            instrument=self
        )

        self.bandwidth = Parameter(
            'bandwidth',
            label='Output bandwidth',
            get_cmd='BAND?',
            set_cmd='BAND {}',
            val_mapping=_enum('high', 'low'),
            instrument=self
        )

        self.guard = Parameter(
            'guard',
            label='Guard',
            get_cmd='GARD?',
            set_cmd='GARD {}',
            val_mapping=_enum('internal', 'external'),
            instrument=self
        )

        self.sense = Parameter(
            'sense',
            label='Voltage sensing',
            get_cmd='SENS?',
            set_cmd='SENS {}',
            val_mapping={'2-wire': '0', '4-wire': '1'},
            instrument=self
        )

        self.ramp_time = Parameter(
            'ramp_time',
            label='Ramp time',
            unit='s',
            get_cmd='RTIM?',
            set_cmd='RTIM {:.1f}',
            get_parser=float,
            vals=vals.Numbers(0, 3600),
            instrument=self
        )

        self.ramp_state = Parameter(
            'ramp_state',
            label='Ramp state',
            get_cmd='RCTL?',
            val_mapping={'not_ramping': '0', 'paused': '1', 'ramping': '2'},
            instrument=self
        )

        self.ready_delay = Parameter(
            'ready_delay',
            label='Ready delay',
            unit='s',
            get_cmd='RDLY?',
            set_cmd='RDLY {:.3f}',
            get_parser=float,
            vals=vals.Numbers(0.002, 100),
            instrument=self
        )

        self.real_time_value = Parameter(
            'real_time_value',
            label='Real time output value (V or A)',
            get_cmd='REAL?',
            get_parser=float,
            instrument=self
        )

        # Source ranges and values
        self.voltage_range = Parameter(
            'voltage_range',
            label='Voltage range',
            unit='V',
            get_cmd='RANG? 0',
            set_cmd='RANG 0,{:e}',
            get_parser=float,
            vals=vals.Numbers(-30, 30),
            instrument=self
        )

        self.current_range = Parameter(
            'current_range',
            label='Current range',
            unit='A',
            get_cmd='RANG? 1',
            set_cmd='RANG 1,{:e}',
            get_parser=float,
            vals=vals.Numbers(-0.2, 0.2),
            instrument=self
        )

        self.voltage = Parameter(
            'voltage',
            label='Source voltage',
            unit='V',
            get_cmd='SORC? 0',
            set_cmd='SORC 0,{:e}',
            get_parser=float,
            vals=vals.Numbers(-30, 30),
            instrument=self
        )

        self.current = Parameter(
            'current',
            label='Source current',
            unit='A',
            get_cmd='SORC? 1',
            set_cmd='SORC 1,{:e}',
            get_parser=float,
            vals=vals.Numbers(-0.2, 0.2),
            instrument=self
        )

        # Limits
        self.voltage_limit_pos = Parameter(
            'voltage_limit_pos',
            label='Positive voltage limit',
            unit='V',
            get_cmd=partial(self._get_limit, 0, 0),
            set_cmd=partial(self._set_limit, 0, 0),
            vals=vals.Numbers(-30, 30),
            instrument=self
        )

        self.voltage_limit_neg = Parameter(
            'voltage_limit_neg',
            label='Negative voltage limit',
            unit='V',
            get_cmd=partial(self._get_limit, 0, 1),
            set_cmd=partial(self._set_limit, 0, 1),
            vals=vals.Numbers(-30, 30),
            instrument=self
        )

        self.current_limit_pos = Parameter(
            'current_limit_pos',
            label='Positive current limit',
            unit='A',
            get_cmd=partial(self._get_limit, 1, 0),
            set_cmd=partial(self._set_limit, 1, 0),
            vals=vals.Numbers(-0.2, 0.2),
            instrument=self
        )

        self.current_limit_neg = Parameter(
            'current_limit_neg',
            label='Negative current limit',
            unit='A',
            get_cmd=partial(self._get_limit, 1, 1),
            set_cmd=partial(self._set_limit, 1, 1),
            vals=vals.Numbers(-0.2, 0.2),
            instrument=self
        )

        # Measurements
        self.integration_time = Parameter(
            'integration_time',
            label='Measurement integration time',
            get_cmd='INTG?',
            set_cmd='INTG {}',
            val_mapping=_enum('very_fast', 'fast', 'medium', 'slow'),
            instrument=self
        )

        self.measurement_enable = Parameter(
            'measurement_enable',
            label='Measurements',
            get_cmd='MEAS?',
            set_cmd='MEAS {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.measured_voltage = Parameter(
            'measured_voltage',
            label='Measured voltage (with REL)',
            unit='V',
            get_cmd='MONO? 0',
            get_parser=self._parse_measurement,
            instrument=self
        )

        self.measured_current = Parameter(
            'measured_current',
            label='Measured current (with REL)',
            unit='A',
            get_cmd='MONO? 1',
            get_parser=self._parse_measurement,
            instrument=self
        )

        self.measured_voltage_raw = Parameter(
            'measured_voltage_raw',
            label='Measured voltage (without REL)',
            unit='V',
            get_cmd='MONO? 2',
            get_parser=self._parse_measurement,
            instrument=self
        )

        self.measured_current_raw = Parameter(
            'measured_current_raw',
            label='Measured current (without REL)',
            unit='A',
            get_cmd='MONO? 3',
            get_parser=self._parse_measurement,
            instrument=self
        )

        self.vrel = Parameter(
            'vrel',
            label='Voltage REL value',
            unit='V',
            get_cmd='MREL? 0',
            set_cmd='MREL 0,{:e}',
            get_parser=float,
            vals=vals.Numbers(-30, 30),
            instrument=self
        )

        self.irel = Parameter(
            'irel',
            label='Current REL value',
            unit='A',
            get_cmd='MREL? 1',
            set_cmd='MREL 1,{:e}',
            get_parser=float,
            vals=vals.Numbers(-0.2, 0.2),
            instrument=self
        )

        self.measurement_timer = Parameter(
            'measurement_timer',
            label='Measurement timer',
            unit='s',
            get_cmd='MTIM?',
            set_cmd='MTIM {:.1f}',
            get_parser=float,
            vals=vals.Numbers(0.2, 3600),
            instrument=self
        )

        self.measurement_trigger_source = Parameter(
            'measurement_trigger_source',
            label='Measurement trigger source',
            get_cmd='MTSO?',
            set_cmd='MTSO {}',
            val_mapping=_enum('auto', 'timer', 'ready', 'interface'),
            instrument=self
        )

        # Digital IO
        self.bnc_input = Parameter(
            'bnc_input',
            label='BNC input source',
            get_cmd='BISO?',
            set_cmd='BISO {}',
            val_mapping=_enum('none', 'output_control', 'step_trigger'),
            instrument=self
        )

        self.bnc_output = Parameter(
            'bnc_output',
            label='BNC output source',
            get_cmd='BOSO?',
            set_cmd='BOSO {}',
            val_mapping=_enum('output_on', 'step_trigger', 'ready',
                              'measure_active', 'step_active'),
            instrument=self
        )

        self.debounce = Parameter(
            'debounce',
            label='Digital input debounce',
            get_cmd='DEBO?',
            set_cmd='DEBO {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.output_control = Parameter(
            'output_control',
            label='Output control',
            get_cmd='OCTL?',
            set_cmd='OCTL {}',
            val_mapping=_enum('none', 'enable', 'control'),
            instrument=self
        )

        # Logging
        self.logging_mode = Parameter(
            'logging_mode',
            label='Logging interface',
            get_cmd='ELOG?',
            set_cmd='ELOG {}',
            val_mapping=_enum('disabled', 'usb', 'remote'),
            instrument=self
        )

        self.log_halt_on_error = Parameter(
            'log_halt_on_error',
            label='Halt on logging error',
            get_cmd='LHOE?',
            set_cmd='LHOE {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.log_points = Parameter(
            'log_points',
            label='Number of points to log (0 = continuous)',
            get_cmd='NMPT?',
            set_cmd='NMPT {:d}',
            get_parser=int,
            vals=vals.Ints(0, 100000),
            instrument=self
        )

        self.usb_log_file_number = Parameter(
            'usb_log_file_number',
            label='USB log file number',
            get_cmd='ULFN?',
            set_cmd='ULFN {:d}',
            get_parser=int,
            vals=vals.Ints(0, 9999),
            instrument=self
        )

        # Sequencer
        self.sequencer_mode = Parameter(
            'sequencer_mode',
            label='Sequencer mode',
            get_cmd='QMOD? t',
            set_cmd='QMOD {}',
            val_mapping={
                'normal': 'NORM',
                'program': 'PRGM',
                'run': 'RUN',
                'pause': 'PAUSE',
                'step_adjust': 'STEPADJ'},
            instrument=self
        )

        self.sequencer_step = Parameter(
            'sequencer_step',
            label='Sequencer step number',
            get_cmd='QLIN?',
            set_cmd='QLIN {:d}',
            get_parser=int,
            vals=vals.Ints(1, 10000),
            instrument=self
        )

        self.sequencer_points_limit = Parameter(
            'sequencer_points_limit',
            label='Sequencer number of points limit (0 = all)',
            get_cmd='QNPT?',
            set_cmd='QNPT {:d}',
            get_parser=int,
            vals=vals.Ints(0, 10000),
            instrument=self
        )

        self.sequencer_repeat = Parameter(
            'sequencer_repeat',
            label='Sequencer repeat',
            get_cmd='QREP?',
            set_cmd='QREP {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.step_timer = Parameter(
            'step_timer',
            label='Step timer (0 = disabled)',
            unit='s',
            get_cmd='QTIM?',
            set_cmd='QTIM {:.1f}',
            get_parser=float,
            vals=vals.MultiType(vals.Enum(0), vals.Numbers(0.1, 3600)),
            instrument=self
        )

        self.sequencer_trigger_mode = Parameter(
            'sequencer_trigger_mode',
            label='Sequencer trigger mode',
            get_cmd='QTRG?',
            set_cmd='QTRG {}',
            val_mapping=_enum('normal', 'measure_end'),
            instrument=self
        )

        self.sequencer_last_step = Parameter(
            'sequencer_last_step',
            label='Number of steps in the program',
            get_cmd='QEND? 0',
            get_parser=int,
            instrument=self
        )

        # Miscellaneous
        self.key_click = Parameter(
            'key_click',
            label='Key click',
            get_cmd='BEEP?',
            set_cmd='BEEP {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.device_name = Parameter(
            'device_name',
            label='Device name',
            get_cmd='NAME?',
            set_cmd='NAME {}',
            vals=vals.Strings(0, 7),
            instrument=self
        )

        self.lock = Parameter(
            'lock',
            label='Interface lock',
            get_cmd='LOCK?',
            set_cmd='LOCK {}',
            val_mapping={
                'unlocked': '0',
                'front_panel_lockout': '1',
                'local_lock': '2'},
            instrument=self
        )

        self.usb_enabled = Parameter(
            'usb_enabled',
            label='USB port',
            get_cmd='EUSB?',
            set_cmd='EUSB {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.temperature = Parameter(
            'temperature',
            label='Instrument temperature',
            unit='°C',
            get_cmd='TEMP?',
            get_parser=float,
            instrument=self
        )

        self.date_time = Parameter(
            'date_time',
            label='Instrument date and time',
            get_cmd='TDAT?',
            instrument=self
        )

        self.mac_address = Parameter(
            'mac_address',
            label='Ethernet MAC address',
            get_cmd='EMAC?',
            get_parser=lambda s: s.strip('"'),
            instrument=self
        )

        # Status registers
        self.esr = Parameter(
            'esr',
            label='Standard event status register (clears on read)',
            get_cmd='*ESR?',
            get_parser=int,
            instrument=self
        )

        self.ese = Parameter(
            'ese',
            label='Standard event status enable register',
            get_cmd='*ESE?',
            set_cmd='*ESE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 255),
            instrument=self
        )

        self.sre = Parameter(
            'sre',
            label='Service request enable register',
            get_cmd='*SRE?',
            set_cmd='*SRE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 255),
            instrument=self
        )

        self.stb = Parameter(
            'stb',
            label='Status byte',
            get_cmd='*STB?',
            get_parser=int,
            instrument=self
        )

        self.psc = Parameter(
            'psc',
            label='Power-on status clear',
            get_cmd='*PSC?',
            set_cmd='*PSC {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.lsee = Parameter(
            'lsee',
            label='Logging and sequencer enable register',
            get_cmd='LSEE?',
            set_cmd='LSEE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 65535),
            instrument=self
        )

        self.lser = Parameter(
            'lser',
            label='Logging and sequencer status register (latched)',
            get_cmd='LSER?',
            get_parser=int,
            instrument=self
        )

        self.lssr = Parameter(
            'lssr',
            label='Logging and sequencer state register (immediate)',
            get_cmd='LSSR?',
            get_parser=int,
            instrument=self
        )

        self.omee = Parameter(
            'omee',
            label='Output and measurement enable register',
            get_cmd='OMEE?',
            set_cmd='OMEE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 65535),
            instrument=self
        )

        self.omer = Parameter(
            'omer',
            label='Output and measurement status register (latched)',
            get_cmd='OMER?',
            get_parser=int,
            instrument=self
        )

        self.omsr = Parameter(
            'omsr',
            label='Output and measurement state register (immediate)',
            get_cmd='OMSR?',
            get_parser=int,
            instrument=self
        )

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        """ Return the Instrument Identifier Message """
        idstr = self.ask('*IDN?')
        idparts = [p.strip() for p in idstr.split(',', 3)]
        idparts += [None] * (4 - len(idparts))  # type: ignore[list-item]
        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    # IEEE-488.2 common commands
    def reset(self) -> None:
        """Reset the DC215 to its default configuration"""
        self.log.info('Reset DC215 to its default configuration.')
        self.write('*RST')

    def clear_status(self) -> None:
        """Clear the ESR, OMER and LSER registers and the error buffer"""
        self.write('*CLS')

    def save_settings(self, location: int) -> None:
        """
        Save the instrument settings.

        Args:
            location: 1-9 for user setups, 10 for the settings recalled at
                start up.
        """
        if location not in range(1, 11):
            raise ValueError('location must be between 1 and 10')
        self.write(f'*SAV {location:d}')

    def recall_settings(self, location: int) -> None:
        """
        Recall the instrument settings.

        Args:
            location: 0 for factory defaults, 1-9 for user setups.
        """
        if location not in range(0, 10):
            raise ValueError('location must be between 0 and 9')
        self.write(f'*RCL {location:d}')

    def trigger(self) -> None:
        """Generate a measurement trigger (if the trigger source is
        'interface')"""
        self.write('*TRG')

    def self_test(self) -> int:
        """
        Run the instrument self test (takes several seconds, turns off the
        output, ramps, sequences and logging).

        Returns:
            0 if successful, 17 if one or more tests failed.
        """
        old_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 60000
        try:
            return int(self.ask('*TST?'))
        finally:
            self.visa_handle.timeout = old_timeout

    def wait(self) -> None:
        """Wait for all pending operations to complete"""
        self.write('*WAI')

    def operation_complete(self) -> bool:
        """Return True when all prior commands have completed"""
        return self.ask('*OPC?').strip() == '1'

    def set_operation_complete_flag(self) -> None:
        """Set the OPC flag of the ESR when all prior commands completed"""
        self.write('*OPC')

    # Interface commands
    def local(self) -> None:
        """Set the instrument to local mode"""
        self.write('LOCL')

    def get_error(self) -> int:
        """Pop the last error from the error buffer (0 = no error)"""
        return int(self.ask('LERR?'))

    def decode_error(self, code: int) -> str:
        """Translate an error code into human readable text"""
        return self.ask(f'EDCD? {int(code):d}').strip('"')

    def get_errors(self) -> List[Tuple[int, str]]:
        """Read and decode all errors in the error buffer"""
        errors = []
        for _ in range(20):
            code = self.get_error()
            if code == 0:
                break
            errors.append((code, self.decode_error(code)))
        return errors

    def get_interface_config(self, parameter: int) -> str:
        """
        Query a remote interface parameter (IFCF).

        Args:
            parameter: 0 RS-232 enable, 1 RS-232 baud rate (0=4800 ... 5=115200),
                2 GPIB enable, 3 GPIB address, 4 TCP/IP enable, 5 DHCP enable,
                6 Auto-IP enable, 7 Static IP enable, 8 Raw socket enable,
                9 Telnet enable, 10 VXI-11 enable, 11 10/100 Base-T,
                12 static IP address, 13 static subnet mask, 14 static gateway.
        """
        if parameter not in range(0, 15):
            raise ValueError('parameter must be between 0 and 14')
        return self.ask(f'IFCF? {parameter:d}')

    def set_interface_config(self, parameter: int, value: Any) -> None:
        """
        Set a remote interface parameter (IFCF); see `get_interface_config`.
        Changes only take effect after an interface reset (`reset_interface`)
        or a power cycle.
        """
        if parameter not in range(0, 15):
            raise ValueError('parameter must be between 0 and 14')
        self.write(f'IFCF {parameter:d},{value}')

    def reset_interface(self, interface: int) -> None:
        """Reset an interface: 0 RS-232, 1 GPIB, 2 TCP/IP"""
        if interface not in range(0, 3):
            raise ValueError('interface must be 0 (RS-232), 1 (GPIB) or '
                             '2 (TCP/IP)')
        self.write(f'IFRS {interface:d}')

    def get_ip_config(self, parameter: int) -> str:
        """Query the active TCP/IP configuration: 0 link, 1 IP address,
        2 subnet mask, 3 gateway"""
        if parameter not in range(0, 4):
            raise ValueError('parameter must be between 0 and 3')
        return self.ask(f'IPCF? {parameter:d}')

    def set_terminator_chars(self, *codes: int) -> None:
        """
        Set the response terminator (XTRM) from up to 3 ASCII codes,
        e.g. ``set_terminator_chars(13, 10)`` for CR LF.
        """
        if not 1 <= len(codes) <= 3:
            raise ValueError('Between 1 and 3 ASCII codes are required')
        self.write('XTRM ' + ','.join(f'{int(c):d}' for c in codes))

    # Output commands
    def ramp_control(self, action: str) -> None:
        """
        Control the output ramp.

        Args:
            action: 'cancel_revert' (back to start value), 'cancel_keep'
                (keep present value), 'cancel_final' (jump to final value),
                'pause' or 'resume'.
        """
        if action not in self._ramp_actions:
            raise ValueError(f'action must be one of '
                             f'{list(self._ramp_actions)}')
        self.write(f'RCTL {self._ramp_actions[action]:d}')

    def _get_limit(self, source: int, side: int) -> float:
        """side: 0 for positive, 1 for negative limit"""
        return float(self.ask(f'LIMI? {source:d}').split(',')[side])

    def _set_limit(self, source: int, side: int, value: float) -> None:
        limits = [float(x) for x in
                  self.ask(f'LIMI? {source:d}').split(',')[:2]]
        limits[side] = value
        self.write(f'LIMI {source:d},{limits[0]:e},{limits[1]:e}')

    @staticmethod
    def _parse_measurement(s: str) -> float:
        value = float(s)
        if value == _OVERLOAD:
            log.warning('Measurement not available (overload value '
                        'returned). Check the OMER/OMSR registers.')
            return float('nan')
        return value

    # Sequencer commands
    def clear_sequence(self) -> None:
        """Erase the sequencer program memory"""
        self.write('QCLR')
        self.operation_complete()

    def step_trigger(self) -> None:
        """Generate a step trigger (sequencer in run mode)"""
        self.write('QSTP')

    def _sequence_range_code(self, source_type: str, range_value: float
                             ) -> int:
        for code, (typ, nominal) in self._sequence_ranges.items():
            if typ == source_type and abs(nominal - abs(range_value)) < 1e-12:
                return code
        raise ValueError(f'Invalid {source_type} range {range_value}')

    def set_sequence_step(self, step: int, source_type: str,
                          range_value: float, value: float) -> None:
        """
        Program a sequencer step (sequencer must be in 'program' mode). The
        step can be at most the last programmed step + 1.

        Args:
            step: Step number (1-10000)
            source_type: 'voltage' or 'current' (same for the whole program)
            range_value: Nominal range (voltage: 30, 10, 1, 0.1, 0.01 V;
                current: 0.2, 0.1, 0.01, 0.001 A)
            value: Source value within the range
        """
        code = self._sequence_range_code(source_type, range_value)
        self.write(f'QDAT {int(step):d},{code:d},{value:e}')

    def get_sequence_step(self, step: int) -> Dict[str, Any]:
        """Read a sequencer step as a dict with source_type, range, value"""
        code_str, value = self.ask(f'QDAT? {int(step):d}').split(',')[:2]
        source_type, range_value = self._sequence_ranges[int(code_str)]
        return {'source_type': source_type, 'range': range_value,
                'value': float(value)}

    def remove_last_sequence_step(self) -> int:
        """Erase the last programmed step (program mode only) and return the
        new number of steps"""
        return int(self.ask('QEND? 1'))

    def get_sequence(self) -> List[Dict[str, Any]]:
        """Read the whole sequencer program"""
        n_steps = self.sequencer_last_step()
        return [self.get_sequence_step(i) for i in range(1, n_steps + 1)]

    def set_sequence(self, steps: List[Tuple[str, float, float]]) -> None:
        """
        Replace the sequencer program.

        Args:
            steps: list of (source_type, range_value, value) tuples, see
                `set_sequence_step`.
        """
        self.sequencer_mode('normal')
        self.clear_sequence()
        self.sequencer_mode('program')
        try:
            for i, (source_type, range_value, value) in enumerate(steps, 1):
                self.set_sequence_step(i, source_type, range_value, value)
        finally:
            self.sequencer_mode('normal')

    def save_sequence_usb(self, number: int) -> None:
        """Save the sequence to PRGM<number>.CSV on the USB drive. Errors are
        reported in the LSER/LSSR registers."""
        if not 0 <= number <= 9999:
            raise ValueError('number must be between 0 and 9999')
        self.write(f'QSUS {number:d}')

    def load_sequence_usb(self, number: int) -> None:
        """Load the sequence from PRGM<number>.CSV on the USB drive. Errors
        are reported in the LSER/LSSR registers."""
        if not 0 <= number <= 9999:
            raise ValueError('number must be between 0 and 9999')
        self.write(f'QSUL {number:d}')

    # Logging commands
    def get_log_record(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the oldest log record (logging mode must be 'remote').

        Returns:
            A dict with time, source_type, range, source_value, voltage and
            current, or None if no record is available.
        """
        record = self.ask('RLOG?').strip()
        if record == '!':
            return None
        t, typ, rng, src, volt, curr = record.split(',')[:6]
        return {'time': float(t), 'source_type': typ, 'range': rng,
                'source_value': float(src), 'voltage': float(volt),
                'current': float(curr)}

    # Miscellaneous commands
    def set_display(self, display: str, setting: int) -> None:
        """
        Set a display setting (DISP).

        Args:
            display: 'source' (left display: 0 source, 1 real time,
                2 +limit, 3 -limit, 4 Tramp, 5 Rdelay), 'monitor'
                (0 voltage, 1 current), 'sequence' (0 monitor, 1 step
                number) or 'freeze' (0 freeze, 1 unfreeze).
            setting: Setting index, see above.
        """
        if display not in self._display_sources:
            raise ValueError(f'display must be one of '
                             f'{list(self._display_sources)}')
        self.write(f'DISP {self._display_sources[display]:d},{int(setting):d}')

    def get_display(self, display: str) -> int:
        """Query a display setting, see `set_display`"""
        if display not in self._display_sources:
            raise ValueError(f'display must be one of '
                             f'{list(self._display_sources)}')
        return int(self.ask(f'DISP? {self._display_sources[display]:d}'))

    def freeze_display(self) -> None:
        """Halt front panel updates"""
        self.set_display('freeze', 0)

    def unfreeze_display(self) -> None:
        """Resume front panel updates"""
        self.set_display('freeze', 1)

    def press_key(self, key: str) -> None:
        """Simulate a key press, see `_key_codes` for the available keys"""
        if key not in self._key_codes:
            raise ValueError(f'key must be one of {list(self._key_codes)}')
        self.write(f'KEYS {self._key_codes[key]:d}')

    def set_date_time(self, year: int, month: int, day: int, hour: int,
                      minute: int) -> None:
        """Set the instrument date and time"""
        for i, value in enumerate((year, month, day, hour, minute)):
            self.write(f'TDAT {i:d},{int(value):d}')
