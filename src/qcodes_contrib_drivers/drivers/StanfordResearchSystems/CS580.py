# -*- coding: utf-8 -*-
"""QCoDeS-Driver for Stanford Research Systems CS580 current source:
https://www.thinksrs.com/products/cs580.html

Authors:
    Julien Barrier, <julien@julienbarrier.eu>
"""

import logging
from typing import Optional, Any, Dict, List, Tuple

from qcodes.instrument import VisaInstrument
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes import validators as vals

log = logging.getLogger(__name__)


class CS580(VisaInstrument):
    """
    Class to represent an SRS CS580 current source

    status: beta-version

    The instrument is forced into numeric token mode (``TOKN 0``) on
    connection, so that all token-type queries (gain, overload, ...) are
    answered with integers. Do not change ``token_mode`` to ``'on'`` while
    using this driver, as the parsers expect numeric answers.

    Args:
        name (str): name for the instrument
        address (str): Visa resource name to connect
        terminator (str): Visa terminator
    """

    _gains = {
        1e-9: 0, 10e-9: 1, 100e-9: 2,
        1e-6: 3, 10e-6: 4, 100e-6: 5,
        1e-3: 6, 10e-3: 7, 50e-3: 8
    }

    _n_to_gains = {g: k for k, g in _gains.items()}

    _overload_status = {
        0: 'None',
        1: 'Compliance limit reached',
        2: 'Analog input overload',
        3: 'Compliance limit reached and analog input overload'
    }

    _execution_errors = {
        0: 'No execution error since last LEXE?',
        1: 'Illegal value',
        2: 'Wrong token',
        3: 'Invalid bit',
        4: 'Queue full',
        5: 'Not compatible'
    }

    _command_errors = {
        0: 'No command error since last LCME?',
        1: 'Illegal command',
        2: 'Undefined command',
        3: 'Illegal query',
        4: 'Illegal set',
        5: 'Missing parameter(s)',
        6: 'Extra parameter(s)',
        7: 'Null parameter(s)',
        8: 'Parameter buffer overflow',
        9: 'Bad floating-point',
        10: 'Bad integer',
        11: 'Bad integer token',
        12: 'Bad token value',
        13: 'Bad hex block',
        14: 'Unknown token'
    }

    def __init__(
            self,
            name: str,
            address: str,
            terminator: str = '\r\n',
            **kwargs: Any) -> None:
        super().__init__(name, address=address, terminator=terminator,
                         **kwargs)

        # Numeric answers to token queries are expected by the parsers below
        self.write('TOKN 0')

        self.gain = Parameter(
            'gain',
            label='Gain',
            unit='A/V',
            get_cmd='GAIN?',
            set_cmd=self._set_gain,
            get_parser=self._parse_get_gain,
            vals=vals.Enum(*self._gains.keys()),
            instrument=self
        )

        self.input = Parameter(
            'input',
            label='Analog input',
            get_cmd='INPT?',
            set_cmd='INPT {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.speed = Parameter(
            'speed',
            label='Speed',
            get_cmd='RESP?',
            set_cmd='RESP {}',
            val_mapping={
                'fast': '0',
                'slow': '1'},
            instrument=self
        )

        self.shield = Parameter(
            'shield',
            label='Inner shield',
            get_cmd='SHLD?',
            set_cmd=self._set_shield,
            val_mapping={
                'guard': '0',
                'return': '1'},
            instrument=self
        )

        self.isolation = Parameter(
            'isolation',
            label='Isolation',
            get_cmd='ISOL?',
            set_cmd=self._set_isolation,
            val_mapping={
                'ground': '0',
                'float': '1'},
            instrument=self
        )

        self.output = Parameter(
            'output',
            label='Output',
            get_cmd='SOUT?',
            set_cmd='SOUT {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.current = Parameter(
            'current',
            label='DC current',
            unit='A',
            get_cmd='CURR?',
            set_cmd=self._set_current,
            get_parser=float,
            vals=vals.Numbers(-100e-3, 100e-3),
            instrument=self
        )

        self.voltage = Parameter(
            name='voltage',
            label='Compliance voltage',
            unit='V',
            get_cmd='VOLT?',
            set_cmd='VOLT {:f}',
            get_parser=float,
            vals=vals.Numbers(min_value=0.0, max_value=50.0),
            instrument=self
        )

        self.alarm = Parameter(
            name='alarm',
            label='Audible alarms',
            get_cmd='ALRM?',
            set_cmd='ALRM {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.token_mode = Parameter(
            name='token_mode',
            label='Token response mode',
            get_cmd='TOKN?',
            set_cmd='TOKN {}',
            val_mapping=create_on_off_val_mapping(on_val='1', off_val='0'),
            instrument=self
        )

        self.overload = Parameter(
            name='overload',
            label='Overload status',
            get_cmd='OVLD?',
            get_parser=self._parse_overload,
            instrument=self
        )

        self.status_byte = Parameter(
            name='status_byte',
            label='Status byte',
            get_cmd='*STB?',
            get_parser=int,
            instrument=self
        )

        self.service_request_enable = Parameter(
            name='service_request_enable',
            label='Service request enable register',
            get_cmd='*SRE?',
            set_cmd='*SRE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 255),
            instrument=self
        )

        self.event_status = Parameter(
            name='event_status',
            label='Standard event status register (cleared on read)',
            get_cmd='*ESR?',
            get_parser=int,
            instrument=self
        )

        self.event_status_enable = Parameter(
            name='event_status_enable',
            label='Standard event status enable register',
            get_cmd='*ESE?',
            set_cmd='*ESE {:d}',
            get_parser=int,
            vals=vals.Ints(0, 255),
            instrument=self
        )

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        """ Return the Instrument Identifier Message """
        idstr = self.ask('*IDN?')
        idparts: List[Optional[str]] = [p.strip() for p in idstr.split(',', 3)]
        idparts += [None] * (4 - len(idparts))
        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def get_overload(self) -> str:
        """ Reads the current value of the signal overload status."""
        self.log.info('Get overload status')
        return self.overload()

    def reset(self) -> None:
        """
        Reset the CS580 to its default configuration (gain 1 mA/V, input on,
        fast response, shield return, floating, output off, compliance
        voltage 10 V, current 0 A, alarms on).
        """
        self.log.info('Reset CS580 to its default configuration.')
        self.write('*RST')
        for parameter in self.parameters.values():
            parameter.cache.invalidate()
        return None

    def clear_status(self) -> None:
        """Clear the standard event status register (*CLS)"""
        self.write('*CLS')

    def operation_complete(self) -> bool:
        """
        Query *OPC?. The instrument answers 1 once all preceding commands
        have been executed, which is useful to pace streams of commands.
        """
        return self.ask('*OPC?').strip() == '1'

    def set_operation_complete(self) -> None:
        """Set the OPC bit of the standard event status register (*OPC)"""
        self.write('*OPC')

    # Status registers, optionally addressed bit by bit
    def get_status_byte(self, bit: Optional[int] = None) -> int:
        """Read the status byte (or only bit ``bit``, 0-7)"""
        return int(self.ask(self._bit_cmd('*STB?', bit)))

    def get_event_status(self, bit: Optional[int] = None) -> int:
        """Read the standard event status register (or only bit ``bit``,
        0-7). The bits that are read are cleared."""
        return int(self.ask(self._bit_cmd('*ESR?', bit)))

    def get_service_request_enable(self, bit: Optional[int] = None) -> int:
        """Read the service request enable register (or only bit ``bit``)"""
        return int(self.ask(self._bit_cmd('*SRE?', bit)))

    def set_service_request_enable(self, value: int,
                                   bit: Optional[int] = None) -> None:
        """Set the service request enable register to ``value`` (0-255) or,
        if ``bit`` is given, set that bit to ``value`` (0 or 1)"""
        self.write(self._bit_cmd('*SRE', bit, value))

    def get_event_status_enable(self, bit: Optional[int] = None) -> int:
        """Read the standard event status enable register (or only bit
        ``bit``)"""
        return int(self.ask(self._bit_cmd('*ESE?', bit)))

    def set_event_status_enable(self, value: int,
                                bit: Optional[int] = None) -> None:
        """Set the standard event status enable register to ``value`` (0-255)
        or, if ``bit`` is given, set that bit to ``value`` (0 or 1)"""
        self.write(self._bit_cmd('*ESE', bit, value))

    @staticmethod
    def _bit_cmd(cmd: str, bit: Optional[int],
                 value: Optional[int] = None) -> str:
        if bit is not None and bit not in range(8):
            raise ValueError('bit must be between 0 and 7')
        if value is None:  # query
            return cmd if bit is None else f'{cmd} {bit:d}'
        if bit is None:
            if value not in range(256):
                raise ValueError('value must be between 0 and 255')
            return f'{cmd} {value:d}'
        if value not in (0, 1):
            raise ValueError('bit value must be 0 or 1')
        return f'{cmd} {bit:d},{value:d}'

    # Errors
    def get_execution_error(self) -> Tuple[int, str]:
        """Read and clear the last execution error (LEXE?)"""
        code = int(self.ask('LEXE?'))
        return code, self._execution_errors.get(code, 'Unknown')

    def get_command_error(self) -> Tuple[int, str]:
        """Read and clear the last command error (LCME?)"""
        code = int(self.ask('LCME?'))
        return code, self._command_errors.get(code, 'Unknown')

    def check_errors(self, raise_exc: bool = True) -> List[str]:
        """
        Read (and clear) the last execution and command errors.

        Args:
            raise_exc: If True, raise a RuntimeError if an error occurred.

        Returns:
            List of error messages (empty if no error).
        """
        errors = []
        code, msg = self.get_execution_error()
        if code != 0:
            errors.append(f'Execution error {code}: {msg}')
        code, msg = self.get_command_error()
        if code != 0:
            errors.append(f'Command error {code}: {msg}')
        if errors and raise_exc:
            raise RuntimeError('; '.join(errors))
        return errors

    # Setters enforcing the interlocks described in the manual
    def _set_gain(self, gain: float) -> None:
        # GAIN may not be set while both the input and the output are on
        if self.input() and self.output():
            raise RuntimeError('The gain cannot be changed while both the '
                               'analog input and the output are enabled.')
        self.write(f'GAIN {self._gains[gain]:d}')

    def _set_shield(self, value: str) -> None:
        if self.output():
            raise RuntimeError('The inner shield cannot be changed while '
                               'the output is enabled.')
        self.write(f'SHLD {value}')

    def _set_isolation(self, value: str) -> None:
        if self.output():
            raise RuntimeError('The isolation cannot be changed while the '
                               'output is enabled.')
        self.write(f'ISOL {value}')

    def _set_current(self, current: float) -> None:
        # The current range is always +/- 2 V times the gain
        limit = 2 * self.gain()
        if abs(current) > limit * (1 + 1e-9):
            raise ValueError(f'Current {current} A is out of range for the '
                             f'present gain: +/-{limit} A')
        self.write(f'CURR {current:e}')

    def _parse_get_gain(self, s: str) -> float:
        return self._n_to_gains[int(s)]

    def _parse_overload(self, s: str) -> str:
        try:
            return self._overload_status[int(s)]
        except (ValueError, KeyError):
            return s
