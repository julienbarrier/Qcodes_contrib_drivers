"""QCoDeS driver for the Keysight 33502A isolated amplifier."""
from qcodes import validators as vals
from qcodes.instrument import Instrument, VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.parameters import Parameter


class Keysight33502AOutputChannel(InstrumentChannel):
    """QCoDeS driver for the Keysight 33502A isolated amplifier."""

    def __init__(self, parent: Instrument, name: str, channum: int, **kwargs):
        """Class constructor.

        Args:
            parent: The instrument to which the channel is attached.
            name: The name of the channel.
            channum: The number of the channel in question (1-2).
            **kwargs: Kwargs are forwarded to base class.

        """
        super().__init__(parent, name, **kwargs)

        self.coupling: Parameter = self.add_parameter(
            'coupling',
            label=f'Channel {channum} coupling mode',
            set_cmd=f'INPut{channum}:COUPling {{}}',
            get_cmd=f'INPut{channum}:COUPling?',
            get_parser=str.rstrip,
            vals=vals.Enum('AC', 'DC'),
        )

        self.impedance: Parameter = self.add_parameter(
            'impedance',
            label=f'Channel {channum} impedance',
            set_cmd=f'INPut{channum}:IMPedance {{}}',
            get_cmd=f'INPut{channum}:IMPedance?',
            get_parser=float,
            unit='ohm',
            vals=vals.MultiType(vals.Numbers(50, 1e6), vals.Enum('MIN', 'MAX', 'DEF')),
        )

        self.path: Parameter = self.add_parameter(
            'path',
            label=f'Channel {channum} gain mode',
            set_cmd=f'ROUTe{channum}:PATH {{}}',
            get_cmd=f'ROUTe{channum}:PATH?',
            get_parser=str.rstrip,
            vals=vals.Enum('DIR', 'AMPL'),
        )

        self.state: Parameter = self.add_parameter(
            'state',
            label=f'Channel {channum} state',
            set_cmd=f'OUTPut{channum}:STATe {{}}',
            get_cmd=f'OUTPut{channum}:STATe?',
            get_parser=str.rstrip,
            val_mapping={'ON': 1, 'OFF': 0},
            vals=vals.Enum('ON', 'OFF'),
        )


OutputChannel = Keysight33502AOutputChannel


class Keysight33502A(VisaInstrument):
    """Class for Keysight 33502A isolated amplifier."""

    default_terminator = '\n'

    def __init__(self, name: str, address: str, silent: bool = False, **kwargs):
        """Class constructor.

        Args:
            name: The name of the instrument used internally by QCoDeS. Must be unique.
            address: The VISA resource name.
            silent: If True, no connect message is printed.
            **kwargs: Kwargs are forwarded to base class.

        """
        super().__init__(name, address, **kwargs)

        self.num_channels = 2

        for i in range(1, self.num_channels + 1):
            channel = Keysight33502AOutputChannel(self, f'ch{i}', i)
            self.add_submodule(f'ch{i}', channel)

        # This command resets the amplifier to the factory configuration.
        self.add_function('reset', call_cmd='*RST')

        # This command clears the event registers in all register groups. This command also clears the error queue.
        self.add_function('clear', call_cmd='*CLS')

        if not silent:
            self.connect_message()
