from typing import assert_type

import pytest

from qcodes_contrib_drivers.drivers.Keysight.Keysight_33502A import Keysight33502A, Keysight33502AOutputChannel


@pytest.fixture(scope='function', name='driver')
def _make_driver():
    """Create a simulated Keysight 33502A instrument for testing."""
    driver = Keysight33502A(
        'Keysight_33502A',
        address='GPIB::1::INSTR',
        pyvisa_sim_file='qcodes_contrib_drivers.sims:Keysight_33502A.yaml',
    )
    yield driver
    driver.close()


def test_init(driver: Keysight33502A):
    idn_dict = driver.IDN()

    assert idn_dict['vendor'] == 'Agilent Technologies'
    assert idn_dict['model'] == '33502A'


def test_channel(driver: Keysight33502A):
    assert driver.num_channels == 2
    assert isinstance(driver.ch1, Keysight33502AOutputChannel)
    assert isinstance(driver.ch2, Keysight33502AOutputChannel)
    assert_type(driver.ch1, Keysight33502AOutputChannel)
    assert_type(driver.ch2, Keysight33502AOutputChannel)


def test_coupling(driver: Keysight33502A):
    assert driver.ch1.coupling() == 'DC'
    driver.ch1.coupling('AC')
    assert driver.ch1.coupling() == 'AC'
    driver.ch1.coupling('DC')


def test_impedance(driver: Keysight33502A):
    assert driver.ch1.impedance() == 1000000
    driver.ch1.impedance(50)
    assert driver.ch1.impedance() == 50
    driver.ch1.impedance(1000000)


def test_path(driver: Keysight33502A):
    assert driver.ch1.path() == 'DIR'
    driver.ch1.path('AMPL')
    assert driver.ch1.path() == 'AMPL'
    driver.ch1.path('DIR')


def test_state(driver: Keysight33502A):
    assert driver.ch1.state() == 'OFF'
    driver.ch1.state('ON')
    assert driver.ch1.state() == 'ON'
    driver.ch1.state('OFF')
