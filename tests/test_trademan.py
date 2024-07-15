#!/usr/bin/env python

"""Tests for `trademan` package."""


import unittest

from trademan import data
from trademan.portfolio import *
from trademan.trademan.data import get_tickers

import subprocess as subp

class TestTrademan(unittest.TestCase):
    """Tests for `trademan` package."""

    def test_market_dl(self):
        """gets appl"""
        dl = data.get_tickers('AAPL')
        perf = data.data_db[f'perf/AAPL']

    def test_cli(self) :
        os.system("""trademan -cls etfs -gamma 1 -alloc 100000 -in QQQ,SCHG,VGT,SLV,VIG,SPY,VOO,VUG,IAU,PAVE,NANC""")
        os.system("""trademan -cls etfs -gamma 20 -alloc 100000 -in QQQ,SCHG,VGT,SLV,VIG,SPY,VOO,VUG,IAU,PAVE,NANC,KRUZ -cycl-err 100""")