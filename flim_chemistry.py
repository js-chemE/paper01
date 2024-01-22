import numpy as np


def ph2hplus(ph):
    return 10 ** (-ph)


def ph2poh(ph):
    return 14 - ph


def poh2ohminus(poh):
    return 10 ** (-poh)


def ohminus2poh(ohminus):
    return -np.log10(ohminus)


def ph2ohminus(ph):
    poh = ph2poh(ph)
    return poh2ohminus(poh)


def ohminus2ph(ohminus):
    poh = ohminus2poh(ohminus)
    return 14 - poh
