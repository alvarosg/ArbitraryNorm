
from matplotlib.colors import Normalize
import matplotlib.cbook as cbook
import numpy as np


class ArbitraryNorm(Normalize):
    """
    Normalization allowing the definition of any arbitrary non linear
    function for the colorbar, for both, the positive, and the negative
    directions.
    >>> norm=ArbitraryNorm(fpos=(lambda x: x**0.5),
                           fposinv=(lambda x: x**2),
                           fneg=(lambda x: x**0.25),
                           fneginv=(lambda x: x**4))
    """

    def __init__(self, fpos=(lambda x: x**.5),
                 fposinv=(lambda x: x**2),
                 fneg=None, fneginv=None,
                 center=.5,
                 vmin=None, vmax=None, clip=False):
        """
        *fpos*:
        Non-linear function used to normalize the positive range. Must be
        able to operate take floating point numbers and numpy arrays.
        *fposinv*:
        Inverse function of *fpos*.
        *fneg*:
        Non-linear function used to normalize the negative range. It
        not need to take in to account the negative sign. Must be
        able to operate take floating point numbers and numpy arrays.
        *fneginv*:
        Inverse function of *fneg*.
        *center*: Value between 0. and 1. indicating the color in the colorbar
        that will be assigned to the zero value.
        """

        if fneg is None:
            fneg = fpos
        if fneginv is None:
            fneginv = fposinv
        if vmin is not None and vmax is not None:
            if vmin > vmax:
                raise ValueError("vmin must be less than vmax")
        self.fneg = fneg
        self.fpos = fpos
        self.fneginv = fneginv
        self.fposinv = fposinv
        self.center = center
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        vmin = self.vmin
        vmax = self.vmax

        widthpos = 1 - self.center
        widthneg = self.center

        result[result > vmax] = vmax
        result[result < vmin] = vmin

        maskpositive = result > 0
        masknegative = result < 0
        if vmax > 0 and vmin < 0:
            result[masknegative] = - \
                self.fneg(result[masknegative] / vmin) * widthneg
            result[maskpositive] = self.fpos(
                result[maskpositive] / vmax) * widthpos

        elif vmax > 0 and vmin >= 0:
            result[maskpositive] = self.fpos(
                (result[maskpositive] - vmin) / (vmax - vmin)) * widthpos

        elif vmax <= 0 and vmin < 0:
            result[masknegative] = - \
                self.fneg((result[maskpositive] - vmax) /
                          (vmin - vmax)) * widthneg

        result = result + widthneg

        self.autoscale_None(result)
        return result

    def inverse(self, value):

        vmin = self.vmin
        vmax = self.vmax
        widthpos = 1 - self.center
        widthneg = self.center

        value = value - widthneg

        if cbook.iterable(value):

            maskpositive = value > 0
            masknegative = value < 0

            if vmax > 0 and vmin < 0:
                value[masknegative] = \
                    self.fneginv(-value[masknegative] / widthneg) * vmin
                value[maskpositive] = self.fposinv(
                    value[maskpositive] / widthpos) * vmax

            elif vmax > 0 and vmin >= 0:
                value[maskpositive] = self.fposinv(
                    value[maskpositive] / widthpos) * (vmax - vmin) + vmin
                value[masknegative] = -self.fposinv(
                    value[masknegative] / widthneg) * (vmax - vmin) + vmin
            elif vmax <= 0 and vmin < 0:
                value[masknegative] = self.fneginv(
                    -value[masknegative] / widthneg) * (vmin - vmax) + vmax

        else:

            if vmax > 0 and vmin < 0:
                if value < 0:
                    value = self.fneginv(-value / widthneg) * vmin
                else:
                    value = self.fposinv(value / widthpos) * vmax

            elif vmax > 0 and vmin >= 0:
                if value > 0:
                    value = self.fposinv(value / widthpos) * \
                        (vmax - vmin) + vmin

            elif vmax <= 0 and vmin < 0:
                if value < 0:
                    value = self.fneginv(-value / widthneg) * \
                        (vmin - vmax) + vmax
        return value

    def ticks(self, N=13):
        return self.inverse(np.linspace(0, 1, N))

    def autoscale(self, A):
        """
        vmin = self.vmin
        vmax = self.vmax

        if vmax==0 or np.ma.max(A)==0:
            self.vmin = np.ma.min(A)
            self.vmax = -self.vmin
        elif vmin==0 or np.ma.min(A)==0:
            self.vmax = np.ma.max(A)
            self.vmin = -self.vmax
        else:
            self.vmin = np.ma.min(A)
            self.vmax = np.ma.max(A)
        """
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        if self.vmin is not None and self.vmax is not None:
            return
        if self.vmin is None:
            self.vmin = np.ma.min(A)
        if self.vmax is None:
            self.vmax = np.ma.max(A)


class PositiveArbitraryNorm(Normalize):
    """
    Normalization allowing the definition of any arbitrary non linear
    function for the colorbar for positive data.
    >>> norm=PositiveArbitraryNorm(fpos=(lambda x: x**0.5),
                                   fposinv=(lambda x: x**2))
    """

    def __init__(self, fpos=(lambda x: x**0.5),
                 fposinv=(lambda x: x**2),
                 vmin=None, vmax=None, clip=False):
        """
        *fpos*:
        Non-linear function used to normalize the positive range. Must be
        able to operate take floating point numbers and numpy arrays.
        *fposinv*:
        Inverse function of *fpos*.
        """

        if vmin is not None and vmax is not None:
            if vmin > vmax:
                raise ValueError("vmin must be less than vmax")
        self.fpos = fpos
        self.fposinv = fposinv
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        vmin = self.vmin
        vmax = self.vmax

        result[result > vmax] = vmax
        result[result < vmin] = vmin

        result = self.fpos((result - vmin) / (vmax - vmin))

        self.autoscale_None(result)
        return result

    def inverse(self, value):

        vmin = self.vmin
        vmax = self.vmax

        if cbook.iterable(value):
            value = self.fposinv(value) * (vmax - vmin) + vmin
        else:
            value = self.fposinv(value) * (vmax - vmin) + vmin
        return value

    def ticks(self, N=11):
        return self.inverse(np.linspace(0, 1, N))

    def autoscale(self, A):
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        if self.vmin is not None and self.vmax is not None:
            return
        if self.vmin is None:
            self.vmin = np.ma.min(A)
        if self.vmax is None:
            self.vmax = np.ma.max(A)


class NegativeArbitraryNorm(Normalize):
    """
    Normalization allowing the definition of any arbitrary non linear
    function for the colorbar for negative data.
    >>> norm=NegativeArbitraryNorm(fneg=(lambda x: x**0.5),
                                   fneginv=(lambda x: x**2))
    """

    def __init__(self, fneg=(lambda x: x**0.5), fneginv=(lambda x: x**2),
                 vmin=None, vmax=None, clip=False):
        """
        *fneg*:
        Non-linear function used to normalize the negative range. It
        not need to take in to account the negative sign. Must be
        able to operate take floating point numbers and numpy arrays.
        *fneginv*:
        Inverse function of *fneg*.
        """

        if vmin is not None and vmax is not None:
            if vmin > vmax:
                raise ValueError("vmin must be less than vmax")
        self.fneg = fneg
        self.fneginv = fneginv
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        vmin = self.vmin
        vmax = self.vmax

        result[result > vmax] = vmax
        result[result < vmin] = vmin

        result = -self.fneg((result - vmax) / (vmin - vmax))
        result = result + 1

        self.autoscale_None(result)
        return result

    def inverse(self, value):

        vmin = self.vmin
        vmax = self.vmax

        value = value - 1

        if cbook.iterable(value):
            value = self.fneginv(-value) * (vmin - vmax) + vmax
        else:
            value = self.fneginv(value) * (vmin - vmax) + vmax

        return value

    def ticks(self, N=11):
        return self.inverse(np.linspace(0, 1, N))

    def autoscale(self, A):
        self.vmin = np.ma.min(A)
        self.vmax = np.ma.max(A)

    def autoscale_None(self, A):
        if self.vmin is not None and self.vmax is not None:
            return
        if self.vmin is None:
            self.vmin = np.ma.min(A)
        if self.vmax is None:
            self.vmax = np.ma.max(A)


class SymRootNorm(ArbitraryNorm):
    """
    Root normalization for positive and negative data.
    >>> norm=PositiveRootNorm(orderneg=3,orderpos=7)
    """
    def __init__(self, orderpos=2, orderneg=None,
                 vmin=None, vmax=None, clip=False, center=0.5):
        """
        *orderpos*:
        Degree of the root used to normalize the data for the positive
        direction.
        *orderneg*:
        Degree of the root used to normalize the data for the negative
        direction. By default equal to *orderpos*.
        """

        if orderneg is None:
            orderneg = orderpos
        ArbitraryNorm.__init__(self,
                               fneg=(lambda x: x**(1. / orderneg)),
                               fneginv=(lambda x: x**(orderneg)),
                               fpos=(lambda x: x**(1. / orderpos)),
                               fposinv=(lambda x: x**(orderpos)),
                               center=center,
                               vmin=vmin, vmax=vmax, clip=clip)


class PositiveRootNorm(PositiveArbitraryNorm):
    """
    Root normalization for positive data.
    >>> norm=PositiveRootNorm(vmin=0,orderpos=7)
    """
    def __init__(self, orderpos=2, vmin=None, vmax=None, clip=False):
        """
        *orderpos*:
        Degree of the root used to normalize the data for the positive
        direction.
        """
        PositiveArbitraryNorm.__init__(self,
                                       fpos=(lambda x: x**(1. / orderpos)),
                                       fposinv=(lambda x: x**(orderpos)),
                                       vmin=vmin, vmax=vmax, clip=clip)


class NegativeRootNorm(NegativeArbitraryNorm):
    """
    Root normalization for negative data.
    >>> norm=NegativeRootNorm(vmax=0,orderneg=2)
    """
    def __init__(self, orderneg=2, vmin=None, vmax=None, clip=False):
        """
        *orderneg*:
        Degree of the root used to normalize the data for the negative
        direction.
        """
        NegativeArbitraryNorm.__init__(self,
                                       fneg=(lambda x: x**(1. / orderneg)),
                                       fneginv=(lambda x: x**(orderneg)),
                                       vmin=vmin, vmax=vmax, clip=clip)
