
from matplotlib.colors import Normalize
import matplotlib.cbook as cbook
import numpy as np


class ArbitraryNorm(Normalize):
    """
    Normalization allowing the definition of any non linear
    function for different ranges of the colorbar.
    >>> norm=ArbitraryNorm(fpos=(lambda x: x**0.5),
    >>>                    fposinv=(lambda x: x**2),
    >>>                    fneg=(lambda x: x**0.25),
    >>>                    fneginv=(lambda x: x**4))
    """

    def __init__(self, flist,
                 finvlist=None,
                 refpoints_cm=[None], refpoints_data=[None],
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

        if vmin is not None and vmax is not None:
            if vmin >= vmax:
                raise ValueError("vmin must be less than vmax")

        if finvlist == None:
            finvlist = [None] * len(flist)

        if len(flist) != len(finvlist):
            raise ValueError("len(flist) must be equal than len(finvlist)")

        if len(refpoints_cm) != len(flist) - 1:
            raise ValueError(
                "len(refpoints_cm) must be equal than len(flist) -1")

        if len(refpoints_data) != len(refpoints_cm):
            raise ValueError(
                "len(refpoints_data) must be equal than len(refpoints_cm)")

        self._refpoints_cm = np.concatenate(
            ([0.0], np.array(refpoints_cm), [1.0]))
        if any(np.diff(self._refpoints_cm) <= 0):
            raise ValueError(
                "refpoints_cm values must be monotonically increasing and within the (0.0,1.0) interval")

        self._refpoints_data = np.concatenate(
            ([None], np.array(refpoints_data), [None]))

        if len(self._refpoints_data[1:-1]) > 2 and any(np.diff(self._refpoints_data[1:-1]) <= 0):
            raise ValueError(
                "refpoints_data values must be monotonically increasing")

        # Parsing the function strings if any:
        self._flist = []
        self._finvlist = []
        for i in range(len(flist)):
            funs = ArbitraryNorm._fun_parser((flist[i], finvlist[i]))
            if funs[0] is None or funs[1] is None:
                raise ValueError(
                    "Inverse function not provided for %i range" % i)

            self._flist.append(ArbitraryNorm._fun_normalizer(funs[0]))
            self._finvlist.append(ArbitraryNorm._fun_normalizer(funs[1]))

        super(ArbitraryNorm, self).__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        vmin = self.vmin
        vmax = self.vmax
        self._refpoints_data[0] = vmin
        self._refpoints_data[-1] = vmax
        rp_d = self._refpoints_data
        rp_cm = self._refpoints_cm
        if len(rp_d[1:-1]) > 0 and (any(rp_d[1:-1]) <= vmin or any(rp_d[1:-1]) >= vmax):
            raise ValueError(
                "data reference points must be within the (vmin,vmax) interval")

        widths_cm = np.diff(rp_cm)
        widths_d = np.diff(rp_d)
        resultnorm = result.copy() * 0

        result[result >= vmax] = vmax
        result[result <= vmin] = vmin

        for i in range(len(widths_cm)):
            width_cm = widths_cm[i]
            width_d = widths_d[i]
            mask = (result >= rp_d[i]) * (result <= rp_d[i + 1])
            resultnorm[mask] = self._flist[i](
                (result[mask] - rp_d[i]) / width_d) * width_cm + rp_cm[i]
        self.autoscale_None(resultnorm)
        return resultnorm

    def inverse(self, value):

        vmin = self.vmin
        vmax = self.vmax
        self._refpoints_data[0] = vmin
        self._refpoints_data[-1] = vmax
        rp_d = self._refpoints_data
        rp_cm = self._refpoints_cm
        widths_cm = np.diff(rp_cm)
        widths_d = np.diff(rp_d)

        if cbook.iterable(value):
            value_aux = value.copy() * 0
            for i in range(len(widths_cm)):
                width_cm = widths_cm[i]
                width_d = widths_d[i]
                mask = (value >= rp_cm[i]) * (value <= rp_cm[i + 1])
                value_aux[mask] = self._finvlist[i](
                    (value[mask] - rp_cm[i]) / width_cm) * width_d + rp_d[i]
            value = value_aux
        else:
            for i in range(len(widths_cm)):
                width_cm = widths_cm[i]
                width_d = widths_d[i]
                if (value >= rp_cm[i]) and (value <= rp_cm[i + 1]):
                    value = self._finvlist[i](
                        (result[mask] - rp_cm[i]) / width_cm) * width_d + rp_d[i]
        return value

    @staticmethod
    def _fun_parser(funsin):
        flog = 2000
        funs = [('linear', (lambda x: x), (lambda x: x)),
                ('quadratic', (lambda x: x**2), (lambda x: x**(1. / 2))),
                ('cubic', (lambda x: x**3), (lambda x: x**(1. / 3))),
                ('sqrt', (lambda x: x**(1. / 2)), (lambda x: x**2)),
                ('crt', (lambda x: x**(1. / 3)), (lambda x: x**3)),
                ('log', (lambda x: np.log10(x * flog + 1) / np.log10(flog + 1)),
                 (lambda x: (10**(np.log10(flog + 1) * x) - 1) / flog))]

        if isinstance(funsin[0], basestring):
            funstrs = []
            for fstr, fun, inv in funs:
                funstrs.append(fstr)
                if funsin[0] == fstr:
                    return fun, inv
            raise ValueError(
                "the only strings recognized as functions are %s" % funstrs)
        else:
            return funsin

    @staticmethod
    def _fun_normalizer(fun):
        if fun(0.) == 0. and fun(1.) == 1.:
            return fun
        elif fun(0.) == 0.:
            return (lambda x: fun(x) / fun(1.))
        else:
            return (lambda x: (fun(x) - fun(0.)) / (fun(1.) - fun(0.)))

    def ticks(self, N=None):

        rp_cm = self._refpoints_cm
        widths_cm = np.diff(rp_cm)

        if N is None:
            N = max([13, len(rp_cm)])

        if N < len(rp_cm):
            ValueError(
                "the number of ticks must me larger that the number or intervals +1")

        ticks = rp_cm.copy()

        available_ticks = N - len(-rp_cm)
        distribution = widths_cm * (available_ticks) / widths_cm.sum()
        nticks = np.floor(distribution)

        while(nticks.sum() < available_ticks):
            ind = np.argmax((distribution - nticks))
            nticks[ind] += 1

        for i in range(len(nticks)):
            if nticks[i] > 0:
                N = nticks[i]
                auxticks = np.linspace(rp_cm[i], rp_cm[i + 1], N + 2)
                ticks = np.concatenate([ticks, auxticks[1:-1]])
        return self.inverse(np.sort(ticks))

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


class MirrorArbitraryNorm(ArbitraryNorm):
    """
    Normalization allowing the definition of any arbitrary non linear
    function for the colorbar, for both, the positive, and the negative
    directions.
    >>> norm=ArbitraryNorm(fpos=(lambda x: x**0.5),
    >>>                    fposinv=(lambda x: x**2),
    >>>                    fneg=(lambda x: x**0.25),
    >>>                    fneginv=(lambda x: x**4))
    """

    def __init__(self,
                 fpos, fposinv=None,
                 fneg=None, fneginv=None,
                 center_cm=.5, center_data=0.0,
                 vmin=None, vmax=None, clip=False):

        if fneg is None and fneginv is not None:
            raise ValueError("fneginv not expected without fneg")

        if fneg is None:
            fneg = fpos
            fneginv = fposinv

        fpos, fposinv = ArbitraryNorm._fun_parser([fpos, fposinv])
        fneg, fneginv = ArbitraryNorm._fun_parser([fneg, fneginv])

        if fposinv is None:
            raise ValueError(
                "Inverse function must be provided for the positive interval")
        if fneginv is None:
            raise ValueError(
                "Inverse function must be provided for the negative interval")

        if vmin is not None and vmax is not None:
            if vmin >= vmax:
                raise ValueError("vmin must be less than vmax")

        if center_cm <= 0.0 or center_cm >= 1.0:
            raise ValueError("center must be within the (0.0,1.0) interval")

        refpoints_cm = np.array([center_cm])
        refpoints_data = np.array([center_data])

        flist = [(lambda x:(-fneg(-x + 1) + 1)), fpos]
        finvlist = [(lambda x:(-fneginv(-x + 1) + 1)), fposinv]

        super(MirrorArbitraryNorm, self).__init__(flist=flist,
                                                  finvlist=finvlist,
                                                  refpoints_cm=refpoints_cm,
                                                  refpoints_data=refpoints_data,
                                                  vmin=vmin, vmax=vmax, clip=clip)


class SingleArbitraryNorm(ArbitraryNorm):
    """
    Normalization allowing the definition of any arbitrary non linear
    function for the colorbar, for both, the positive, and the negative
    directions.
    >>> norm=ArbitraryNorm(fpos=(lambda x: x**0.5),
    >>>                    fposinv=(lambda x: x**2),
    >>>                    fneg=(lambda x: x**0.25),
    >>>                    fneginv=(lambda x: x**4))
    """

    def __init__(self, f, finv=None,
                 vmin=None, vmax=None, clip=False):

        fp, finv = ArbitraryNorm._fun_parser([f, finv])

        if finv is None:
            raise ValueError("Inverse function not provided")

        if vmin is not None and vmax is not None:
            if vmin >= vmax:
                raise ValueError("vmin must be less than vmax")

        refpoints_cm = np.array([])
        refpoints_data = np.array([])

        flist = [f]
        finvlist = [finv]

        super(SingleArbitraryNorm, self).__init__(flist=flist,
                                                  finvlist=finvlist,
                                                  refpoints_cm=refpoints_cm,
                                                  refpoints_data=refpoints_data,
                                                  vmin=vmin, vmax=vmax, clip=clip)


class MirrorRootNorm(MirrorArbitraryNorm):
    """
    Root normalization for positive and negative data.
    >>> norm=PositiveRootNorm(orderneg=3,orderpos=7)
    """

    def __init__(self, orderpos=2, orderneg=None,
                 center_cm=0.5, center_data=0.0,
                 vmin=None, vmax=None, clip=False,
                 ):
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
        super(MirrorRootNorm, self).__init__(fneg=(lambda x: x**(1. / orderneg)),
                                             fneginv=(lambda x: x**(orderneg)),
                                             fpos=(lambda x: x **
                                                   (1. / orderpos)),
                                             fposinv=(lambda x: x**(orderpos)),
                                             center_cm=center_cm, center_data=center_data,
                                             vmin=vmin, vmax=vmax, clip=clip)


class RootNorm(SingleArbitraryNorm):
    """
    Root normalization for positive data.
    >>> norm=PositiveRootNorm(vmin=0,orderpos=7)
    """

    def __init__(self, order=2, vmin=None, vmax=None, clip=False):
        """
        *order*:
        Degree of the root used to normalize the data for the positive
        direction.
        """
        super(RootNorm, self).__init__(f=(lambda x: x**(1. / order)),
                                       finv=(lambda x: x**(order)),
                                       vmin=vmin, vmax=vmax, clip=clip)
