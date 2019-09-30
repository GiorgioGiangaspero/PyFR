# -*- coding: utf-8 -*-

import numpy as np

from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.util import proxylist


class BaseDualPseudoStepper(BaseDualPseudoIntegrator):
    def collect_stats(self, stats):
        super().collect_stats(stats)

        # Total number of RHS evaluations
        stats.set('solver-time-integrator', 'nfevals', self._stepper_nfevals)

        # Total number of pseudo-steps
        stats.set('solver-time-integrator', 'npseudosteps', self.npseudosteps)

    def _rhs_with_dts(self, t, uin, fout):
        # Compute -∇·f
        self.system.rhs(t, uin, fout)

        # Coefficients for the physical stepper
        svals = [sc/self._dt for sc in self._stepper_coeffs]

        # Physical stepper source addition -∇·f - dQ/dt
        axnpby = self._get_axnpby_kerns(len(svals) + 1, subdims=self._subdims)
        self._prepare_reg_banks(fout, self._idxcurr, *self._stepper_regidx)
        self._queue % axnpby(1, *svals)


class DualEulerPseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'euler'

    @property
    def _pseudo_stepper_has_lerrest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 2

    @property
    def _pseudo_stepper_order(self):
        return 1

    def step(self, t):
        add = self._add
        rhs = self._rhs_with_dts

        r0, r1 = self._pseudo_stepper_regidx

        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        rhs(t, r0, r1)
        add(0, r1, 1, r0, self._dtau, r1)

        return r1, r0


class DualTVDRK3PseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'tvd-rk3'

    @property
    def _pseudo_stepper_has_lerrest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 3*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 3

    def step(self, t):
        add = self._add
        rhs = self._rhs_with_dts
        dtau = self._dtau

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._pseudo_stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage;
        # r2 = -∇·f(r0) - dQ/dt; r1 = r0 + dtau*r2
        rhs(t, r0, r2)
        add(0, r1, 1, r0, dtau, r2)

        # Second stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 3/4*r0 + 1/4*r1 + 1/4*dtau*r2
        rhs(t, r1, r2)
        add(1/4, r1, 3/4, r0, dtau/4, r2)

        # Third stage;
        # r2 = -∇·f(r1) - dQ/dt; r1 = 1/3*r0 + 2/3*r1 + 2/3*dtau*r2
        rhs(t, r1, r2)
        add(2/3, r1, 1/3, r0, 2*dtau/3, r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualRK4PseudoStepper(BaseDualPseudoStepper):
    pseudo_stepper_name = 'rk4'

    @property
    def _pseudo_stepper_has_lerrest(self):
        return False

    @property
    def _stepper_nfevals(self):
        return 4*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 3

    @property
    def _pseudo_stepper_order(self):
        return 4

    def step(self, t):
        add = self._add
        rhs = self._rhs_with_dts
        dtau = self._dtau

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time
        r0, r1, r2 = self._pseudo_stepper_regidx

        # Ensure r0 references the bank containing u(n+1,m)
        if r0 != self._idxcurr:
            r0, r1 = r1, r0

        # First stage; r1 = -∇·f(r0) - dQ/dt;
        rhs(t, r0, r1)

        # Second stage; r2 = r0 + dtau/2*r1; r2 = -∇·f(r2) - dQ/dt;
        add(0, r2, 1, r0, dtau/2, r1)
        rhs(t, r2, r2)

        # As no subsequent stages depend on the first stage we can
        # reuse its register to start accumulating the solution with
        # r1 = r0 + dtau/6*r1 + dtau/3*r2
        add(dtau/6, r1, 1, r0, dtau/3, r2)

        # Third stage; here we reuse the r2 register
        # r2 = r0 + dtau/2*r2 - dtau/2*dQ/dt
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau/2, r2, 1, r0)
        rhs(t, r2, r2)

        # Accumulate; r1 = r1 + dtau/3*r2
        add(1, r1, dtau/3, r2)

        # Fourth stage; again we reuse r2
        # r2 = r0 + dtau*r2
        # r2 = -∇·f(r2) - dQ/dt;
        add(dtau, r2, 1, r0)
        rhs(t, r2, r2)

        # Final accumulation r1 = r1 + dtau/6*r2 = u(n+1,m+1)
        add(1, r1, dtau/6, r2)

        # Return the index of the bank containing u(n+1,m+1)
        return r1, r0


class DualEmbeddedPairPseudoStepper(BaseDualPseudoStepper):
    # Coefficients
    a = []
    b = []
    bhat = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute the error coeffs
        self.e = [b - bh for b, bh in zip(self.b, self.bhat)]

        self._nstages = len(self.b)

        # Register a kernel to multiply rhs with local pseudo time-step
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.localdtau'
        )

        tplargs = dict(ndims=self.system.ndims, nvars=self.system.nvars)

        self.dtau_upts = proxylist([])
        for ele, shape in zip(self.system.ele_map.values(),
                              self.system.ele_shapes):
            # Allocate storage for the local pseudo time-step
            dtaumat = self.backend.matrix(shape, np.ones(shape)*self._dtau,
                                          tags={'align'})
            self.dtau_upts.append(dtaumat)

            # Append the local dtau kernels to the proxylist
            self.pintgkernels['localdtau'].append(
                self.backend.kernel(
                    'localdtau', tplargs=tplargs, dims=[ele.nupts, ele.neles],
                    negdivconf=ele.scal_upts_inb, dtau_upts=dtaumat
                )
            )

    def localdtau(self, uinbank, inv=0):
        self.system.eles_scal_upts_inb.active = uinbank
        self._queue % self.pintgkernels['localdtau'](inv=inv)

    @property
    def _pseudo_stepper_has_lerrest(self):
        return self._pseudo_controller_needs_lerrest and self.bhat


class DualRKVdH2RPseudoStepper(DualEmbeddedPairPseudoStepper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Compute the c vector
        self.c = [0.0] + [sum(self.b[:i]) + ai for i, ai in enumerate(self.a)]

    @property
    def _stepper_nfevals(self):
        return len(self.b)*self.nsteps

    @property
    def _pseudo_stepper_nregs(self):
        return 4 if self._pseudo_stepper_has_lerrest else 3

    def step(self, t):
        add, rhs = self._add, self._rhs_with_dts
        errest = self._pseudo_stepper_has_lerrest

        rold = self._idxcurr

        if errest:
            r2, r1, rerr = set(self._pseudo_stepper_regidx) - {rold}
        else:
            r2, r1 = set(self._pseudo_stepper_regidx) - {rold}

        # Copy the current solution
        add(0.0, r1, 1.0, rold)

        # Evaluate the stages in the scheme
        for i in range(self._nstages):
            # Compute -∇·f
            rhs(t, r2 if i > 0 else r1, r2)

            self.localdtau(r2)

            if errest:
                # Accumulate the error term in rerr
                add(1.0 if i > 0 else 0.0, rerr, self.e[i], r2)

            # Sum (special-casing the final stage)
            if i < self._nstages - 1:
                add(1.0, r1, self.a[i], r2)
                add(self.b[i] - self.a[i], r2, 1.0, r1)
            else:
                add(1.0, r1, self.b[i], r2)

            # Swap
            r1, r2 = r2, r1

        # Return
        return (r2, rold, rerr) if errest else (r2, rold)


class DualRK34PseudoStepper(DualRKVdH2RPseudoStepper):
    pseudo_stepper_name = 'rk34'

    a = [
        11847461282814 / 36547543011857,
        3943225443063 / 7078155732230,
        -346793006927 / 4029903576067
    ]

    b = [
        1017324711453 / 9774461848756,
        8237718856693 / 13685301971492,
        57731312506979 / 19404895981398,
        -101169746363290 / 37734290219643
    ]

    bhat = [
        15763415370699 / 46270243929542,
        514528521746 / 5659431552419,
        27030193851939 / 9429696342944,
        -69544964788955 / 30262026368149
    ]

    @property
    def _pseudo_stepper_order(self):
        return 3


class DualRK45PseudoStepper(DualRKVdH2RPseudoStepper):
    pseudo_stepper_name = 'rk45'

    a = [
        970286171893 / 4311952581923,
        6584761158862 / 12103376702013,
        2251764453980 / 15575788980749,
        26877169314380 / 34165994151039
    ]

    b = [
        1153189308089 / 22510343858157,
        1772645290293 / 4653164025191,
        -1672844663538 / 4480602732383,
        2114624349019 / 3568978502595,
        5198255086312 / 14908931495163
    ]

    bhat = [
        1016888040809 / 7410784769900,
        11231460423587 / 58533540763752,
        -1563879915014 / 6823010717585,
        606302364029 / 971179775848,
        1097981568119 / 3980877426909
    ]

    @property
    def _pseudo_stepper_order(self):
        return 4

class DualPseudoRK1011Stepper(DualEmbeddedPairPseudoStepper):
    pseudo_stepper_name = 'rk1011'
    bhat = [0.393532508039953, 0.103698834129218, 1.762063014496891, -7.975545473944710, 17.369144216329801, -23.392811504681202, 20.502287838369053, -11.411634409951532, 3.649264977212526, 0.0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _pseudo_stepper_nregs(self):
        return 5

    @property
    def _pseudo_stepper_order(self):
        return 1

    def step(self, t):
        add = self._add
        rhs = self._rhs_with_dts
        errest = self._pseudo_stepper_has_lerrest

        Adiag = np.array([0, 1.0/9.0, 0.032204258058884271953559164103353, 0.052242410573493235081965480048893,
                          0.076088154842620117634943710527295, 0.10518779980582534938626793064032, 0.14190034937115256208350899669313,
                          0.19033481524088430303365271356597, 0.25842115960389150375320355124131, 0.36419395849639307760625683840772])

        C = np.array([0.0, 1./9, 2./9, 3./9, 4./9, 5./9, 6./9, 7./9, 8./9, 1.0])

        Bhat = np.array([0.393532508039953, 0.103698834129218, 1.762063014496891, -7.975545473944710, 17.369144216329801,
                         -23.392811504681202, 20.502287838369053, -11.411634409951532, 3.649264977212526, 0.0])

        kappa = 0.501223692278266
        B = np.array([(1. - kappa), 0., 0., 0., 0., 0., 0., 0., 0., kappa])
        E = [b - bh for b, bh in zip(B, Bhat)]

        # Get the bank indices for pseudo-registers (n+1,m; n+1,m+1; rhs),
        # where m = pseudo-time and n = real-time

        r0, r1, r2, r3, rerr = set(self._pseudo_stepper_regidx)

        if r0 != self._idxcurr:
            r0, r3 = r3, r0

        # First stage; r1 = -∇·f(r0)
        rhs(t, r0, r1)
        self.localdtau(r1)
        add(0.0, r3, 1.0, r0, B[0], r1)
        if errest:
            add(0.0, rerr, E[0], r1)

        # Second stage;
        add(0, r2, 1, r0, Adiag[1], r1)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[1], r2)

        # Third stage;
        add(0, r2, 1, r0, (C[2] - Adiag[2]), r1, Adiag[2], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[2], r2)

        # Fourth stage;
        add(0, r2, 1, r0, (C[3] - Adiag[3]), r1,  Adiag[3], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[3], r2)

        # Fifth stage;
        add(0, r2, 1, r0, (C[4] - Adiag[4]), r1, Adiag[4], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[4], r2)

        # Sixth stage;
        add(0, r2, 1, r0, (C[5] - Adiag[5]), r1, Adiag[5], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[5], r2)

        # Seventh stage;
        add(0, r2, 1, r0, (C[6] - Adiag[6]), r1, Adiag[6], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[6], r2)

        # Eighth stage;
        add(0, r2, 1, r0, (C[7] - Adiag[7]), r1, Adiag[7], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[7], r2)

        # Ninth stage;
        add(0, r2, 1, r0, (C[8] - Adiag[8]), r1, Adiag[8], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)

        if errest:
            add(1.0, rerr, E[8], r2)

        # Tenth stage;
        add(0, r2, 1, r0, (C[9] - Adiag[9]), r1, Adiag[9], r2)
        rhs(t, r2, r2)
        self.localdtau(r2)
        add(1.0, r3, B[9], r2)
        if errest:
            add(1.0, rerr, E[9], r2)

        # Return the index of the bank containing u(n+1,m+1)
        return (r3, r0, rerr) if errest else (r3, r0)

