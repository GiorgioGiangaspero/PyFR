# -*- coding: utf-8 -*-

from pyfr.integrators.dual.phys.base import BaseDualIntegrator

def axes2fig(axes, fig=None):
    import joblib
    import matplotlib.pyplot as plt
    from itertools import product
    """Converts ndarray of matplotlib object to matplotlib figure.

    Scikit-optimize plotting functions return ndarray of axes. This can be tricky
    to work with so you can use this function to convert it to the standard figure format.

    Args:
        axes(`numpy.ndarray`): Array of matplotlib axes objects.
        fig('matplotlib.figure.Figure'): Matplotlib figure on which you may want to plot
            your axes. Default None.

    Returns:
        'matplotlib.figure.Figure': Matplotlib figure with axes objects as subplots.

    Examples:
        Assuming you have a `scipy.optimize.OptimizeResult` object you want to plot::

            from skopt.plots import plot_evaluations
            eval_plot = plot_evaluations(result, bins=20)
            >>> type(eval_plot)
                numpy.ndarray

            from neptunecontrib.viz.utils import axes2fig
            fig = axes2fig(eval_plot)
            >>> fig
                matplotlib.figure.Figure

    """
    try:
        h, w = axes.shape
        if not fig:
            fig = plt.figure(figsize=(h * 3, w * 3))
        for i, j in product(range(h), range(w)):
            fig._axstack.add(fig._make_key(axes[i, j]), axes[i, j])
    except AttributeError:
        if not fig:
            fig = plt.figure(figsize=(6, 6))
        fig._axstack.add(fig._make_key(axes), axes)
    return fig

class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

    def _accept_step(self, idxcurr):
        self.tcurr += self._dt
        self.nacptsteps += 1
        self.nacptchain += 1

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.pseudoitegrator.system.filt(idxcurr)

        # Invalidate the solution cache
        self._curr_soln = None

        # Fire off any event handlers
        self.completed_step_handlers(self)

        # Clear the pseudo step info
        self.pseudointegrator.pseudostepinfo = []


class DualNoneController(BaseDualController):
    controller_name = 'none'

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            self.pseudointegrator.pseudo_advance(self.tcurr)
            self._accept_step(self.pseudointegrator._idxcurr)


class DualSkoptController(BaseDualController):
    controller_name = 'skopt'

    def _reject_step(self, err=None):
        self.nacptchain = 0
        self.nrjctsteps += 1

        # Copy the previously stored solution at time n and n-1 such
        # that we can start fresh at the next iteration.
        psnregs = self.pseudointegrator.pintg._pseudo_stepper_nregs

        u_src   = self.pseudointegrator.pintg._regidx[psnregs-2]
        um1_src = self.pseudointegrator.pintg._regidx[psnregs-1]
        # same as
        # u_src   = self.pseudointegrator.pintg._pseudo_stepper_regidx[-2]
        # um1_src = self.pseudointegrator.pintg._pseudo_stepper_regidx[-1]

        u_dst   = self.pseudointegrator.pintg._regidx[psnregs]
        um1_dst = self.pseudointegrator.pintg._regidx[psnregs+1]

        # print('Reject sol approach 1:')
        # print('u_src, um1_src, u_dst, um1_dst = {},{},{},{}'.format(u_src, um1_src, u_dst, um1_dst))

        # u_dst   = self.pseudointegrator.pintg._idxcurr
        # um1_dst = self.pseudointegrator.pintg._idxprev

        # print('Reject sol approach 2:')
        # print('u_src, um1_src, u_dst, um1_dst = {},{},{},{}'.format(u_src, um1_src, u_dst, um1_dst))

        self.pseudointegrator.pintg._add(0, u_dst,   1, u_src)
        self.pseudointegrator.pintg._add(0, um1_dst, 1, um1_src)

    def store_curr_sol(self):
        psnregs = self.pseudointegrator.pintg._pseudo_stepper_nregs

        u_src   = self.pseudointegrator.pintg._regidx[psnregs]
        um1_src = self.pseudointegrator.pintg._regidx[psnregs+1]

        u_dst   = self.pseudointegrator.pintg._regidx[psnregs-2]
        um1_dst = self.pseudointegrator.pintg._regidx[psnregs-1]
        # same as
        # u_dst   = self.pseudointegrator.pintg._pseudo_stepper_regidx[-2]
        # um1_dst = self.pseudointegrator.pintg._pseudo_stepper_regidx[-1]

        # print('Store current sol approach 1:')
        # print('u_src, um1_src, u_dst, um1_dst = {},{},{},{}'.format(u_src, um1_src, u_dst, um1_dst))

        # u_src   = self.pseudointegrator.pintg._idxcurr
        # um1_src = self.pseudointegrator.pintg._idxprev

        # print('Store current sol approach 2:')
        # print('u_src, um1_src, u_dst, um1_dst = {},{},{},{}'.format(u_src, um1_src, u_dst, um1_dst))

        self.pseudointegrator.pintg._add(0, u_dst,   1, u_src)
        self.pseudointegrator.pintg._add(0, um1_dst, 1, um1_src)

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        from copy import deepcopy
        optdone = False
        self.csteps = deepcopy(self.pseudointegrator.csteps)

        while self.tcurr < t:
            if self.tcurr < 5.0 or optdone:
                self.pseudointegrator.csteps = self.csteps

                self.pseudointegrator.pseudo_advance(self.tcurr)
                self._accept_step(self.pseudointegrator._idxcurr)
            else:
                if not optdone:
                    # after the specified time run the optimization
                    print('Starting the optimization...')
                    self.run_optimisation()
                    optdone = True

                    # import sys
                    # sys.exit('Optimisation concluded!')

    def run_optimisation(self):
        import joblib
        from skopt.space import Integer
        #from skopt.utils import use_named_args
        from skopt.plots import plot_convergence, plot_evaluations, plot_objective
        from skopt import gp_minimize, forest_minimize


        # Store the current solutions
        self.store_curr_sol()

        # get the number of levels and initial settings
        # from the cycle information
        cycle, csteps = self.pseudointegrator.cycle, self.pseudointegrator.csteps

        #initial guess
        x0 = csteps

        # set the optimisation space
        space  = [Integer(0, 4, name='clevel-{}'.format(i)) for i in range(len(csteps))]

        # run it!
        use_GP = False
        max_iter = 15

        if use_GP:
            res = gp_minimize(self.objective_function, space, n_calls=max_iter,
                              random_state=123, verbose=False, x0=x0)
        else:
            res = forest_minimize(self.objective_function, space, n_calls=max_iter,
                                  random_state=123, verbose=False, x0=x0)

        print('minimum cputime = {}'.format(res.fun))
        print('optimal cycle   = {}'.format(res.x))

        fig1 = axes2fig(plot_convergence(res))
        fig1.savefig('convergence.png')
        # fig2 = axes2fig(plot_evaluations(res))
        # fig2.savefig('evaluations.png')
        # fig3 = axes2fig(plot_objective(res))
        # fig3.savefig('objective.png')

        return res

    def objective_function(self, x):
        import time

        #print('Evaluating one objective function')

        #change the pMG properties according to x
        self.pseudointegrator.csteps = x
        #TMPTMP
        print('TMP: overwriting cycle!')
        self.pseudointegrator.csteps = self.csteps

        # verify the change
        print('csteps_opt_cal = {}'.format(self.pseudointegrator.csteps))

        #get the starting cpu time
        CPU_time = time.time()

        #do one full cycle. This function should return
        #wether or not the residuals have converged
        if self.pseudointegrator.pseudo_advance(self.tcurr):
            # compute the CPU time
            CPU_time = time.time() - CPU_time
        else:
            # if the residuals have not converged, set the CPU
            # time to an asbsurdly high value.
            CPU_time = 1.e+12
        # then reject the step to make sure we are restarting
        # every time from the same solution.
        self._reject_step()

        #print('CPU_time = {}'.format(CPU_time))
        return CPU_time

