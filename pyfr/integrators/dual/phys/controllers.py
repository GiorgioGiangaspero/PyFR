# -*- coding: utf-8 -*-

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


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

        # Rotate the stepper registers to the left such that at the next call
        # we will have the same physical solutions as for the previous call
        psnregs = self.pseudointegrator.pintg._pseudo_stepper_nregs
        snregs = self.pseudointegrator.pintg._stepper_nregs

        self.pseudointegrator.pintg._regidx[psnregs:psnregs + snregs] = (
            self.pseudointegrator.pintg._stepper_regidx[1:] +
            [self.pseudointegrator.pintg._stepper_regidx[0]]
        )

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            if self.tcurr < 10.0:
                self.pseudointegrator.pseudo_advance(self.tcurr)
                self._accept_step(self.pseudointegrator._idxcurr)
            else:
                # after the specified time run the optimisation
                print('Starting the optimisationi...')
                self.run_optimisation()

                import sys
                sys.exit('Optimisation concluded!')

    def run_optimisation(self):
        from skopt.space import Integer
        #from skopt.utils import use_named_args
        from skopt.plots import plot_convergence, plot_evaluations, plot_objective
        from skopt import gp_minimize, forest_minimize

        # get the number of levels and initial settings
        # from the cycle information
        cycle, csteps = self.pseudointegrator.cycle, self.pseudointegrator.csteps

        #initial guess
        x0 = csteps

        # set the optimisation space
        space  = [Integer(0, 4, name='level-{}'.format(i)) for i in range(len(csteps))]

        # run it!
        use_GP = False
        max_iter = 500

        if use_GP:
            res = gp_minimize(self.objective_function, space, n_calls=max_iter,
                              random_state=123, verbose=False, x0=x0)
        else:
            res = forest_minimize(self.objective_function, space, n_calls=max_iter,
                                  random_state=123, verbose=False, x0=x0)

        print('minimum cputime = {}'.format(res.fun))
        print('optimal cycle   = {}'.format(res.x))

        #plot_convergence(res)
        #plot_evaluations(res)
        #plot_objective(res)

        return res

    def objective_function(self, x):
        import time

        #print('Evaluating one objective function')

        #change the pMG properties according to x
        self.pseudointegrator.csteps = x

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

        print('CPU_time = {}'.format(CPU_time))
        return CPU_time
        
