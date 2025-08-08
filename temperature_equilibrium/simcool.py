#!/usr/bin/env python

import argparse
import numpy as np
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

class SimCool:
    def __init__(self):
        self.args = dict(
            output=None,
            thick_s=None, thick_e=None,
            alpha_s=None, alpha_e=None,
            temp_0=None, temp_e=None,
            run_dx=None, run_dt=None, run_tt=None,
            save_dx=None, save_dt=None
        )
        self.steps = dict(
            x=None, T=None,
            AinvB=None, Ainvb=None,
            save_ndt=None, mask_x=None
        )
        self.results = dict(
            xs=None, ts=None, Ts=None
        )

    def build_argparser(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument('--output', default='test', type=str, help='output basename.')
        parser.add_argument('--thick-s', default=100, type=float, help='thickness of specimen, in nm.')
        parser.add_argument('--thick-e', default=1000, type=float, help='thickness of environment on each side, in nm.')
        parser.add_argument('--alpha-s', default=133, type=float, help='thermal diffusivity of specimen, in nm^2/ns.')
        parser.add_argument('--alpha-e', default=20000, type=float, help='thermal diffusivity of environment, in nm^2/ns.')
        parser.add_argument('--temp-0', default=300, type=float, help='temperature of specimen at initial, in T.')
        parser.add_argument('--temp-e', default=300, type=float, help='temperature of environment, in T.')
        parser.add_argument('--run-dx', default=1, type=float, help='length step for simulation, in nm.')
        parser.add_argument('--run-dt', default=0.01, type=float, help='time step for simulation, in ns.')
        parser.add_argument('--run-tt', default=1000, type=float, help='total time for simulation, in ns.')
        parser.add_argument('--save-dx', default=1, type=float, help='length step for saving, in nm.')
        parser.add_argument('--save-dt', default=1000, type=float, help='time step for saving, in ns.')
        return parser

    def load_args(self, args=None):
        if args is None:
            parser = self.build_argparser()
            args = parser.parse_args()
            args = vars(args)
        self.args.update(args)
        return self

    def load_state(self, fstate):
        state = np.load(fstate, allow_pickle=True)
        self.args.update(state['args'].item())
        self.steps.update(state['steps'].item())
        self.results.update(state['results'].item())
        return self

    def save_state(self, fstate):
        np.savez(
            fstate,
            args=self.args,
            steps=self.steps,
            results=self.results
        )
    
    def prep_matrices(self):
        """
        AT(n+1) = BT(n) + b
        """
        # read args
        args = self.args
        dx = args['run_dx']
        dt = args['run_dt']
        # make auxiliary quantities
        thick_s2 = args['thick_s']/2
        box2 = args['thick_e'] + thick_s2        
        x = np.arange(-box2+dx, box2, dx).astype(int)  # len=N-2
        nx = len(x)
        c = dt/(4*dx**2)

        # construct b
        b = np.zeros(nx, dtype=float)
        b[0] = 2*c*args['temp_e']*args['alpha_e']
        b[-1] = b[0]

        # construct A, B
        l = c*np.piecewise(
            x,
            [
                (x-0.5*dx) < -thick_s2,
                np.abs(x-0.5*dx) <= thick_s2,
                (x-0.5*dx) > thick_s2
            ],
            [args['alpha_e'], args['alpha_s'], args['alpha_e']]
        )
        r = c*np.piecewise(
            x,
            [
                (x+0.5*dx) < -thick_s2,
                np.abs(x+0.5*dx) <= thick_s2,
                (x+0.5*dx) > thick_s2
            ],
            [args['alpha_e'], args['alpha_s'], args['alpha_e']]
        )
        A = sparse.csc_matrix(sparse.dia_array(
            (
                [1+l+r, -np.roll(r, 1), -np.roll(l, -1)],
                [0, 1, -1]
            ), 
            shape=(nx, nx)
        ))
        B = sparse.csc_matrix(sparse.dia_array(
            (
                [1-l-r, np.roll(r, 1), np.roll(l, -1)],
                [0, 1, -1]
            ), 
            shape=(nx, nx)
        ))

        # calc. AinvB, Ainvb
        Ainv = sparse.csr_matrix(sparse.linalg.inv(A))
        AinvB = Ainv @ B
        Ainvb = Ainv @ b

        # calc. init T
        T0 = np.piecewise(
            x,
            [
                x <= -thick_s2,
                np.abs(x) < thick_s2,
                x >= thick_s2
            ],
            [args['temp_e'], args['temp_0'], args['temp_e']]
        )

        # assign
        return x, T0, AinvB, Ainvb
    
    def prep_simulation(self):
        args = self.args

        # get matrices
        x, T0, AinvB, Ainvb = self.prep_matrices()
        
        # get saving-related
        save_ndt = int(args['save_dt']/args['run_dt'])
        save_ndx = int(args['save_dx']/args['run_dx'])
        mask_x = np.arange(len(x)) % save_ndx == 0

        self.steps.update(dict(
            x=x, T=T0,
            AinvB=AinvB, Ainvb=Ainvb,
            save_ndt=save_ndt, mask_x=mask_x
        ))

        self.results.update(dict(
            xs=x[mask_x],
            ts=[0.],
            Ts=[T0[mask_x]]
        ))

    def run_simulation(self, tt=None):
        args = self.args
        steps = self.steps
        fstate = args['output']+'.npz'

        if tt is None:
            tt = args['run_tt']
        t0 = self.results['ts'][-1]
        t = self.results['ts'][-1]
        T = steps['T']

        print(args['output']+': running simulation')
        while t < t0+tt:
            print(t)
            for i in range(steps['save_ndt']):
                T = steps['AinvB']@T + steps['Ainvb']
            t += args['save_dt']
            self.results['ts'].append(t)
            self.steps['T'] = T
            self.results['Ts'].append(T[steps['mask_x']])
            self.save_state(fstate)
        
        self.results['ts'] = np.asarray(self.results['ts'])
        self.results['Ts'] = np.asarray(self.results['Ts'])
        self.save_state(fstate)

    def get_temp_center(self):
        """
        returns: time and temperature for the center point
        """
        ic = np.argmin(np.abs(self.results['xs']))
        tc = np.asarray(self.results['ts'])
        Tc = np.asarray(self.results['Ts'])[:, ic]
        return tc, Tc

    def get_fpt_center(self, Tf_list):
        """
        fpt: first passage time to Tf
        returns: list of fpt that reaches Tf
        """
        tc, Tc = self.get_temp_center()
        fpt_list = []
        for Tf in Tf_list:
            fpt = tc[Tc<=Tf]
            if len(fpt) == 0:
                fpt = np.nan
            else:
                fpt = fpt[0]
            fpt_list.append(fpt)
        return fpt_list
        

    def plot_profile(self, ts=None, n=5, save=True):
        args = self.args
        results = self.results

        if ts is None:
            nt = len(results['ts'])
            n = min(nt, n)
            ts = np.asarray(results['ts'])[np.linspace(0, nt-1, n).astype(int)]

        fig, ax = plt.subplots(
            constrained_layout=True
        )
        for i, t in enumerate(results['ts']):
            if t in ts:
                label_i = f"t={t:.0f}ns"
                ax.plot(results['xs'], results['Ts'][i], label=label_i)
        ax.axvline(-args['thick_s']/2, color='gray', alpha=0.5)
        ax.axvline(args['thick_s']/2, color='gray', alpha=0.5)
        ax.axhline(args['temp_0'], color='gray', alpha=0.5)
        ax.axhline(args['temp_e'], color='gray', alpha=0.5)
        ax.legend(loc=1)
        ax.set(xlabel='x/nm', ylabel='T/K')
        if save:
            fig.savefig(args['output']+'.png')

    def workflow(self, args=None):
        self.load_args(args)
        self.prep_simulation()
        self.run_simulation(tt=self.args['run_tt'])
        self.plot_profile(n=5)

if __name__ == "__main__":
    sim = SimCool()
    sim.workflow()
