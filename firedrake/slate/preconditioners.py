from __future__ import absolute_import, print_function, division

import ufl

from firedrake import assemble
from firedrake.matrix_free.preconditioners import PCBase
from firedrake.petsc import PETSc
from firedrake.slate import Tensor


class HybridizationSlatePC(PCBase):
    """
    """

    def initialize(self, pc):
        """
        """
        from firedrake import (FunctionSpace, TrialFunction, TestFunction,
                               Function, TrialFunctions, TestFunctions,
                               BrokenElement, dS, FacetNormal)
        from firedrake.formmanipulation import ArgumentReplacer
        from ufl.algorithms.map_integrands import map_integrand_dags

        prefix = pc.getOptionsPrefix()
        _, P = pc.getOperators()
        context = P.getPythonContext()
        test, trial = context.a.arguments()

        # Break continuity of the space
        V = test.function_space()
        broken_elements = [BrokenElement(V_.ufl_element()) for V_ in V]
        if len(broken_elements) == 1:
            V_new = FunctionSpace(V.mesh(), broken_elements[0])
            arg_map = {test: TestFunction(V_new),
                       trial: TrialFunction(V_new)}
        else:
            V_new = FunctionSpace(V.mesh(), ufl.MixedElement(broken_elements))
            arg_map = {test: TestFunctions(V_new),
                       trial: TrialFunctions(V_new)}

        replacer = ArgumentReplacer(arg_map)

        # New from for the broken problem
        self.new_a = map_integrand_dags(replacer, context.a)

        # Create space of approximate traces and set up functions for the trace problem
        T = FunctionSpace(V.mesh(), "HDiv Trace", V_new.ufl_element().degree() - 1)
        self.broken_solution = Function(V_new)
        self.broken_rhs = Function(V_new)
        self.trace_solution = Function(T)

        # Generate Slate expressions for the Schur-complement system
        A = Tensor(self.new_a)
        gammar = TestFunction(T)
        n = FacetNormal(V.mesh())
        K = Tensor(gammar('+') * ufl.dot(trial[0], n) * dS)
        self.schur_lhs = K * A.inv * K.T
        self.schur_rhs = K * A.inv * self.broken_rhs

        # Set up KSP
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOperators(self.schur_lhs, self.schur_rhs)
        ksp.setOptionsPrefix(prefix + "trace_")
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

    def update(self, pc):
        """ """
        # Don't reconstruct symbolic objects, but update values in operator
        assemble(self.schur_lhs, tensor=self.schur_lhs)

    def apply(self, pc, x, y):
        """ """
        # Transfer non-broken x into firedrake function_space
        # transfer non-broken data into self.broken_rhs
        # compute trace rhs and assemble
        # solve trace solution and update self.trace_solution
        # back-sub into broken solutions
        # project broken solution into non-broken space and copy into y

    def applyTranspose(self, pc, x, y):
        """ """
        raise NotImplementedError("Not implemented yet!")
