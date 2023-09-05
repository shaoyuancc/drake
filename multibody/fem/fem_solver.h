#pragma once

#include <memory>
#include <unordered_set>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_cholesky_solver.h"
#include "drake/multibody/contact_solvers/block_sparse_lower_triangular_or_symmetric_matrix.h"
#include "drake/multibody/contact_solvers/schur_complement.h"
#include "drake/multibody/fem/discrete_time_integrator.h"
#include "drake/multibody/fem/fem_model.h"
#include "drake/multibody/fem/fem_state.h"

namespace drake {
namespace multibody {
namespace fem {
namespace internal {

template <typename T>
class FemSolver;

/* Data structure to store data used in the FemSolver.
 @tparam_double_only */
template <typename T>
class FemSolverData {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FemSolverData);

  /* Constructs a FemSolverData that is compatible with the given model. */
  explicit FemSolverData(const FemModel<T>& model);

  const contact_solvers::internal::Block3x3SparseSymmetricMatrix&
  tangent_matrix() const {
    return *tangent_matrix_;
  }

  const contact_solvers::internal::SchurComplement& schur_complement() const {
    return schur_complement_;
  }

  const std::unordered_set<int>& nonparticipating_vertices() const {
    return nonparticipating_vertices_;
  }

  /* @pre All entries in `nonparticipating_vertices` are in
   [0, tangent_matrix().block_cols()). */
  void set_nonparticipating_vertices(
      std::unordered_set<int> nonparticipating_vertices) {
    nonparticipating_vertices_ = std::move(nonparticipating_vertices);
  }

 private:
  friend class FemSolver<T>;
  copyable_unique_ptr<contact_solvers::internal::Block3x3SparseSymmetricMatrix>
      tangent_matrix_;
  contact_solvers::internal::BlockSparseCholeskySolver<Matrix3<T>>
      linear_solver_;
  contact_solvers::internal::SchurComplement schur_complement_;
  std::unordered_set<int> nonparticipating_vertices_;
  VectorX<T> b_;
  VectorX<T> dz_;
};

/* FemSolver solves discrete dynamic elasticity problems. The governing PDE of
 the dynamics is spatially discretized in FemModel and temporally discretized by
 DiscreteTimeIntegrator. FemSolver provides the `AdvanceOneTimeStep()` function
 that advances the free-motion states (i.e. without considering contacts or
 constraints) of the spatially discretized FEM model by one time step according
 to the prescribed discrete time integration scheme using a Newton-Raphson
 solver.
 @tparam_double_only */
template <typename T>
class FemSolver {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FemSolver);

  /* Constructs a new FemSolver that solves the given `model` with the
   `integrator` provided to advance time.
   @note The `model` and `integrator` pointers persist in `this` FemSolver and
   thus the model and the integrator must outlive this solver.
   @pre model != nullptr.
   @pre integrator != nullptr.*/
  FemSolver(const FemModel<T>* model,
            const DiscreteTimeIntegrator<T>* integrator);

  // TODO(#20086): Clean up this messy interface.
  /* Advances the state of the FEM model by one time step with the integrator
   prescribed at construction.
   @param[in] prev_state    The state of the FEM model evaluated at the previous
                            time step.
   @param[out] next_state   The state of the FEM model evaluated at the next
                            time step.
   @param[in, out] data     On input, provides the set of participating vertices
                            to help evalulate free-motion state quantities. It
                            also serves scratch pad for storing intermediary
                            data used in the computation. On output, stores the
                            Schur complement of the tangent matrix (of the force
                            balance equations) at the free motion state. If no
                            Newton-Raphson iteration is taken (i.e. in steady
                            state), data.schur_complement remains unchanged.
   @returns the number of Newton-Raphson iterations the solver takes to
   converge if the solver converges or -1 if the solver fails to converge.
   @pre next_state != nullptr and data != nullptr.
   @throws std::exception if the input `prev_state` or `next_state` is
   incompatible with the FEM model solved by this solver. */
  int AdvanceOneTimeStep(const FemState<T>& prev_state, FemState<T>* next_state,
                         FemSolverData<T>* data) const;

  /* Returns the FEM model that this solver solves for. */
  const FemModel<T>& model() const { return *model_; }

  /* Returns the discrete time integrator that this solver uses. */
  const DiscreteTimeIntegrator<T>& integrator() const { return *integrator_; }

  /* Sets the relative tolerance, unitless. See solver_converged() for how
   the tolerance is used. The default value is 1e-4. */
  void set_relative_tolerance(double tolerance) {
    relative_tolerance_ = tolerance;
  }

  double relative_tolerance() const { return relative_tolerance_; }

  /* Sets the absolute tolerance with unit Newton. See solver_converged() for
   how the tolerance is used. The default value is 1e-6. */
  void set_absolute_tolerance(double tolerance) {
    absolute_tolerance_ = tolerance;
  }

  double absolute_tolerance() const { return absolute_tolerance_; }

  /* The solver is considered as converged if ‖r‖ < max(εᵣ * ‖r₀‖, εₐ) where r
   and r₀ are `residual_norm` and `initial_residual_norm` respectively, and εᵣ
   and εₐ are relative and absolute tolerance respectively. */
  bool solver_converged(const T& residual_norm,
                        const T& initial_residual_norm) const;

 private:
  /* Uses a Newton-Raphson solver to solve for the unknown z such that the
   residual is zero, i.e. b(z) = 0, up to the specified tolerances. The input
   FEM state is non-null and is guaranteed to be compatible with the FEM model.

   @param[in, out] state  As input, `state` provides an initial guess of
   the solution. As output, `state` reports the equilibrium state.
   @param[in, out] data   On input, provides data in addition to the FemState
   (such as participating vertices and time step) to help evalulate free-motion
   state quantities. It also serves scratch pad for storing intermediary data
   used in the computation. On output, stores the Schur complement of the
   tangent matrix at the free motion state.
   @returns the number of iterations it takes for the solver to converge or -1
   if the solver fails to converge. */
  int SolveWithInitialGuess(FemState<T>* state, FemSolverData<T>* data) const;

  /* The FEM model being solved by `this` solver. */
  const FemModel<T>* model_{nullptr};
  /* The discrete time integrator the solver uses. */
  const DiscreteTimeIntegrator<T>* integrator_{nullptr};
  /* Tolerance for convergence. */
  double relative_tolerance_{1e-4};  // unitless.
  // TODO(xuchenhan-tri): Consider using an absolute tolerance with velocity
  // unit so that how stiff the material is doesn't affect the convergence
  // criterion.
  double absolute_tolerance_{1e-6};  // unit N.
  /* Max number of Newton-Raphson iterations the solver takes before it gives
   up. */
  int kMaxIterations_{100};
};

}  // namespace internal
}  // namespace fem
}  // namespace multibody
}  // namespace drake
