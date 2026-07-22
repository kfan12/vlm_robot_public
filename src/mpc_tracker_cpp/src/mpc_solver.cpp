#include "mpc_tracker_cpp/mpc_solver.hpp"
#include <OsqpEigen/OsqpEigen.h>
#include <cmath>
#include <Eigen/Sparse>
#include <algorithm>

namespace
{
    inline double wrap_pi(double angle)
    {
        while (angle > M_PI)
            angle -= 2 * M_PI;
        while (angle < -M_PI)
            angle += 2 * M_PI;
        return angle;
    }
} // namespace

MpcSolver::MpcSolver(const VehicleParams &vehicle_params,
                     const MpcParams &mpc_params)
    : vp_(vehicle_params), mp_(mpc_params)
{
}

// Build and solve the linearized MPC as a QP problem.
// State error:[e_lat,e_head,e_speed]
// Control: [a, delta]
// total variables: (N+1)*3 + N*2

MpcResult MpcSolver::Solve(
    const VehicleState &current_state,
    const std::vector<PathPoint> &reference_path,
    double previous_accel,
    double previous_steer)
{
    MpcResult result;

    if (reference_path.size() < 2)
        return result; // not enough reference points

    const int n_state = 3;   // [e_lat, e_head, e_speed]
    const int n_control = 2; // [a, delta]
    const int N = mp_.N;
    const int n_vars = (N + 1) * n_state + N * n_control;

    const double L = vp_.wheelbase;
    const double dt = mp_.dt;
    // Two distinct roles for "speed", previously conflated:
    //  - v_ref: the speed-tracking SETPOINT the speed state should chase (cost gradient).
    //  - v_lin: the operating point the lateral/heading dynamics are LINEARIZED around.
    // v_lin must be the vehicle's CURRENT speed, not the target: using the target
    // linearizes the coupling around a speed the robot may not have reached (esp. at a
    // turn<->straight regime switch), making the model inconsistent with the plant. At
    // steady state v_lin ~= v_ref so this is a no-op; it only differs during transients.
    const double v_ref = mp_.target_speed;
    const double v_lin = std::clamp(current_state.v, 0.0, vp_.max_speed);

    // Reference tangent psi_ref from POINT geometry (not the poses' orientation,
    // which may be missing or wrong). Use a LOOKAHEAD bearing — the bearing from
    // point i to a point ~lookahead_m ahead on the path — instead of the raw local
    // segment tangent (ref[i]->ref[i+1], ~ds baseline). The local tangent turns a
    // couple cm of centerline noise into ~0.1-0.2 rad of reference-heading swing,
    // which the heading gain saturates into a steering limit cycle (the observed
    // straight-line oscillation). The long baseline averages that jitter out, and
    // curvature (kappa, a difference of tangents) inherits the de-noising. The node
    // schedules lookahead_m per maneuver (long on straights to kill noise, short in
    // turns to preserve curvature). lookahead_m <= 0 restores the local tangent.
    const int M = static_cast<int>(reference_path.size());
    const double look = mp_.lookahead_m;
    std::vector<double> psi_ref(M), kappa(M, 0.0);
    for (int i = 0; i < M; ++i)
    {
        int j = i + 1; // local-tangent default (look <= 0)
        if (look > 0.0)
        {
            j = i;
            while (j + 1 < M && reference_path[j].s - reference_path[i].s < look)
                ++j;
        }
        if (j >= M)
            j = M - 1;
        if (j == i)
            psi_ref[i] = (i > 0) ? psi_ref[i - 1] : 0.0; // no point ahead: hold
        else
            psi_ref[i] = std::atan2(reference_path[j].y - reference_path[i].y,
                                    reference_path[j].x - reference_path[i].x);
    }
    // kappa needs psi_ref[i+1], so it must run AFTER all tangents are filled.
    for (int i = 0; i < M - 1; ++i)
    {
        double ds = reference_path[i + 1].s - reference_path[i].s;
        if (ds > 1e-6)
            kappa[i] = wrap_pi(psi_ref[i + 1] - psi_ref[i]) / ds;
    }

    /*
    z = [e_lat_0, e_head_0, e_speed_0,   ← state errors at step 0, e_speed is actual speed, not error
         e_lat_1, e_head_1, e_speed_1,   ← state errors at step 1
         ...
         e_lat_N, e_head_N, e_speed_N,   ← state errors at step N
         a_0,   δ_0,                     ← controls at step 0
         a_1,   δ_1,                     ← controls at step 1
         ...
         a_{N-1}, δ_{N-1}]               ← controls at step N-1


    z = [e_lat_0, e_head_0, v_0,
         e_lat_1, e_head_1, v_1,
         …,
         e_lat_N, e_head_N, v_N,
         a_0, δ_0,
         a_1, δ_1,
         …,
         a_{N-1}, δ_{N-1}]

    */

    /*
    ```
        J = Σ_{i=0}^{N} [w_lat · e_lat(i)²  +  w_head · e_head(i)²  +  w_speed · e_speed(i)²]   // tracking error cost
            + Σ_{i=0}^{N-1} [w_accel · a(i)²  +  w_steer · δ(i)²]   // control effort cost
            + Σ_{i=0}^{N-2} [w_accel_change · Δa(i)²  +  w_steer_change · Δδ(i)²]  // control smoothness cost (Δa(i) = a(i+1)-a(i), Δδ(i) = δ(i+1)-δ(i))
        ```
    */

    // initial condition at the foot point (closest path point) = reference_path[0]

    const auto &ref0 = reference_path.front();
    const double psi0 = psi_ref.front();

    // Signed lateral offset in the path frame, positive = left of path
    // (same convention as solve_proportional and the e_lat dynamics row).
    const double e_lat0 = -std::sin(psi0) * (current_state.x - ref0.x) +
                          std::cos(psi0) * (current_state.y - ref0.y);
    const double e_head0 = wrap_pi(current_state.psi - psi0);
    const double v0 = current_state.v;

    // Cost matrix P and gradient q. Weights follow the (1/2)*w*(.)^2 convention
    // (OSQP applies the 1/2). P is built from triplets (setFromTriplets SUMS
    // duplicates) so the diagonal accumulates the state/effort weights plus the
    // control-rate (Δu) contributions. OsqpEigen takes the upper triangle, so
    // off-diagonal Δu coupling is emitted once as the upper (i<j) entry.
    const double w_da = mp_.w_accel_change; // Δaccel weight
    const double w_ds = mp_.w_steer_change; // Δsteer weight
    std::vector<Eigen::Triplet<double>> Ptr;
    Ptr.reserve(n_vars + 4 * N);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n_vars);

    // state tracking costs (diagonal) + speed target (linear term in q)
    for (int i = 0; i < N + 1; ++i)
    {
        Ptr.emplace_back(i * n_state + 0, i * n_state + 0, mp_.w_lat);
        Ptr.emplace_back(i * n_state + 1, i * n_state + 1, mp_.w_head);
        Ptr.emplace_back(i * n_state + 2, i * n_state + 2, mp_.w_speed);
        q(i * n_state + 2) = -mp_.w_speed * v_ref;
    }

    // control effort (diagonal) + control-rate smoothness (Δu). The Δu penalty
    // includes the change from the PREVIOUS applied command (u_0 - prev)^2 —
    // this is what damps the tick-to-tick steering limit cycle, and is why the
    // solver takes previous_accel/previous_steer (the QP ignored them before).
    auto ai = [&](int i) { return (N + 1) * n_state + i * n_control + 0; };
    auto si = [&](int i) { return (N + 1) * n_state + i * n_control + 1; };
    for (int i = 0; i < N; ++i)
    {
        Ptr.emplace_back(ai(i), ai(i), mp_.w_accel);
        Ptr.emplace_back(si(i), si(i), mp_.w_steer);
    }
    // boundary term (u_0 - prev)^2: diagonal += w, gradient += -w*prev
    Ptr.emplace_back(ai(0), ai(0), w_da);
    Ptr.emplace_back(si(0), si(0), w_ds);
    q(ai(0)) += -w_da * previous_accel;
    q(si(0)) += -w_ds * previous_steer;
    // within-horizon (u_{i+1} - u_i)^2: diagonal += w on both, upper off-diag -w
    for (int i = 0; i < N - 1; ++i)
    {
        Ptr.emplace_back(ai(i), ai(i), w_da);
        Ptr.emplace_back(ai(i + 1), ai(i + 1), w_da);
        Ptr.emplace_back(ai(i), ai(i + 1), -w_da);
        Ptr.emplace_back(si(i), si(i), w_ds);
        Ptr.emplace_back(si(i + 1), si(i + 1), w_ds);
        Ptr.emplace_back(si(i), si(i + 1), -w_ds);
    }

    Eigen::SparseMatrix<double> P(n_vars, n_vars);
    P.setFromTriplets(Ptr.begin(), Ptr.end());
    P.makeCompressed(); // OSQP requires compressed sparse column format

    // constraints: l<= Az <= u
    const int n_constraints = n_state * N + n_vars; // dynamics (n_state * N) + control bounds (n_control * N) + state bounds (n_state * (N+1))
    Eigen::SparseMatrix<double> A(n_constraints, n_vars);
    Eigen::VectorXd l(n_constraints);
    Eigen::VectorXd u(n_constraints);
    std::vector<Eigen::Triplet<double>> Tr;
    Tr.reserve(n_state * N * 3 + n_vars); // rough estimate of nonzeros

    // system dynamics constraints: e_lat, e_head, v, 3 rows per step, N steps.
    // Coupling terms linearized around v_lin (current speed), NOT v_ref (target):
    // e_lat_{i+1} = e_lat_i + dt*v_lin*e_head_i
    // e_head_{i+1} = e_head_i + dt*(v_lin/L)*delta_i - dt*v_lin*kappa_i
    // v_{i+1} = v_i + dt*a_i   (speed dynamics unaffected)

    for (int i = 0; i < N; ++i)
    {
        const int row = i * n_state;
        const int xi = i * n_state;
        const int xi1 = (i + 1) * n_state;
        const int ui = (N + 1) * n_state + i * n_control;
        const double k_i = kappa[std::min(i, M - 1)];

        // e_lat_{i+1} - e_lat_i - dt*v_lin*e_head_i = 0
        Tr.emplace_back(row + 0, xi1 + 0, 1.0);
        Tr.emplace_back(row + 0, xi + 0, -1.0);
        Tr.emplace_back(row + 0, xi + 1, -dt * v_lin);
        l(row + 0) = u(row + 0) = 0.0;

        // e_head_{i+1} - e_head_i - dt*(v_lin/L)*delta_i = -dt*v_lin*k_i
        Tr.emplace_back(row + 1, xi1 + 1, 1.0);
        Tr.emplace_back(row + 1, xi + 1, -1.0);
        Tr.emplace_back(row + 1, ui + 1, -dt * v_lin / L);
        l(row + 1) = u(row + 1) = -dt * v_lin * k_i;

        // v_{i+1} - v_i - dt*a_i = 0
        Tr.emplace_back(row + 2, xi1 + 2, 1.0);
        Tr.emplace_back(row + 2, xi + 2, -1.0);
        Tr.emplace_back(row + 2, ui + 0, -dt);
        l(row + 2) = u(row + 2) = 0.0;
    }

    // boundary conditions
    // e_lat_0, e_head_0,  v_0
    const int base = N * n_state;
    for (int i = 0; i < n_vars; ++i)
    {
        Tr.emplace_back(base + i, i, 1.0);

        l(base + i) = -OsqpEigen::INFTY;
        u(base + i) = OsqpEigen::INFTY;
    }
    // initial condition at the foot point (closest path point) = reference_path[0]
    l(base + 0) = u(base + 0) = e_lat0;
    l(base + 1) = u(base + 1) = e_head0;
    l(base + 2) = u(base + 2) = v0;

    // speed bounds: 0 <= v <= max_speed, for all steps after the pinned step 0
    for (int i = 1; i <= N; ++i)
    {
        l(base + i * n_state + 2) = 0.0;
        u(base + i * n_state + 2) = vp_.max_speed;
    }

    // control bounds: max_decel <= a <= max_accel, -max_steer <= delta <= max_steer, for all steps
    for (int i = 0; i < N; ++i)
    {
        l(base + (N + 1) * n_state + i * n_control + 0) = vp_.max_decel;
        u(base + (N + 1) * n_state + i * n_control + 0) = vp_.max_accel;
        l(base + (N + 1) * n_state + i * n_control + 1) = -vp_.max_steer;
        u(base + (N + 1) * n_state + i * n_control + 1) = vp_.max_steer;
    }

    A.setFromTriplets(Tr.begin(), Tr.end());
    A.makeCompressed();

    // OSQP solver setup
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(n_vars);
    solver.data()->setNumberOfConstraints(n_constraints);
    if (!solver.data()->setHessianMatrix(P) ||
        !solver.data()->setGradient(q) ||
        !solver.data()->setLinearConstraintsMatrix(A) ||
        !solver.data()->setLowerBound(l) ||
        !solver.data()->setUpperBound(u) ||
        !solver.initSolver())
    {
        return solve_proportional(current_state, reference_path, previous_accel, previous_steer);
    }
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError ||
        solver.getStatus() != OsqpEigen::Status::Solved)
    {
        return solve_proportional(current_state, reference_path, previous_accel, previous_steer);
    }

    Eigen::VectorXd z = solver.getSolution();

    // Extract first control input (a_0, delta_0) and predicted states
    result.accel = std::clamp(z[(N + 1) * n_state + 0], vp_.max_decel, vp_.max_accel);
    result.steer = std::clamp(z[(N + 1) * n_state + 1], -vp_.max_steer, vp_.max_steer);
    result.success = true;

    // Predicted preview: roll the NONLINEAR model out under the QP's control
    // sequence (the QP states are linearized errors, not poses).
    VehicleState s = current_state;
    for (int i = 0; i < N; ++i)
    {
        const int ui = (N + 1) * n_state + i * n_control;
        VehicleControl uc{std::clamp(z[ui + 0], vp_.max_decel, vp_.max_accel),
                          std::clamp(z[ui + 1], -vp_.max_steer, vp_.max_steer)};
        s = integrate_kinematic(s, uc, vp_, dt);
        result.predicted_states.push_back(s);
    }

    return result;
}

// -----------------------------------------------------------------------
// FALLBACK: proportional control on the MPC tracking-error state
// [e_lat, e_head, v] — the pre-QP stand-in, kept as the safety net for when
// the OSQP solve above fails to initialize or converge (infeasible, bad
// path, ...): never leave the car without a command. It is the single-step,
// unconstrained limit of the QP, so it shares the error definitions and the
// mpc_params.yaml weights stay meaningful here too: a heavier state weight
// (relative to its control weight) -> higher gain.
// -----------------------------------------------------------------------
MpcResult MpcSolver::solve_proportional(const VehicleState &current_state,
                                        const std::vector<PathPoint> &reference_path,
                                        double previous_accel,
                                        double previous_steer)
{
    MpcResult result;
    result.used_fallback = true;

    VehicleState s = current_state;
    double best_steer = 0.0;
    double best_accel = 0.0;

    // Reference foot point = closest path point (ref[0], where resampling began).
    // Take the path tangent from the geometry (ref[0] -> lookahead point), not from
    // the poses' orientation, so this works even if the drawn path has no headings.
    //
    // Heading reference = bearing from ref0 to a point ~mp_.lookahead_m ahead on the
    // path (pure-pursuit style). Using the LOCAL tangent (ref0 -> ref1, ~0.1 m
    // baseline) turns a couple cm of near-end lateral noise into ~0.1-0.2 rad of
    // reference-heading swing, which the heading gain then saturates into a steering
    // limit cycle. The longer lookahead baseline averages that jitter out and previews
    // the path. mp_.lookahead_m <= 0 restores the old local tangent for A/B testing.
    const auto &ref0 = reference_path.front();
    size_t look_idx;
    if (mp_.lookahead_m > 0.0)
    {
        look_idx = reference_path.size() - 1; // fall back to the farthest point if the path is short
        for (size_t i = 1; i < reference_path.size(); ++i)
        {
            double d = std::hypot(reference_path[i].x - ref0.x,
                                  reference_path[i].y - ref0.y);
            if (d >= mp_.lookahead_m)
            {
                look_idx = i;
                break;
            }
        }
    }
    else
    {
        look_idx = std::min<size_t>(1, reference_path.size() - 1);
    }
    const auto &ref_look = reference_path[look_idx];
    double psi_ref = std::atan2(ref_look.y - ref0.y, ref_look.x - ref0.x);

    // Cross-track error: signed lateral offset in the path frame (+ = left of path).
    double dx = s.x - ref0.x;
    double dy = s.y - ref0.y;
    double e_lat = -std::sin(psi_ref) * dx + std::cos(psi_ref) * dy;

    // Heading error, normalized to [-pi, pi].
    double e_head = s.psi - psi_ref;
    while (e_head > M_PI)
        e_head -= 2 * M_PI;
    while (e_head < -M_PI)
        e_head += 2 * M_PI;

    // Gains from the cost-weight ratios (state weight / control weight). e_lat and
    // e_head together act as PD on cross-track error (e_head ~ d(e_lat)/dt), so the
    // law is well-damped without a separate derivative term. Steering positive = left,
    // so a robot left of / pointing left of the path steers right (negative).
    double k_lat = mp_.w_lat / std::max(mp_.w_steer, 1e-6);
    double k_head = mp_.w_head / std::max(mp_.w_steer, 1e-6);
    double k_speed = mp_.w_speed / std::max(mp_.w_accel, 1e-6);

    best_steer = std::clamp(-(k_lat * e_lat + k_head * e_head),
                            -vp_.max_steer, vp_.max_steer);

    // Speed: the third state slot is ACTUAL speed (not an error) — same as the QP,
    // where the target is applied through the gradient q, not the state. The
    // proportional accel drives actual speed toward target_speed.
    best_accel = std::clamp(k_speed * (mp_.target_speed - s.v),
                            vp_.max_decel, vp_.max_accel);

    // Blend with the previous command — stand-in for the QP's control-rate
    // (smoothness) weights. A higher rate-weight => stronger blend toward the
    // previous command => smoother / less oscillation, which is exactly the knob
    // the Day 15 tuning rules reach for (raise weight_steer_rate to damp wobble).
    // Normalized to [0,1) so it can't run away; kRateRef sets where the default
    // weights sit (w_steer_change=2 -> ~0.33, matching the old fixed 0.3).
    const double kRateRef = 4.0;
    double smooth_steer = mp_.w_steer_change / (mp_.w_steer_change + kRateRef);
    double smooth_accel = mp_.w_accel_change / (mp_.w_accel_change + kRateRef);
    best_accel = (1.0 - smooth_accel) * best_accel + smooth_accel * previous_accel;
    best_steer = (1.0 - smooth_steer) * best_steer + smooth_steer * previous_steer;

    result.accel = best_accel;
    result.steer = best_steer;
    result.success = true;

    // Constant-control rollout for the /mpc_predicted_path preview.
    VehicleControl u{best_accel, best_steer};
    for (int i = 0; i < mp_.N; ++i)
    {
        s = integrate_kinematic(s, u, vp_, mp_.dt);
        result.predicted_states.push_back(s);
    }

    return result;
}