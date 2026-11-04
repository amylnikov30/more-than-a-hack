// Possession HMM: design notes for the More Than A Hack challenge (BAR-ATH)
// Compile with:  typst compile hmm_writeup.typ

#set document(title: "A sticky HMM for possession", author: "more-than-a-hack team")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show link: underline

#let argmax = math.op("arg max", limits: true)
#let lse = math.op("logsumexp", limits: true)
#let note(body) = block(
  fill: rgb("#eef3fb"), stroke: (left: 2pt + rgb("#3b6fb6")),
  inset: 10pt, radius: 2pt, width: 100%, body,
)
#let warn(body) = block(
  fill: rgb("#fbf1e6"), stroke: (left: 2pt + rgb("#c47a1c")),
  inset: 10pt, radius: 2pt, width: 100%, body,
)

#align(center)[
  #text(size: 18pt, weight: "bold")[A sticky hidden Markov model for possession]
  #v(2pt)
  #text(size: 11pt)[Design notes, assumptions and maths: FC Barcelona vs Athletic Club]
]

#v(8pt)

#note[
  *Summary.* An HMM is the right backbone for this problem, not an overly obvious one. Possession
  is a sequence of long, persistent regimes observed through noisy evidence, which is exactly what
  an HMM models. Your "threshold before switching" idea is already built in: under Viterbi decoding
  the self-transition probability _is_ the threshold, in units of accumulated log-likelihood
  ratio (@sec-threshold). Most of the score will come from four things around the HMM:
  team-symmetric features that keep working when the ball is missing (@sec-features), honest
  calibration of the emissions (@sec-temper), a small amount of state structure for loose
  balls (@sec-substates), and tuning on _other_ labelled games instead of the public window
  (@sec-validation).
]

= The problem in HMM terms

Per half $h in {1, 2}$, let $t = 1, dots, T_h$ index the graded video frames. Each frame has a
hidden *label* $z_t$ from a small set of state values. (A precise wording: "each frame _has_ a
state". The model has $K$ state values, not one per frame.) The basic model uses
$ z_t in cal(K) = {H, A, D}, $
for home possession, away possession and dead ball. Each frame also carries an observation vector
$bold(x)_t in RR^F$ of features computed from tracking, plus optional discrete evidence (ball
missing, Smart Tagging).

The HMM factorises the joint distribution of labels and observations as
$ p(z_(1:T), bold(x)_(1:T)) = pi_(z_1) product_(t=2)^T A_(z_(t-1) z_t)
  product_(t=1)^T b_(z_t) (bold(x)_t), $ <eq:joint>
where $pi$ is the initial distribution, $A_(j k) = p(z_t = k | z_(t-1) = j)$ is the transition
matrix, and $b_k (bold(x)) = p(bold(x)_t = bold(x) | z_t = k)$ is the emission density.

Run the model *separately on each half*. Write `X` on every ungraded frame. Use different
initial distributions for the two halves:

- *1st half:* the graded block starts at the kickoff, so $pi = (1/2, 1/2, 0)$ over $(H, A, D)$
  and the features decide who kicked off.
- *2nd half:* the graded block starts 2,934 frames _before_ the kickoff, and all of them are `D`
  in the ground truth. Use $pi approx (0, 0, 1)$ so the model starts in `D` and has to find the
  kickoff.

== Assumptions, stated up front

Your technical document should list these, and say where each one fails.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 6pt,
  stroke: 0.5pt + luma(180),
  table.header([*Assumption*], [*Where it breaks here*], [*Mitigation*]),
  [*A1 Markov:* $z_t$ depends only on $z_(t-1)$],
  [Possession durations are not geometric. Restarts after `D` go to a specific team.],
  [Sub-state chains / semi-Markov durations (@sec-durations). Restart features.],
  [*A2 Conditional independence:* $bold(x)_t$ depends only on $z_t$],
  [Badly broken at 25 fps. Consecutive frames are nearly identical, so evidence is counted
   many times over.],
  [Tempered emissions or subsampling (@sec-temper).],
  [*A3 Gaussian emissions*],
  [Distances are skewed, counts are discrete, tracking glitches give heavy tails.],
  [Log transforms, Student-$t$ or mixture emissions.],
  [*A4 Diagonal covariance*],
  [Several features measure nearly the same thing (e.g. nearest distance and players within 2 m).],
  [Full covariance per state, or decorrelate features, or fold into tempering.],
  [*A5 Missing at random*],
  [The ball goes missing _more_ in crowded boxes, aerial duels and stoppages.],
  [Model missingness itself as an observation (@sec-missing).],
  [*A6 Team symmetry:* the $A$ state is the mirror of $H$],
  [Barcelona and Athletic play differently.],
  [Tie only the means' structure, or tie with a small learned asymmetry.],
  [*A7 Stationarity* within a half],
  [Fatigue, score state, tactical changes.],
  [Usually small. Per-half parameters if needed.],
)

= Emission model <sec-emissions>

== Diagonal Gaussian emissions

The simplest emission model is a diagonal Gaussian per state:
$ log b_k (bold(x)_t) = sum_(d=1)^F log cal(N) (x_(t d); mu_(k d), sigma_(k d)^2)
  = - 1/2 sum_(d=1)^F [ log(2 pi sigma_(k d)^2) + (x_(t d) - mu_(k d))^2 / sigma_(k d)^2 ]. $
<eq:emission>

The quantity that drives every decision is the per-frame *log-likelihood ratio* between two states,
$ ell_t (j, k) = log b_k (bold(x)_t) - log b_j (bold(x)_t). $ <eq:llr>
With equal variances this is linear in $bold(x)_t$. That means a Gaussian HMM with tied variances
is a linear classifier per frame plus a temporal penalty for switching.

== Missing data is free in a diagonal Gaussian <sec-missing>

Let $O_t subset.eq {1, dots, F}$ be the features observed on frame $t$. A marginal of a Gaussian is
Gaussian, so a missing feature is integrated out exactly by *dropping its term* from
@eq:emission:
$ log b_k (bold(x)_t) = sum_(d in O_t) log cal(N) (x_(t d); mu_(k d), sigma_(k d)^2). $
<eq:emission-missing>
If every feature is missing, $b_k = 1$ for all $k$ and the transitions alone carry the state
through the gap. This is the right behaviour: going quiet during an occlusion leaves the
possession unchanged. (With full covariance, use the sub-vector $bold(x)_(t, O_t)$ with the
sub-matrix $Sigma_(k, O_t O_t)$.)

That is only exact if the data are *missing at random*. Here they are not: the ball disappears more
in some states than others. The cheap fix is to observe the missingness itself. Add a binary
$m_t = bb(1)["ball missing at" t]$ with a Bernoulli emission,
$ p(m_t | z_t = k) = rho_k^(m_t) (1 - rho_k)^(1 - m_t), $
and multiply it into $b_k$. EM learns $rho_k$. For example, it can learn that the ball being out
of view for several seconds is evidence for `D`.

#warn[
  *Design for the ball being missing.* About 41% of graded frames have no ball. Every feature that
  needs the ball goes `NaN` on those frames, so you need *some features that don't need the ball*
  (team shape, territory, player speeds; see @sec-features). Otherwise the model is blind for 41%
  of the match and runs on transitions alone.
]

== Team symmetry: tie $A$ to the mirror image of $H$ <sec-symmetry>

Unsupervised EM has no idea which state means "home possession". Left alone, it will happily use
its three states for something else, such as "ball in left half / right half / missing". The fix
is to build the team symmetry of football into the parameters.

Split the features into *antisymmetric* ones, which flip sign when you swap the team labels (e.g.
"away's nearest distance minus home's nearest distance"), and *symmetric* ones, which don't
change (e.g. ball speed). Define the swap operator as a diagonal sign matrix
$ S = "diag"(s_1, dots, s_F), quad s_d = cases(-1 & "if feature" d "is antisymmetric", +1 & "if symmetric"). $
Then *tie* the parameters:
$ bold(mu)_A = S bold(mu)_H, quad Sigma_A = S Sigma_H S quad (= Sigma_H "for diagonal" Sigma). $
<eq:tying>
Orient every antisymmetric feature so that *positive means evidence for home*. The labels are then
identifiable by construction: $H$ is the state whose antisymmetric means are positive. You can
never end up with $H$ and $A$ swapped.

Tying also halves the parameters to learn, and every possession by either team trains both
states (the M-step in @sec-learning effectively mirrors the data). If the teams really do behave
differently (Barcelona's possession style is distinctive), relax it to
$bold(mu)_A = S bold(mu)_H + bold(delta)$ with a penalty on $norm(bold(delta))$.

= Transition model and the switching threshold

== Sticky transitions

For the three-state model, use a "sticky" transition matrix (rows: from, columns: to, order
$H, A, D$):
$ A = mat(
  1 - epsilon - eta, epsilon, eta;
  epsilon, 1 - epsilon - eta, eta;
  omega, omega, 1 - 2 omega;
). $ <eq:transitions>
This assumes the same rates for both teams, consistent with @eq:tying. Here $epsilon$ is the
per-frame probability of a turnover, $eta$ of the ball going dead, and $omega$ of a restart
going to a given team.

The time spent in a state is *geometric*. For state $H$ with self-transition
$a = 1 - epsilon - eta$:
$ P("duration" = d) = a^(d-1) (1 - a), quad EE["duration"] = 1 / (1 - a) "frames". $ <eq:geometric>
So if a typical spell lasts $L$ seconds, set $1 - a approx 1 \/ (25 L)$. For $L = 10$ s that
gives $1 - a = 0.004$.

#warn[
  *Where these numbers come from matters.* Setting $epsilon, eta, omega$ from the public
  5-minute window (its possession count or mean duration) is tuning on the public ground truth,
  which the rules forbid. Get them from *labelled other games* or from published football
  statistics, and write down which.
]

== Viterbi decoding is a penalised segmentation <sec-threshold>

The decoded label sequence is the most probable path,
$ hat(z)_(1:T) = argmax_(z_(1:T)) log p(z_(1:T), bold(x)_(1:T)). $ <eq:map>
Take the negative log of @eq:joint and write the cost of a path as
$ C(z_(1:T)) = sum_(t=1)^T underbrace(- log b_(z_t) (bold(x)_t), "frame cost")
  + sum_(t=2)^T underbrace(- log A_(z_(t-1) z_t), "transition cost"). $
For a sticky matrix, staying costs almost nothing ($-log(1 - epsilon - eta) approx 0$) and
switching costs a lot. Up to a constant, Viterbi therefore minimises
$ C(z_(1:T)) approx sum_t - log b_(z_t) (bold(x)_t) + sum_(t: z_t != z_(t-1)) lambda_(z_(t-1) z_t),
  quad lambda_(j k) = log A_(j j) / A_(j k). $ <eq:penalised>
This is an $L_0$-penalised change-point problem: fit the data, pay a fixed price $lambda$ per
switch.

*This is your threshold, made exact.* Take the two-state case $H$/$A$ with
$lambda = log((1 - epsilon) \/ epsilon)$. Compare staying in $H$ throughout against a path that
switches to $A$ for frames $s, dots, e$ and then switches back. The stays inside the excursion
cost the same either way, so the excursion wins exactly when
$ sum_(t=s)^e ell_t (H, A) > 2 lambda. $ <eq:excursion>
A *permanent* switch (no return) needs only
$ sum_(t=s)^(e) ell_t (H, A) > lambda, $ <eq:switch>
where $e$ is the end of the new spell. So:

- A brief touch or deflection by the other team produces a short burst of positive
  $ell_t$. If the burst totals less than $2 lambda$, it is *ignored*. That is exactly the "a touch
  without control is not a change of possession" rule.
- A real turnover produces sustained evidence that eventually exceeds $lambda$, and the label
  changes.
- $lambda$ is the *single knob* for how eager the model is to switch. For $epsilon = 0.004$,
  $lambda = log(0.996 \/ 0.004) approx 5.5$ nats, so a deflection needs about 11 nats of
  evidence to register.

#note[
  *Offline decoding has no detection delay.* If you implement the threshold *online* (switch once
  the running evidence crosses $lambda$), every switch is reported only after the evidence has
  built up, which can easily be half a second late. On `spellF1` a one-second delay drops a
  perfect submission to 0.65. Viterbi uses the whole sequence: it decides that a switch happened
  and then backtracks to put it at the frame where the evidence _started_. Always decode offline.
  We have the whole match.
]

== Correlated frames: temper the emissions <sec-temper>

Assumption A2 says the $bold(x)_t$ are independent given $z_t$. At 25 fps that is false.
Features change smoothly, so if they stay correlated for about $tau$ frames, the model counts
every real piece of evidence roughly $tau$ times. Two things follow:

+ The posteriors are wildly overconfident. Probabilities of $0.9999$ mean nothing.
+ The evidence term in @eq:penalised is inflated by $tau$ relative to $lambda$, so the model
  switches far too eagerly. That is the "flicker" failure mode.

The standard fix is to *temper* the emissions with an exponent $1 \/ tau$:
$ tilde(b)_k (bold(x)_t) prop b_k (bold(x)_t)^(1 \/ tau). $ <eq:temper>
Under Viterbi, tempering is equivalent to multiplying every switch cost by $tau$. So in practice
there is *one* effective hyperparameter, $tau lambda$, and it is the thing to tune (on other
games). Subsampling to 5 fps is a cruder version of the same thing ($tau = 5$), but it costs
boundary precision. Tempering keeps the full frame rate.

This is also why letting EM *learn* $A$ freely tends to backfire. With overconfident emissions,
the maximum-likelihood transition matrix becomes less sticky to explain the flicker. Either fix
$A$, use tempered emissions in the E-step, or put a strong sticky prior on $A$
(@sec-learning).

= Inference algorithms

All recursions run in log space. A half has about 80,000 frames, so plain probabilities underflow
within a few hundred steps. With $K$ states the cost is $O(T K^2)$, which is trivial here.

== Forward–backward (posterior marginals)

$ log alpha_1 (k) = log pi_k + log b_k (bold(x)_1), quad
  log alpha_t (k) = log b_k (bold(x)_t) + lse_j [log alpha_(t-1) (j) + log A_(j k)], $
$ log beta_T (k) = 0, quad
  log beta_t (j) = lse_k [log A_(j k) + log b_k (bold(x)_(t+1)) + log beta_(t+1) (k)]. $
The log-likelihood is $log p(bold(x)_(1:T)) = lse_k log alpha_T (k)$, and
$ gamma_t (k) = p(z_t = k | bold(x)_(1:T)) = exp(log alpha_t (k) + log beta_t (k) - log p(bold(x)_(1:T))), $
$ xi_t (j, k) &= p(z_t = j, z_(t+1) = k | bold(x)_(1:T)) \
  &= exp(log alpha_t (j) + log A_(j k) + log b_k (bold(x)_(t+1)) + log beta_(t+1) (k) - log p(bold(x)_(1:T))). $

== Viterbi (the label sequence you submit)

$ delta_1 (k) = log pi_k + log b_k (bold(x)_1), quad
  delta_t (k) = log b_k (bold(x)_t) + max_j [delta_(t-1) (j) + log A_(j k)], $
$ psi_t (k) = argmax_j [delta_(t-1) (j) + log A_(j k)], quad
  hat(z)_T = argmax_k delta_T (k), quad hat(z)_(t-1) = psi_t (hat(z)_t). $

*Submit the Viterbi path, not the per-frame argmax of $gamma_t$.* The per-frame argmax
maximises expected frame accuracy and can still flicker. Viterbi returns one coherent
segmentation, which is what `spellF1` scores. Use $gamma_t$ for diagnostics: frames where
$max_k gamma_t (k)$ is low are where to look at the video to understand failures.

#note[
  *What about `X`?* On `spellF1` the scorer replaces an `X` with your previous label in that half.
  On any frame, the Viterbi label either equals the previous label, so `X` changes nothing, or is
  a switch Viterbi chose deliberately, where `X` would undo it. On `macroF1` an `X` is always a
  miss. So with an HMM, *write `X` only on ungraded frames.* Any wish to "abstain" on a graded frame
  should be expressed through the model, by raising $tau lambda$.
]

= Learning the parameters <sec-learning>

== Baum–Welch (EM) with missing data and tied states

The E-step computes $gamma_t (k)$ and $xi_t (j, k)$ as above (with tempered emissions if you use
them). The M-step maximises the expected complete-data log-likelihood. Let
$o_(t d) = bb(1)[d in O_t]$ mark observed features.

*Untied state* ($D$): weighted, masked means and variances,
$ mu_(D d) = (sum_t o_(t d) gamma_t (D) x_(t d)) / (sum_t o_(t d) gamma_t (D)), quad
  sigma_(D d)^2 = (sum_t o_(t d) gamma_t (D) (x_(t d) - mu_(D d))^2) / (sum_t o_(t d) gamma_t (D)). $

*Tied pair* ($H$, $A$ under @eq:tying): pool home frames with *mirrored* away frames,
$ mu_(H d) = (sum_t o_(t d) [gamma_t (H) x_(t d) + gamma_t (A) s_d x_(t d)])
  / (sum_t o_(t d) [gamma_t (H) + gamma_t (A)]), quad mu_(A d) = s_d mu_(H d), $
and the same for $sigma_(H d)^2$ using the residuals $x_(t d) - mu_(H d)$ for $H$ frames and
$s_d x_(t d) - mu_(H d)$ for $A$ frames.

*Bernoulli missingness:* $rho_k = sum_t gamma_t (k) m_t \/ sum_t gamma_t (k)$, tied with
$rho_A = rho_H$.

*Transitions* with a sticky Dirichlet prior $A_(j dot) tilde "Dir"(alpha bold(1) + kappa bold(e)_j)$
(the "sticky HMM" of Fox et al., 2008). The MAP update is
$ A_(j k) = (sum_t xi_t (j, k) + alpha - 1 + kappa bb(1)[j = k])
  / (sum_t gamma_t (j) + K (alpha - 1) + kappa). $
A large $kappa$ keeps the chain sticky no matter what the emissions do. The simplest robust option
is still to *fix* $A$ from outside knowledge and learn only the emissions.

== Initialisation

EM finds a local optimum, so start it somewhere sensible:

+ Run the published heuristic: nearest tracked player to the ball, majority vote over a window,
  and `D` where the ball is stationary or out of bounds.
+ Treat those labels as one-hot $gamma_t$ and do a single M-step. That gives initial emissions
  with the right meaning for each state.
+ Iterate EM to convergence, monitoring the log-likelihood (it must never decrease).

With symmetric tying (@sec-symmetry) and this initialisation, the states cannot drift away from
their intended meaning.

= Better state structure

== Control and flight sub-states <sec-substates>

The rules say a loose or contested ball *stays with the last team*, and a change needs
*controlled* possession. A single $H$ state has to explain both "a Barcelona player is dribbling"
and "a Barcelona pass is in the air with nobody near it". Those look nothing alike. The flight
frames then look like weak evidence for either team, which is exactly where spurious switches come
from.

Give each team two *sub-states* and map them to labels deterministically:
$ cal(K) = {H_"c", H_"f", A_"c", A_"f", D}, quad
  g(H_"c") = g(H_"f") = H, quad g(A_"c") = g(A_"f") = A, quad g(D) = D. $
Here c is "a player of that team has the ball close" and f is "the ball is in flight or loose
after that team last controlled it". Allowed transitions:

#table(
  columns: (auto, 1fr, auto),
  inset: 6pt,
  stroke: 0.5pt + luma(180),
  table.header([*from*], [*to*], [*football meaning*]),
  [$H_"c"$], [$H_"c", H_"f", A_"c", D$], [keep / pass or shot / tackle / ball out],
  [$H_"f"$], [$H_"f", H_"c", A_"c", D$], [in flight / received / intercepted / ball out],
  [$D$], [$D, H_"c", A_"c"$], [stoppage / restart by either team],
)
($A$ rows mirror these.) There is *no* $H_"f" -> A_"f"$ edge: the other team must *control* the
ball before possession can change, so a deflection that doesn't stop the ball keeps the path in
$H_"f"$. Emissions: the c states expect a near player (under about 1.5 m) and a ball moving with
that player. The f states expect high ball speed and no one close, and are almost uninformative
about the team. That is the point: in flight, the transitions carry the label.

Decode with Viterbi over the five sub-states and submit $g(hat(z)_t)$. Tie $A_"c"$ to $H_"c"$ and
$A_"f"$ to $H_"f"$ as in @eq:tying. Initialise the split with simple rules, e.g. ball speed above
8 m/s and no player within 2 m means f.

== Durations: semi-Markov and minimum dwell times <sec-durations>

@eq:geometric says the most likely duration of any spell is one frame. That is wrong for
possession, and it is a second reason for flicker. Two fixes:

*Sub-state chains (stays inside the HMM framework).* Replace a state by $m$ copies in a
left-to-right chain, each with self-transition $p$, leaving only from the last copy. The duration
becomes negative binomial,
$ P("duration" = d) = binom(d - 1, m - 1) (1 - p)^m p^(d - m), quad d >= m, $
which has a *hard minimum of $m$ frames* and a mode away from 1. For the "controlled possession"
rule you only need this on *cross-team* entries: route $H_* -> A_"c"$ through a chain
$A_"c"^((1)) -> dots -> A_"c"^((m))$, so a new possession has to last at least $m$ frames (e.g.
$m = 10$, i.e. 0.4 s). Passes within a team are unaffected.

*Explicit-duration HMM (HSMM).* Give each state a duration distribution $p_k (d)$ (log-normal,
or empirical from other games) and run the semi-Markov Viterbi. It costs
$O(T K^2 D_max)$, still fine with $D_max$ around a few hundred frames, or with pruning. See Yu
(2010). This is the principled version, but start with the chains, which reuse the same code.

= Features <sec-features>

Notation (metres): ball position $bold(b)_t$, home/away player positions $cal(P)^H_t, cal(P)^A_t$,
player velocity $bold(v)_i$ from smoothed position differences (the ATD only gives scalar speed).
Let $sigma_h in {+1, -1}$ be the direction home attacks in half $h$. Read it off the kickoff frame:
every player stands in their own half, so the sign of the home centroid relative to the halfway
line gives it. Each feature is oriented so that *positive means home*.

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 6pt,
  stroke: 0.5pt + luma(180),
  table.header([*\#*], [*feature*], [*symm.*], [*ball?*]),
  [1], [Nearest-player contrast: $log(d^A_t + c) - log(d^H_t + c)$, with
    $d^H_t = min_(bold(p) in cal(P)^H_t) norm(bold(p) - bold(b)_t)$, $c approx 0.5$ m],
    [anti], [yes],
  [2], [Players within 2 m of the ball, home minus away], [anti], [yes],
  [3], [*Your pressing idea.* Approach speed
    $u_i = bold(v)_i dot (bold(b)_t - bold(p)_i) \/ norm(bold(b)_t - bold(p)_i)$. Feature
    $macron(u)^A - macron(u)^H$, excluding each team's nearest player. Positive when away
    closes in and home spreads out.], [anti], [yes\*],
  [4], [Goal-side fraction: share of away players between ball and away goal, minus the same
    for home. Defenders sit behind the ball.], [anti], [yes\*],
  [5], [Team spread: $log s^H - log s^A$, with $s$ the RMS distance to the team centroid
    (outfield only). The team in possession stretches the pitch.], [anti], [*no*],
  [6], [Territory: $sigma_h (macron(x)_"all" - L\/2)$, where both blocks sit], [anti], [*no*],
  [7], [Ball velocity towards the away goal: $sigma_h v^b_x$], [anti], [yes],
  [8], [Ball speed, $log(norm(bold(v)^b) + 0.1)$], [symm], [yes],
  [9], [Mean player speed (log). Everyone walks during stoppages.], [symm], [*no*],
  [10], [Ball distance outside the pitch boundary (0 if inside)], [symm], [yes],
  [11], [Ball missing indicator $m_t$ (Bernoulli, @sec-missing)], [symm], [—],
  [12], [Time since the ball last moved faster than 1 m/s (log)], [symm], [yes\*],
)
\* also works with a Kalman-estimated ball position during short gaps (@sec-denoise).

Notes on your ideas:

- *Average team positions* (5, 6) are the most valuable kind of feature, because they *don't need
  the ball* and so keep the model informed on the 41% of frames without it. Spread is a known
  possession signal: teams stretch when attacking and compress when defending.
- *Ball speed* (8) is symmetric. It says nothing about _which_ team has the ball, but it is very
  informative about `D` (a stationary ball before a restart) and about flight vs control
  (@sec-substates). Your intuition that slow means dribble and fast means pass or clearance is
  exactly the c/f distinction.
- *Movement towards or away from the ball* (3) is a good antisymmetric feature, but it *lags*: players
  react to a turnover a few hundred milliseconds after it happens. A lagging feature pulls
  Viterbi's boundary late, and lateness is expensive on `spellF1`. Two remedies: give the
  lagging feature a time shift (use $x_(t + Delta)$, which is legal offline, with $Delta$ estimated
  on other games), or give it a lower weight via a larger variance.
- *Tracking artefacts:* track slots swap identity, which creates teleporting players. Clip
  velocities above about 11 m/s to `NaN` instead of trusting them. Use aggregate statistics
  (minimums, counts, medians) that don't care which slot is which.

*Smart Tagging as a noisy annotator.* Turn each frame's active tags into a categorical observation
$y_t$ (e.g. "Barcelona-attacking phase", "Athletic-attacking phase", "set piece starting", none).
Learn $p(y_t | z_t)$ inside EM. This is the Dawid–Skene idea of estimating an annotator's
confusion matrix without knowing the truth. EM then learns, for example, that "DEFENDING IN
DEFENSIVE THIRD, Team = Barcelona" is evidence for $A$, and how often each tag is wrong. Tags last
for whole phases, so they are extremely autocorrelated. Temper this channel heavily, or use only
their onsets.

= Denoising: wavelets or not? <sec-denoise>

Short answer: *denoise the inputs lightly, never the output labels*. The HMM already does the
temporal smoothing of labels in a principled way. Wavelets are not wrong, but they are not the
first tool to reach for:

- *Ball track: use a Kalman filter with a Rauch–Tung–Striebel smoother.* Use a constant-velocity
  state $(x, y, v_x, v_y)$. A missing observation simply skips the update step, so it handles gaps
  natively and returns an uncertainty with every estimate. Fill only short gaps (up to roughly
  0.5 s). Over long gaps the prediction is fiction, so leave them `NaN`.
- *Player velocities: use a Savitzky–Golay derivative* (window of 7–9 frames, quadratic). Plain
  finite differences of jittery positions amplify the noise.
- *Keep windows short.* Every smoothing window blurs the exact transition frames. Being 8 frames
  late already costs about 0.1 `spellF1` on a perfect submission. Smoothing also increases the
  frame-to-frame correlation in @sec-temper.
- *When wavelets would help:* their selling point is preserving sharp discontinuities, such as the
  instant of a kick, while removing noise. If the Kalman filter visibly smears kicks, wavelet
  shrinkage on the ball's velocity, or a Kalman filter that allows jumps, is a reasonable
  second step. It is low priority.

= Validation without breaking the rules <sec-validation>

The public 5 minutes cannot be used for tuning. The organisers show that tuning even a single
window size on them picks the wrong value. The ranking also has a noise band of about $plus.minus
0.04$ `spellF1`. So the tuning loop has to live somewhere else:

+ *Labelled other games.* The rules allow external data about other games, if the organisers can
  access it and the technical document explains it. Candidates to check (licence, access, and how
  each defines possession):
  - Metrica Sports sample data (#link("https://github.com/metrica-sports/sample-data")[github.com/metrica-sports/sample-data]):
    three anonymised games, 25 fps tracking plus event data, from the same vendor. At least one is
    in the same EPTS format, and `kloppy` loads it. Derive possession labels from the event
    stream.
  - The DFL/Sportec open dataset (Bassek et al., 2025): Bundesliga tracking at 25 fps plus events,
    with a per-frame ball-possession and ball-status flag. That is essentially the same kind of
    label as our ground truth ("dead where the tracking says dead, otherwise the owning team").
    `kloppy` has a loader.
+ *Simulate our noise on the clean games.* Their tracking is cleaner than our automatic ATD.
  Delete the ball in bursts until about 41% of frames are missing, preferentially when players are
  crowded around it. Fragment and swap track slots. Add positional jitter. Tune $tau lambda$, the
  sub-state thresholds and $m$ on *this* corrupted data. `evaluation/run_evaluation.py --gt <path>`
  accepts any `frame,label` CSV, so you can score with the official metric.
+ *Unsupervised checks on BAR-ATH.* Possession count and duration histogram per half (compare with
  the other games), the share of `D`, and video overlays of the decoded label via the homography.
  Looking at the video to understand failures is fine. Correcting labels by hand is not.
+ *Public window: last, once, for correctness only.* Frames aligned, labels oriented, file valid.

Since `spellF1` is noisy, compare model variants with `--bootstrap 1000` on the other games
before believing a difference.

= Implementation sketch

Pure NumPy is enough, and about 100 lines covers everything above. `hmmlearn` is convenient but
handles neither `NaN` features nor tied mirrored states, so it is easier to write the recursions
yourselves.

```python
import numpy as np
from scipy.special import logsumexp

def log_emissions(X, mu, var, tau=1.0):
    """X: (T, F) with NaN = missing; mu, var: (K, F). Returns (T, K), tempered by 1/tau."""
    obs = ~np.isnan(X)
    Xf = np.where(obs, X, 0.0)
    ll = -0.5 * (np.log(2 * np.pi * var)[None] + (Xf[:, None, :] - mu[None]) ** 2 / var[None])
    return np.where(obs[:, None, :], ll, 0.0).sum(-1) / tau     # missing features drop out

def forward_backward(logB, logA, logpi):
    T, K = logB.shape
    la, lb = np.empty((T, K)), np.zeros((T, K))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    loglik = logsumexp(la[-1])
    gamma = np.exp(la + lb - loglik)
    xi_sum = np.exp(logsumexp(la[:-1, :, None] + logA[None]
                              + (logB[1:] + lb[1:])[:, None, :], axis=0) - loglik)
    return gamma, xi_sum, loglik

def viterbi(logB, logA, logpi):
    T, K = logB.shape
    delta, back = logpi + logB[0], np.empty((T, K), dtype=np.int8)
    for t in range(1, T):
        s = delta[:, None] + logA            # s[j, k]: best path ending j -> k
        back[t] = s.argmax(0)
        delta = s.max(0) + logB[t]
    z = np.empty(T, dtype=np.int8)
    z[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        z[t - 1] = back[t, z[t]]
    return z
```

The full pipeline per half: build features, then initialise from the heuristic, then run EM
(masked, tied, tempered), then Viterbi, then map sub-states to `H`/`A`/`D`. Write `X` outside
`games.half_video_blocks("BAR-ATH")` and run `validate_submission.py`. The Python loops take a few
seconds per half. Use `numba` if you sweep many hyperparameters.

#warn[
  *Data handling.* The terms forbid uploading the match data, or extracts of it, to cloud-hosted
  LLMs. Keep the feature tables, tracking rows and decoded labels off them. Code, error messages
  and aggregate scores are fine.
]

= Suggested order of work

+ *Baseline HMM.* Three states, features 1, 5, 8, 9, 11, fixed sticky $A$, no EM. Check it beats
  the published floor (`spellF1` 0.2433) on simulated-noise other games.
+ *Calibrate.* Add tempering, then grid-search $tau lambda$ on other games against `spellF1`.
  This is probably the single biggest gain.
+ *EM with tying and the missingness channel.* Learn emissions on BAR-ATH itself (unsupervised,
  which is legal) after initialising from the heuristic.
+ *Control/flight sub-states and minimum dwell on cross-team entries.*
+ *Hybrid emissions.* Train a per-frame classifier (gradient boosting or a small network) on
  other games to output $p(z_t | bold(x)_t)$, and plug it into the HMM as a scaled likelihood
  $b_k (bold(x)_t) prop p(z_t = k | bold(x)_t) \/ p(z_t = k)$ (the hybrid HMM/ANN approach from
  speech recognition, Bourlard & Morgan 1994). This drops the Gaussian assumption entirely while
  keeping the same decoder.
+ *Smart Tagging channel, lag-shifted features, HSMM durations.* Do these if time allows.

= Other directions worth knowing about

- *Penalised change-point detection* (e.g. PELT, Killick et al. 2012) directly on the evidence
  stream. It is equivalent to @eq:penalised but optimises the segmentation without a generative
  model.
- *Segment-then-classify:* detect dead-ball stretches first (they are much easier: stationary
  ball, slow players, ball out), then decode only $H$/$A$ inside live stretches. A restart's
  possessor is often obvious from who stands over the ball.
- *Sequence models with a CRF or HMM head* (BiLSTM or temporal convolution plus Viterbi), trained
  on other games. They are more powerful but need the noise simulation to transfer.
- *Better ball recovery from video:* a ball detector run on frames where the ATD lost the ball
  would attack the 41% gap directly. It is expensive, but it is the dataset's biggest weakness.

= References

- L. R. Rabiner (1989). A tutorial on hidden Markov models and selected applications in speech
  recognition. _Proc. IEEE_ 77(2).
- E. B. Fox, E. B. Sudderth, M. I. Jordan, A. S. Willsky (2008). An HDP-HMM for systems with
  state persistence. _ICML_.
- S.-Z. Yu (2010). Hidden semi-Markov models. _Artificial Intelligence_ 174(2).
- H. Bourlard, N. Morgan (1994). _Connectionist Speech Recognition: A Hybrid Approach_. Kluwer.
- A. P. Dawid, A. M. Skene (1979). Maximum likelihood estimation of observer error-rates using
  the EM algorithm. _J. R. Stat. Soc. C_ 28(1).
- R. Killick, P. Fearnhead, I. A. Eckley (2012). Optimal detection of changepoints with a linear
  computational cost. _JASA_ 107(500).
- M. Bassek et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer.
  _Scientific Data_. (Verify the citation and licence before relying on it.)
