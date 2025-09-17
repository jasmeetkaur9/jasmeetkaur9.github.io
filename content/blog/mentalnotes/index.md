---
title: "Pre-training in Reinforcement Learning: Data, Curosity and Structure"
date: 2025-09-16
lastmod: 2025-09-16
tags: ["Reinforcement Learning", "Research Practice"]
author: "Jasmeet Kaur"
description: "Data, Curosity and Structure."
summary: "A short review highlight on recent works."
showToc: false
disableAnchoredHeadings: false
math: true
---

# Introduction

Label-free learning has enabled recent progress in learning from large
amounts of data. Advancements in Large Language Models showcase the
effectiveness of learning the generative process from huge amounts of
text data. This also pushed learning without labels or learning from
auxiliary labels in some of the sub-fields of Reinforcement Learning
([16], [18], [13], [14], etc.). This article provides a focused study of different
works unified by data-driven objectives. It filters the entire
discussion of using data into two different perspectives. First is
learning driven by curiosity to explore the dataset, and the second is
to learn the underlying core information.

This discussion is motivated by works that focus on data invariant to
task-relevant information ([27], [22], [18], [12],
etc.). This discussion highlights the type of underlying structure that
exists in the data, what are the ways to learn about this structure, and
how to adapt these insights to solve downstream tasks with minimal
intervention. The aim is to give an integrated understanding of these
works while providing insights that pave the way for future research.

Prevailing practices demonstrate that the availability of large data has
made its use and scientific consideration inevitable. But not all data
is good. A lot of data used for training RL agents comes from unknown
tasks, suboptimal, and incomplete interactions
([10], [16], [13]). Nevertheless, methods to pre-train using this
data are attractive because solving new tasks from scratch is not
efficient. Utilizing data is beneficial in cases where interacting with
the environment during deployment is not possible.

Most of the papers covered under the discussion presented here are works
in unsupervised and self-supervised pre-training. Some works that lay
the foundation of such approaches have also been described in the
relevant sections. This analysis has been divided into four main
sections. Each section covers a pre-training objective that is motivated
by the two main factors described above. Most recent works cover these
two motivations independently. This article does not propose a novel
method to solve any problem, but offers an organized discussion.

The focus for each section is as follows: exploring datasets for solving
downstream tasks with minimal or no adaptation, learning representations
that capture structure in environmental dynamics, learning
representations that capture similar functional behavior within an
environment, and learning representations that capture structure between
similar and dissimilar behaviors. The objectives include maximizing
entropy, finding temporal structure, learning successor features,
learning with contrastive objectives, and bisimulation metrics. These
methods provide overlapping answers to some important research
questions. These include determining the right type of representations
to learn from data and understanding the trade-offs of different
pre-training objectives for efficient downstream adaptation.

# Problem Statement

All problem settings presented operate in an MDP (Markov-decision
process) framework or some extension of this framework. The environment
is described as an MDP $M$ with state space $\mathcal{S}$, action space
$\mathcal{A}$, transition dynamics $P$, and discount factor $\gamma$. A
policy $\pi(a|s)$ describes what action to take given a state $s$. Each
task is defined via a reward function $r$ that maps state-action pairs
to real-valued rewards. In some cases, an offline dataset of
trajectories is available. This dataset, denoted as $\mathcal{D}$,
consists of $N$ trajectories, where each trajectory $\tau_i$ is a
sequence of tuples $(s_t^i, a_t^i, r_t^i, s_{t+1}^i)$ for $t = 0$ to
$T_i$, with $T_i$ being the horizon length of the trajectory. These
trajectories can be collected using any policy, whether optimal or
suboptimal, and may involve unknown reward functions.

# Methods

This section focuses on four different approaches for representation
learning in RL. These representations allow for few-shot adaptations to
downstream tasks. The section describes the different pre-training
objectives, adaptation or fine-tuning strategies for new tasks,
implementation choices, and open questions. The aim is to cover the
strengths and impact of these contributions.

## Exploration-based Representations

Incorporating exploration as an objective can be used to learn diverse
skills from data. Exploration-based works focus on invariance to task
rewards. This can be done for either the offline data available or for
gathering new data through online interactions. Exploring a domain is
harder than training a policy to maximize reward in an environment
([27]). A pre-training exploratory policy can be used
for zero-shot RL ([27]). This can be done by training
an exploratory policy that maximizes the entropy of a marginal state
visitation distribution for a finite horizon $T$. This distribution is
defined as: $$\begin{equation}
d_{T,\pi}(s) = \frac{1}{T} \sum_{t=0}^{T} d_{t,\pi}(s),
\end{equation}$$ where $d_{t,\pi}(s)$ denotes the $t$-step state
distribution when following policy $\pi$.

Intuitively, the entropy objective helps move towards the states that
the policy has not encountered in the episode. But learning an
exploratory policy is not enough, and disentangling exploration from
underlying dynamics is important. One way to do this is to train an
ensemble of policies and use the exploratory policy to find actions in
the environment when the ensemble fails to agree on an action
([27]). The objective maximized over a horizon T,
becomes: $$\begin{align}
R_T(\pi) 
&= \mathbb{E}_{\mathcal{M} \sim \hat{P}(\mathcal{M})}
  \left[ \emph{H}( d_{T, \pi}(s)) \right] \\
&= \mathbb{E}_{\mathcal{M} \sim \hat{P}(\mathcal{M})}
  \left[
    \emph{H}\!\left( \frac{1}{T} \sum_{t=0}^{T} d_{t,\pi}(s) \right)
  \right].
\end{align}$$

The expectation is over different MDPs sampled from a distribution. This
objective learns a policy that equally visits all the states during the
episodes ([27]). Since sampling from the
state-visitation distribution is not trivial, entropy can be computed by
a particle-based k-nearest neighbor approximation
([27], [23]). For practical
implementation, ([27]) treats the objective as a
reward to be optimized for by training an agent with PPO.

Evidently, the Max-Entropy trained policy shows a small generalization
gap on the unknown tasks for effective zero-shot generalization. In
Zero-shot generalization, a policy is trained on a set of tasks and
tested on unknown tasks. The generalization quality of this policy is
tested by training agents on 200, 500, and 1000 of Procgen's Maze,
Jumper, and Miner ([27]). Such exploration only
learns reward invariance. A promising direction is how well such a
policy performs when combined with tasks that require invariance to
environment features. Another is how to use these environment
interactions more efficiently.

A sample-efficient approach with a similar motivation can be used for
representation learning. Learn a visual representation without task
rewards in a self-supervised way. The pre-training phase has two parts.
First, learn a set of prototypical embeddings that form the fundamental
basis of a low-dimensional latent space. Second, use these prototypes to
maximize an entropy-based intrinsic reward, encouraging exploration of
the environment. The intrinsic reward is similar to the one in
[27], but instead of states, it uses encoded
observations. [23] shows the performance of such
an approach on downstream tasks, and the intrinsic reward is appended
with the task reward for DeepMind Control using SAC.

Both methods assume interaction with the environment in the pre-training
phase, with no fine-tuning for downstream adaptations. But diverse
behavior from offline data can also be learnt in a task-agnostic way.
The idea is to optimize for diverse temporal structure in the
pre-training phase. Learn a state representation $\phi$ that captures
the temporal distance using unlabeled trajectories by using the
equivalence between temporal distance and the optimal goal-conditioned
value function. This representation is used to learn a policy that spans
the latent space and captures diverse skills in the offline data
([16]). Few-shot adaptation to the downstream can be
achieved by learning the task-dependent latent variable. Box 1.
summarizes the pre-training objectives that focus on capturing diversity
in data and respective methods for downstream evaluation.

Curiosity-driven objectives can be used to improve online data
collection and online exploration for new tasks. Such objectives are
effective in solving sparse reward tasks at test time. The unlabeled
data can be labeled with optimistic rewards with RND
([4]) to guide exploration to unknown states. An RL
agent trained with this data can be used to collect data through online
interactions and update the reward model from this online experience.
This offers an iterative approach for fine-tuning the agent with updated
rewards. In [13], the evaluation for this method showed
rapid exploration in challenging sparse-reward domains such as the
AntMaze domain, Adroit hand manipulation, and a visually simulated
robotic manipulation domain.

Such an objective built upon offline data can be used to improve online
exploration. The key idea of this is to extract the low-level skills
using a variational encoder. The posterior is conditioned by a prior to
stay close to the offline dataset. Learn a low-level policy to take
actions around the dataset trajectories and learn a high-level policy
through online interactions. This method is effective for tasks that
need exploration to be solved ([22]). Both these
approaches implicitly capture invariance in the environment dynamics
while using unlabeled data to guide exploration.

## Structured Representations

Successor Features present a way to learn representations that
explicitly exploit structure in the environment. The structure across
different tasks remains the same when the environment dynamics are the
same. This section focuses on one way to learn such invariance.

### Successor Features

Successor Features were introduced to do transfer in RL by exploiting
the underlying structure in dynamics to perform well on new tasks
([2]). If the expected one-step reward associated
with transition $(s, a, s')$ is decomposed into features represented by
$\phi$ and weights $w$, such a representation can be used to compute the
$Q$-value function. This $Q$-value function can be decomposed into the
task-relevant information and cumulant $\psi$, which is called Successor
Features. Once Successor Features are learnt for different tasks,
Generalized Policy Iteration is used to learn a policy that does as good
as any other policy on this new task ([2]).

One of the issues with an earlier application of this idea is that with
SF and GPI, no task-specific information is shared to learn policies
([3]). SF$\&$GPI can be combined with the UVFAs
([19]) to learn the value functions over the task
encodings and learn a span over the policies. While this offers good
generalization, the performance of the policies on a new task depends on
how the policies are sampled. An alternative approach for learning
shared knowledge across different tasks is using an attention head to
learn successor features ([5]). A more robust way
to generate approximations of successor features is to learn a
distribution of the returns ([6]). Finally, such
structural decomposition can be used to learn to adapt to tasks with
non-Markovian rewards. The successor representations capture the
expected steps to reach a state from the current state for the first
time ([15]).

### Successor Measure

The successor feature decomposition can be extended to state-visitation
distributions. Such decompositions learn the structure of the solutions
to tasks. The idea is to decompose the Q-function into the successor
measure and the reward. The successor measure encodes the probability of
going to the next state under the policy ([21]). The
entire family of policies can be parametrized with z, and the Q-function
is used to learn the optimal policies. Learning the successor measure
representation for all z allows for adaptation to any reward function.
For each $z$, find $d \times (S \times A)$-matrices $F_z$ and $B$ such
that the matrix $M^{\pi}_z$ is $M^{\pi}_z = F_z^{\top} B$. Then, for a
reward function $r$, the action-value function is given by
$Q^{\pi}_{z,r} = F_z^{\top} B\,r$ ([21]).

Learning the span of policies does not have to be restricted to optimal
policies. Successor Measure can be used to learn representations in the
entire space of policies. Following from the Bellman Flow equations of
successor measure, the solutions to these exist if it is an affine
combination. For practical implementation, this constrained optimization
problem can be transformed into a single-player game in
[1].

Successor Features and Successor Measure offer a concise way to
represent the underlying structure explicitly. Their ability to transfer
to downstream tasks relies on how the space of policies is encoded with
this information, making it an important area of closer examination.

## Bisimulation-based Representations

Bisimulation metrics capture invariance to task-irrelevant features.
They group functionally similar states in a latent space. Capturing the
behavioral similarity in an MDP can be useful for generalization over
large state spaces ([8]). In theory, Bisimulation
defines an equivalence relation over the state-space. Partitioning the
state-space under this relation is difficult for practical application.
To deal with this, a semi-metric over this relation is defined to
capture how similar two states are. This gives a class of bisimulation
metrics. Bisimulation metrics are used to learn representations from
image observations in reinforcement learning.

Putting bisimulation to practical use has come a long way. Issues
related to its efficient computation and upper bounds for arbitrary
policies proved worthwhile. A sampling-based approximation to the joint
distribution of the transition dynamics made it easier to implement for
learning approximation to a large-state space ([7]). A
common problem that emerges is that of representation collapse, where
two dissimilar states end up with the same representations. One way to
mitigate this is to use cosine distance to measure the distance between
two latent states ([25]). Efficiently computing the
approximation to the Wasserstein distance is an open challenge.
$$\begin{align}
\cos_{\phi}(x, y) 
&= 1 - \frac{\phi(s)^\top \phi(t)}
        {\|\phi(s)\| \cdot \|\phi(t)\|},
\label{eq:cosine} \\[1ex]
F^{\pi}_{\cos\phi}(s, t) 
&= \big| r^{\pi}_s - r^{\pi}_t \big|
   + \gamma \, \mathbb{E}_{s' \sim \hat{P}^{\pi}_s, \; t' \sim \hat{P}^{\pi}_t}
      \big[ \cos_{\phi}(s', t') \big].
\label{eq:simsr}
\end{align}$$

Aforementioned methods provide objectives to learn effective
representations efficiently using bisimulation metrics. The following
subsections describe three domains where bisimulation metrics have been
adapted for specific problems. More important to this discussion are
works on learning action representations and offline data with no action
information. In general, these works show the effectiveness of capturing
this behavioral similarity in different settings. These adaptations have
been successful in solving complex tasks such as robot manipulation
([10], [20]).

### Goal-conditioned Bisimulation Relations

Policy bisimulation can result in the effective transfer of skills
across analogous tasks. Given an MDP $M$, *goal-conditioned bisimulation
relation* $\mathcal{B}$ can be defined ([10]). For
all state - goal pairs $(s, g_s), (t, g_t) \in S \times G$ that are
equivalent under $\mathcal{B}$, the following conditions hold:
$$\begin{align}
    R(s, a, g_s) &= R(t, a, g_t), \quad \forall a \in A \label{eq:reward-cond} \\
    P(G \mid s, a) &= P(G \mid t, a), \quad \forall a \in A, \; G \subseteq S. \label{eq:transition-cond}
\end{align}$$

For practical implementation, an *on-policy* version of this relation
gives rise to a paired-state metric: $$\begin{align}
    d\big((s,g_s), (t,g_t)\big) 
    &= \big| R(s, \pi(s,g_s), g_s) - R(t, \pi(t,g_t), g_t) \big| \notag \\
    &\quad + W_2 \Big( P(\cdot \mid s, \pi(s,g_s)), \;
    P(\cdot \mid t, \pi(t,g_t)) \Big), \label{eq:metric}
\end{align}$$ where $\pi$ is the goal-conditioned policy. The following
objective can be used to learn state-goal representations and train an
offline goal-conditioned policy. [10] used a
dataset collected using a noisy expert to show skill transfer in
manipulation tasks using such a bisim relation. $$\begin{align}
L_{\varphi} &= \biggl( \bigl\|
    \phi(s, g_s) - \phi(t, g_t)
\bigr\|_1
- \bigl\| R_s - R_t \bigr\|_2
- \gamma \bigl\|
    \overline{\phi}(s', g_s') - \overline{\phi}(t', g_t')
\bigr\|_2
\biggr)^2
\end{align}$$

A bisimulation objective can be appended with a forward model
error-based intrinsic reward to improve exploration. This enhances
exploration in the latent space for learning a bisimulation-based
representation. This has been empirically shown to learn robust state
representations for downstream goal-conditioned tasks
([11]).

### Learning from Offline Data

$\pi$-bisimulation-based representations can be learnt using an offline
dataset. Learning from an offline dataset is notoriously difficult using
function approximation. The representations learnt using the
bisimulation metric can result in a value function that does not
diverge. These representations stabilize TD-learning and are Bellman
complete ([17]). [17] defines a
$\pi$-bisimilarity kernel $k^{\pi_e} : S \times S \to \mathbb{R}$ for
pairs of state-actions and under $\pi_e$: $$\begin{align}
k^{\pi_e}(s,a_s; t,a_t) 
&= k_1(s,a_s; t,a_t) 
   + \gamma \, k_2\big(k^{\pi_e}\big)\big(P^{\pi_e}(\cdot|s,a_s), P^{\pi_e}(\cdot|t,a_t)\big),
\label{eq:kpi} \\[1ex]
\end{align}$$ Here, $k_1$ measures short-term similarity based on the
rewards received. $k_2$ measures long-term similarity between
probability distributions by evaluating similarity between samples of
the distributions according to $k^{\pi_e}$.

To use the bisimulation metric for offline datasets, it must learn from
an incomplete dataset with missing transitions. While Implicit Q
learning can be used to mitigate this problem
([10]), the error in the estimated value of bisim
operator can be reduced by direct use of an expectile operator
([26]). $$\begin{align}
F^{\pi_\beta} \phi^{\pi_\beta}(s, t) 
&:= \arg\min_{\phi^{\pi_\beta}} 
\;\; \mathbb{E}_{a_s \sim \pi_\beta(\cdot|s), \; a_t \sim \pi_\beta(\cdot|t)} \bigg[
    \tau \, [\hat{\varepsilon}]_+^2 
    + (1 - \tau) \, [-\hat{\varepsilon}]_+^2
\bigg], \label{eq:expectile_operator} \\
\hat{\varepsilon} 
&= \mathbb{E}_{s' \sim T^{\pi}(s), \; t' \sim T^{\pi}(t)} 
\Big[ \; \big| r(s,a_s) - r(t,a_t) \big| 
    + \gamma \, \overline{\phi}^{\pi_\beta}(s', t') 
    - \phi^{\pi_\beta}(s,t) \;\Big].
\label{eq:residual}
\end{align}$$ For real-world deployment of bisimulation-based
representations, naively training with adversarial states and actions
for such cases does not transfer well to the downstream tasks
([24]). Learn robust representation with perturbed
states and goals closer to the negative samples with a contrastive
objective.

### Action Invariance

Bisimulation-based metrics enable capturing invariance in behavior by
learning action representations. Such long-term action representations
can be learnt in a self-supervised way using bisimulation
([20]). The state-conditional action chunk (denoted by c)
bisimulation metric is a function
$d: \mathcal{S} \times \mathcal{C} \times \mathcal{C} \to \mathbb{R}_{\ge 0}$
such that $$\begin{align}
d(c_i, c_j \mid s_t) 
&= R^{c_i}_{s_t} - R^{c_j}_{s_t} 
   + \gamma \, W_2 \big( P^{c_i}_{s_t}, P^{c_j}_{s_t}; d_{c} \big),
\label{eq:bisim_metric}
\end{align}$$ where $d$ is a pseudometric and $W_2$ is the 2nd
Wasserstein distance between two distributions. Here, $R^c_{s_t}$
represents the cumulative discounted reward for executing chunk $c$
starting at $s_t$ and $P^c_{s_t}$ represents the distribution of
$s_{t+k}$ after executing $c$ from $s_t$. Such representations are
effective for solving complex tasks. This is experimentally shown with
7DOF ARM Control ([20]). Learning action representations is
not restricted to online interactions but can also be achieved from an
offline dataset. To stabilize learning close to the behavioral policy
and and mitigate distribution mismatch, the following objective can be
used ([9]): $$\begin{align}
L(\phi) 
&= \mathbb{E}_{s, a_i, r \sim \mathcal{D}, \; a_j \sim \mathcal{A}} 
   \Big[ \big\| \phi(a_{i}) - \phi(a_{j}) \big\|_1 - \hat{d}(a_i, a_j \mid s) \Big]^2,
\label{eq:mod_obj}
\end{align}$$ where $$\begin{align}
\hat{d}(a_i, a_j \mid s) 
&= \big| r_{i} - \hat{R}(s, \phi(a_{j})) \big|
   + \gamma \, W_2 \big( \hat{P}(\cdot \mid s,  \phi(a_{i})), \hat{P}(\cdot \mid s,  \phi(a_{i})) \big)
   + p \cdot \hat{I}_\beta(a_j \mid s).
\label{eq:estimated_d}
\end{align}$$ Here, $d(a_i, a_j \mid s)$ is an estimate of the true
distance between actions conditioned on the same state $s$. $\hat{R}$
and $\hat{P}$ are the learned reward and transition models, trained
separately. $\hat{I}_\beta(a_j \mid s)$ is a trainable model predicting
whether $a_j$ is out-of-distribution.

While the method above explicitly handles the behavioral distribution, a
similar objective for single-step action representation, denoted by
$\phi$, can be learnt independent of data-collecting policy. Such an
action-bisimulation metric is defined as: $$\begin{align}
d_{\text{a-bisim}}(s_i, s_j, \phi, \varphi) 
&= (1-c) \cdot \| \phi_(s_i) - \phi_(s_j) \|_1 
   + c \cdot \mathbb{E}_{a \sim \mathcal{U}(\mathcal{A})} 
       \Big[ W_1 \big( f(\varphi(s_i), a), f(\varphi_(s_j), a) \big) \Big],
\label{eq:da_bisim}
\end{align}$$ where $c \in [0,1]$ balances the contributions of the
state and action-conditional terms, $W_1$ denotes the 1st Wasserstein
distance and $f$ is the trained forward model. The state representations
are learnt by minimizing the $L_1$ distance between embedded
representations $\varphi_(s_i)$ and $\varphi_(s_j)$ to the action-bisim
metric ([18]): $$\begin{align}
L(\mathcal{D}) 
&= \frac{1}{N} \sum_{s_i, s_j \sim \mathcal{D}} 
   \Big| \| \varphi_(s_i) - \varphi_(s_j) \|_1 
   - d_{\text{a-bisim}}(s_i, s_j, \psi, \varphi) \Big|.
\label{eq:da_bisim_loss}
\end{align}$$

Bisimulation can effectively capture the similarity between states. It
can be adapted for offline datasets, online exploration, and learning
action representations.

## Contrastive Learning based Representations

Contrastive Learning objectives have been used for unsupervised RL. It
is a framework to learn representations by exploiting the structure
between similar and dissimilar pairs of input. This can be achieved in
an unsupervised way by performing a dictionary lookup task wherein the
positives and negatives represent a set of keys with respect to a query
(or an anchor). There are many ways in which such an objective can be
captured ([12]). One of the most adapted ones is the
InfoNCE loss function.

$$\begin{align}
\mathcal{L}_{\text{contrastive}} &= 
- \log \frac{\exp(\text{sim}(q, k^+)/\tau)}
       {\sum_{k \in \mathcal{K}} \exp(\text{sim}(q, k)/\tau)}
\end{align}$$

Here, $q$ is the query embedding, $k^+$ is the positive key embedding,
$\mathcal{K}$ is the set of all keys in the mini-batch (positives and
negatives), $\text{sim}(x, y) = \frac{x \cdot y}{\|x\| \|y\|}$ denotes
the cosine similarity, and $\tau$ is the temperature hyperparameter.

Contrastive Learning can be used to train an RL agent over
representations learnt from image-based observations. [12]
uses a modification of the function in Box 7 to train agents for
handling both continuous and discrete action spaces. Contrastive
Learning can be used to improve the robustness of bisimulation
representation for transfer to downstream tasks. For this,
[24] uses it for perturbing the negative samples.
$P_{trb}$ in the following objective, Eq. 38, denotes the learnt
perturbation parametrized by $\theta$. $$\begin{align}
\mathcal{L}(\theta) &= - \phi \big( P_{trb}^i(s), P_{trb}^i(g) \big)^\top \phi\big( \langle s, g \rangle\big)^{-}
\end{align}$$

Finally, Contrastive Learning is effective for representations over
offline image data. Learn representation over states using a
goal-conditioned offline pre-training objective as in [14]. The
objective minimizes the distance between the goal-conditioned
state-occupancy distribution of the policy and the data distribution.
The dual of this objective yields a contrastive RL objective. The
objective is: $$\begin{align}
\max_{\pi_T, \phi} \; & \mathbb{E}_{\pi_T} \Bigg[ \sum_t \gamma^t r(o; g) \Bigg] \nonumber \\
& - D_{\mathrm{KL}}\big(d_{\pi_T}(o, a_T; g) \,\|\, d_D(o, \tilde{a}_T; g)\big),
\end{align}$$ where $d_{\pi_H}(o, a_H; g)$ is the distribution over
observations and actions visited by the policy $\pi_H$ conditioned on
goal $g$. $d_D(o, \tilde{a}_H; g)$ is the distribution over observations
and dummy actions $\tilde{a}_H$ in the dataset $D$, conditioned on goal
$g$. This method has been effective for zero-shot generalization in
goal-conditioned reinforcement learning.

### Conclusion

This article provides an overview of recent research trends in RL. It
looks at different works using unsupervised and self-supervised methods
of learning in RL and provides a unification of the core objectives. It
covers approaches based on exploration, successor features,
bisimulation, and contrastive learning, highlighting the importance of
curiosity and structure in RL. This synthesis of reviewed work indicates
certain key insights. Curiosity-based approaches capture a notion of
diversity and learns to disentangle this diversity for downstream tasks.
In general, how exploration is related to representation learning for
downstream performance deserves additional study. On the other other,
Successor Features and Successor Measures capture structure explicitly.
Structure in the environment evidently dictates the structure in the
solution space, but this interdependence needs to be looked into
further. Bisimulation learns behavioral similarity, but computing the
metric using probability distributions is a bottleneck. Contrastive
Learning provides a sample-efficient way to learn representations that
are close to positive samples and distant from negative samples.

### Use of Generative AI

LaTeX code for equations has been modified from templates generated with
the help of ChatGPT (OpenAI). In addition, it has been used to assist in
finding word phrasing. The text was proofread using Grammarly's basic
grammar and style suggestions, without the use of AI-generated content.

### Citation 
If you found this useful in your academic work, please cite this using:

Kaur, Jasmeet. Pre-Training in Reinforcement Learning: Data, Curiosity and Structure. https://jasmeetkaur9.github.io/blog/mentalnotes/.


### References 

1. Siddhant Agarwal, Harshit Sikchi, Peter Stone, and Amy Zhang. Proto
successor measure: Representing the behavior space of an rl agent.
*arXiv preprint arXiv:2411.19418*, 2024.

2. André Barreto, Will Dabney, Rémi Munos, Jonathan J Hunt, Tom Schaul,
Hado P van Hasselt, and David Silver. Successor features for transfer in
reinforcement learning. *Advances in neural information processing
systems*, 30, 2017.

3. Diana Borsa, André Barreto, John Quan, Daniel Mankowitz, Rémi Munos,
Hado Van Hasselt, David Silver, and Tom Schaul. Universal successor
features approximators. *arXiv preprint arXiv:1812.07626*, 2018.

4. Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration
by random network distillation. *arXiv preprint arXiv:1810.12894*, 2018.

5. Wilka Carvalho, Angelos Filos, Richard L Lewis, Satinder Singh, et al.
Composing task knowledge with modular successor feature approximators.
*arXiv preprint arXiv:2301.12305*, 2023a.

6. Wilka Carvalho Carvalho, Andre Saraiva, Angelos Filos, Andrew Lampinen,
Loic Matthey, Richard L Lewis, Honglak Lee, Satinder Singh, Danilo
Jimenez Rezende, and Daniel Zoran. Combining behaviors with the
successor features keyboard. *Advances in neural information processing
systems*, 36: 9956--9983, 2023b.

7. Pablo Samuel Castro, Tyler Kastner, Prakash Panangaden, and Mark
Rowland. Mico: Improved representations via sampling-based state
similarity for markov decision processes. *Advances in Neural
Information Processing Systems*, 34: 30113--30126, 2021.

8. Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite
markov decision processes. In *UAI*, volume 4, pages 162--169, 2004.

9. Pengjie Gu, Mengchen Zhao, Chen Chen, Dong Li, Jianye Hao, and Bo An.
Learning pseudometric-based action representations for offline
reinforcement learning. .

10. Philippe Hansen-Estruch, Amy Zhang, Ashvin Nair, Patrick Yin, and Sergey
Levine. Bisimulation makes analogies in goal-conditioned reinforcement
learning. In *International Conference on Machine Learning*, pages
8407--8426. PMLR, 2022.

11. Mete Kemertas and Tristan Aumentado-Armstrong. Towards robust
bisimulation metric learning. *Advances in Neural Information Processing
Systems*, 34: 4764--4777, 2021.

12. Michael Laskin, Aravind Srinivas, and Pieter Abbeel. Curl: Contrastive
unsupervised representations for reinforcement learning. In
*International conference on machine learning*, pages 5639--5650. PMLR,
2020.

13. Qiyang Li, Jason Zhang, Dibya Ghosh, Amy Zhang, and Sergey Levine.
Accelerating exploration with unlabeled prior data. *Advances in Neural
Information Processing Systems*, 36: 67434--67458, 2023.

14. Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani,
Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and
representation via value-implicit pre-training. *arXiv preprint
arXiv:2210.00030*, 2022.

15. Ted Moskovitz, Spencer R Wilson, and Maneesh Sahani. A first-occupancy
representation for reinforcement learning. *arXiv preprint
arXiv:2109.13863*, 2021.

16. Seohong Park, Tobias Kreiman, and Sergey Levine. Foundation policies
with hilbert representations. *arXiv preprint arXiv:2402.15567*, 2024.

17. Brahma S Pavse, Yudong Chen, Qiaomin Xie, and Josiah P Hanna. Stable
offline value function learning with bisimulation-based representations.
*arXiv preprint arXiv:2410.01643*, 2024.

18. Max Rudolph, Caleb Chuck, Kevin Black, Misha Lvovsky, Scott Niekum, and
Amy Zhang. Learning action-based representations using invariance.
*arXiv preprint arXiv:2403.16369*, 2024.

19. Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal
value function approximators. In *International conference on machine
learning*, pages 1312--1320. PMLR, 2015.

20. Lei Shi, HAO Jianye, Hongyao Tang, Zibin Dong, and Yan Zheng.
Self-supervised bisimulation action chunk representation for efficient
rl. In *Neurips Safe Generative AI Workshop 2024*, 2024.

21. Ahmed Touati and Yann Ollivier. Learning one representation to optimize
all rewards. *Advances in Neural Information Processing Systems*, 34:
13--23, 2021.

22. Max Wilcoxson, Qiyang Li, Kevin Frans, and Sergey Levine. Leveraging
skills from unlabeled prior data for efficient online exploration.
*arXiv preprint arXiv:2410.18076*, 2024.

23. Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto.
Reinforcement learning with prototypical representations. In
*International Conference on Machine Learning*, pages 11920--11931.
PMLR, 2021.

24. Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang,
and Wenjie Ruan. Representation-based robustness in goal-conditioned
reinforcement learning. In *Proceedings of the AAAI Conference on
Artificial Intelligence*, volume 38, pages 21761--21769, 2024.

25. Hongyu Zang, Xin Li, and Mingzhong Wang. Simsr: Simple distance-based
state representations for deep reinforcement learning. In *Proceedings
of the AAAI conference on artificial intelligence*, volume 36, pages
8997--9005, 2022.

26. Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam,
Remi Tachet des Combes, and Romain Laroche. Understanding and addressing
the pitfalls of bisimulation-based representations in offline
reinforcement learning. *Advances in Neural Information Processing
Systems*, 36: 28311--28340, 2023.

27. Ev Zisselman, Itai Lavie, Daniel Soudry, and Aviv Tamar. Explore to
generalize in zero-shot rl. *Advances in Neural Information Processing
Systems*, 36: 63174--63196, 2023.

