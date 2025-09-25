---
title: "Pre-training in Reinforcement Learning: Data, Curiosity and Structure"
date: 2025-09-16
math: true
lastmod: 2025-09-16
tags: ["Reinforcement Learning", "Research Practice"]
author: "Jasmeet Kaur"
description: "Data, Curiosity and Structure."
summary: "A short review highlight on recent works."
showToc: false
disableAnchoredHeadings: false
---

# Introduction

Label-free learning has enabled recent progress in learning from large
amounts of data. Advancements in Large Language Models showcase the
effectiveness of learning the generative process from huge amounts of
text data. This also pushed learning without labels and learning from
auxiliary labels in some of the sub-fields of Reinforcement Learning
([16], [18], [14], [15], etc.). This article provides a focused study of different works unified by data-driven objectives. It filters the entire
discussion of using data to learn representations into two perspectives.
First is learning driven by curiosity to explore the dataset, and the
second focuses on learning the underlying structure.

This article covers works that focus on learning from data that is
invariant to task-relevant information ([28],
[23], [19] etc.). The discussion highlights the type of underlying structure in the
data, what are the ways to learn about this structure, and how to adapt
these learned insights to solve downstream tasks with minimal
intervention. The aim is to give an integrated understanding of these
works and where this may lead us in the future.

Prevailing practices demonstrate that the availability of data has made
its use and scientific consideration inevitable. Although not all data
is good, a lot of data used for training RL agents comes from unknown
tasks, suboptimal, and incomplete interactions
([10], [17], [14]). Methods to pre-train using this data are
attractive because solving new tasks from scratch is not efficient.
Utilizing data is beneficial in cases where interacting with the
environment during deployment is not possible.

Most of the papers covered under the discussion presented here are works
related to unsupervised and self-supervised pre-training for learning
representations in RL. Some works that lay the foundation of such
approaches have also been described in the relevant sections. The entire
analysis has been divided into four main sections. Each section covers a
pre-training objective that is motivated by either of the two main
factors described above. This article does not propose a novel method to
solve any problem, but offers an organized discussion.

The focus for each section is as follows: exploring datasets for solving
downstream tasks with minimal or no adaptation, learning representations
that capture structure in environmental dynamics, learning
representations that capture similar functional behavior within an
environment, and learning representations that capture structure between
similar and dissimilar behaviors. The objectives include maximizing
entropy, finding temporal structure, learning successor features,
learning with contrastive objectives, and bisimulation metrics. These
methods provide overlapping answers to some important research
questions. These include finding the right type of representations to
learn from data and understanding the trade-offs of different
pre-training objectives for efficient downstream adaptation.

# Problem Statement

All problem settings presented operate in an MDP (Markov-decision
process ([1])) framework or some extension of this
framework. The environment is described as an MDP $M$ with state space
$\mathcal{S}$, action space $\mathcal{A}$, transition dynamics $P$, and
discount factor $\gamma$. A policy $\pi(a|s)$ describes what action to
take given a state $s$. Each task is defined via a reward function $r$
that maps state-action pairs to real-valued rewards. In some cases, an
offline dataset of trajectories is available. This dataset, denoted as
$\mathcal{D}$, consists of $N$ trajectories, where each trajectory
$\tau_i$ is a sequence of tuples $(s_t^i, a_t^i, s_{t+1}^i)$ for $t = 0$
to $T_i$, with $T_i$ being the horizon length of the trajectory. These
trajectories can be collected using any policy, whether optimal or
suboptimal, and may involve unknown reward functions.

# Methods

This section focuses on four different approaches for representation
learning in RL. These representations aim at few-shot adaptations to
downstream tasks. The section describes the different pre-training
objectives, adaptation or fine-tuning strategies for new tasks,
implementation choices, and open questions. The goal is to cover the
strengths and impact of these contributions.

## Exploration-based Representations

Incorporating exploration as an objective can be used to learn diverse
skills from data. Exploration-based works focus on invariance to task
rewards. This can be done for either the offline data available or for
gathering new data through online interactions. [28]
argues that exploring a domain is harder than training a policy to
maximize reward in an environment. A pre-training exploratory policy can
be used for zero-shot RL ([28]). This can be done by
training an exploratory policy that maximizes the entropy of a marginal
state visitation distribution for a finite horizon $T$. This distribution is
defined as: $$\begin{equation}
d_{T,\pi}(s) = \frac{1}{T} \sum_{t=0}^{T} d_{t,\pi}(s),
\end{equation}$$ where $d_{t,\pi}(s)$ denotes the $t$-step state
distribution when following policy $\pi$.

The entropy objective helps move towards the states that the policy has not encountered in the episode. The expectation is over different MDPs sampled from a distribution. This objective learns a policy that equally visits all the states during the
episodes ([28]). Entropy can be computed by a
particle-based k-nearest neighbor approximation
([28], [24]). For practical
implementation, [28] treats the objective as a reward
to be optimized for by training an agent with PPO. But learning an
exploratory policy is not enough, and disentangling exploration from
underlying dynamics is important. One way to do this is to train an
ensemble of policies and use the exploratory policy to find actions in
the environment when the ensemble fails to agree on an action
([28]).

Evidently, the entropy maximizing policy shows a small generalization
gap on the unknown tasks for effective zero-shot generalization. In
Zero-shot generalization, a policy is trained on a set of tasks and
tested on unknown tasks. The generalization quality of this policy is
tested by training agents on 200, 500, and 1000 of Procgen's Maze,
Jumper, and Miner ([28]) and such exploration learns
reward invariance.

A sample-efficient approach with a similar motivation can be used for
representation learning. Learn a visual representation without task
rewards in a self-supervised way. The pre-training phase has two parts.
First, learn a set of prototypical embeddings that form the fundamental
basis of a low-dimensional latent space. Second, use these prototypes to
maximize an entropy-based intrinsic reward, encouraging exploration of
the environment. The intrinsic reward is similar to the one in
[28], but instead of states, it uses encoded
observations. [24] shows the performance of such
an approach on downstream tasks. It appends the intrinsic reward with
the task reward to train a SAC agent for DeepMind Control.

Both methods assume interaction with the environment in the pre-training
phase, with no fine-tuning for downstream adaptations. But diverse
behavior from offline data can also be learnt in a task-agnostic way.
The idea is to optimize for diverse temporal behavior in the
pre-training phase. Learn a state representation $\phi$ that captures
the temporal structure from unlabeled trajectories by using the
equivalence between temporal distance and the optimal goal-conditioned
value function. [17] uses this representation to learn
a policy that spans the latent space and captures diverse skills in the
offline data. Few-shot adaptation to the downstream can be achieved by
learning the task-dependent information.

Curiosity-driven objectives can be used to improve online data
collection and online exploration for new tasks. Such objectives are
effective in solving sparse reward tasks at test time
([4], [15]). The unlabeled data is
labeled with optimistic rewards using RND ([4]) to
guide exploration to unknown states. An RL agent trained with this data
can be used to collect data through online interactions and update the
reward model using this online experience. This offers an iterative
approach for fine-tuning the agent with updated rewards.
[15] evaluates this method for rapid exploration in
challenging sparse-reward domains such as the AntMaze domain, Adroit
hand manipulation, and a visually simulated robotic manipulation domain.

Offline data can be used with such an objective to improve online
exploration. The key idea of this is to extract the low-level skills
from the offline dataset using a variational encoder. A learnt prior
conditions the posterior to stay close to the offline dataset. Learn a
low-level policy to take actions around the dataset trajectories and
learn a high-level policy through online interactions. This method is
effective for tasks that need exploration to be solved
([23]). Both these approaches improve online
exploration by using unlabeled data and implicitly capturing invariance.

## Structured Representations

A few approaches present a way to learn representations that explicitly
exploit structure in the environment. The structure across different
tasks remains the same when the environment dynamics are the same. This
section focuses on one way to learn such invariance.

### Successor Features

Successor Features were used to do transfer in RL by exploiting the
underlying structure in dynamics and performing well on new tasks
([2]). If the expected one-step reward associated
with transition $(s, a, s')$ is decomposed into features represented by
$\phi$ and weights $w$, such a representation can be used to compute the
$Q$-value function. This $Q$-value function can be decomposed into the
task-relevant information and cumulant $\psi$, which is called Successor
Features. Once Successor Features are learnt for different tasks,
Generalized Policy Iteration is used to learn a policy that does as good
as any prior policy on this new task ([2]).

A key focus of an earlier application of this idea is that with SF and
GPI, no task-specific information is shared to learn policies
([3]). SF$\&$GPI can be combined with the UVFAs
([20]) to learn the value functions over the task
encodings and learn a span over the policies. While this offers good
generalization, the performance of the policies on a new task depends on
how the policies are sampled. An alternative approach for learning
shared knowledge across different tasks is using an attention head to
learn successor features ([5]). A more robust way
to generate approximations of successor features is to learn a
distribution of the returns ([6]). Finally, such
structural decomposition can be used to learn to adapt to tasks with
non-Markovian rewards ([16]). The successor
representations capture the expected steps to reach a state from the
current state for the first time.

### Successor Measure

The successor feature decomposition can be extended to state-visitation
distributions. The idea is to decompose the Q-function into the
successor measure and the reward. The successor measure encodes the
probability of going to the next state under the policy
([22]). The entire family of policies can be
parametrized with z, and the Q-function is used to learn the optimal
policies. As in [22], learning the successor measure
representation for all z allows for adaptation to any reward function.
For each $z$, find  $F_z$ and $B$ such
that the successor measure $M_{z}$ is 
$M_{z} = F_{z} B$.  Then, for a reward function $r$, the action-value function is given by
$Q_{z,r} = F_{z}Br$ ([22]).

Learning the span of policies does not have to be restricted to optimal
policies ([29]). Successor Measure can be used to learn
representations in the entire space of policies. [29]
transforms the constrained optimization problem following from the
Bellman Flow equations of successor measure into a single-player game.

Successor Features and Successor Measure offer a concise way to
represent the underlying structure explicitly. Their ability to transfer
to downstream tasks relies on how the space of policies is encoded with
this information, making it an important area of closer examination.

## Bisimulation-based Representations

Bisimulation metrics capture invariance to task-irrelevant features.
They group functionally similar states in a latent space, and capturing
this similarity in an MDP can be useful for generalization over large
state spaces ([8]). In theory, Bisimulation defines an
equivalence relation over the state-space. Partitioning the state-space
under this relation is difficult for practical application
([8]). To deal with this, a semi-metric over this
relation is defined to capture how similar two states are.

While bisimulation metrics are used to learn representations from image
observations in reinforcement learning now, putting bisimulation to
practical use has come a long way. Issues related to its efficient
computation and finding upper bounds for arbitrary policies proved
worthwhile. A sampling-based approximation to the joint distribution of
the transition dynamics made it easier to implement for learning
approximation to a large-state space ([7]). A common
problem that emerges is that of representation collapse, where two
dissimilar states end up with the same representations. This is more
significant in sparse reward settings. One way to mitigate this is to
use cosine distance to measure the distance between two latent states
([23]).
$$
\cos_{\phi}(x, y) = 1 - \frac{\phi(s)^\top \phi(t)}{\|\phi(s)\| \cdot \|\phi(t)\|}
$$

$$
F^{\pi}_{\cos\phi}(s, t) = \big| r^{\pi}_s - r^{\pi}_t \big| + \gamma +
$$

$$
\mathbb{E}_{s' \sim \hat{P}^{\pi}_s, t' \sim \hat{P}^{\pi}_t}
$$

$$
\big[ \cos_{\phi}(s', t') \big]
$$



Aforementioned methods provide objectives to learn effective
representations efficiently using bisimulation metrics. The following
subsections describe three domains where bisimulation metrics have been
adapted for specific problems. More important to this discussion are
works on learning action representations and learning from offline data
with no action information. In general, these works show the
effectiveness of capturing this behavioral similarity in different
settings. These adaptations have been successful in solving complex
tasks such as robot manipulation ([10],
[21]).

### Goal-conditioned Bisimulation Relations

Bisimulation can result in the effective transfer of skills across
analogous tasks ([10]). Given an MDP $M$,
*goal-conditioned bisimulation relation* $\mathcal{B}$ can be defined
([10]). For
all state - goal pairs $(s, g_s), (t, g_t) \in S \times G$ that are
equivalent under $\mathcal{B}$, the following conditions hold:
$$
R(s, a, g_s) = R(t, a, g_t), \quad \forall a \in A
$$

$$
P(G \mid s, a) = P(G \mid t, a), \quad \forall a \in A, \; G \subseteq S.
$$


For practical implementation, an *on-policy* version of this relation
gives rise to a paired-state metric: 

$$
d\big((s,g_s), (t,g_t)\big) = \big| R(s, \pi(s,g_s), g_s) - R(t, \pi(t,g_t), g_t) \big| + 
$$

$$
W_2 \Big( P(\cdot \mid s, \pi(s,g_s)), P(\cdot \mid t, \pi(t,g_t)) \Big)
$$

where $\pi$ is the goal-conditioned policy. This objective
can be used to learn state-goal representations and train an offline
goal-conditioned policy. [10] used a dataset
collected using a noisy expert to show skill transfer in manipulation
tasks.
$$
\begin{aligned}
L_{\phi} &= \biggl(
&\quad \left\| \phi(s, g_s) - \phi(t, g_t) \right\|_1 
&\quad -\left\| R_s - R_t \right\|_2 
&\quad -\gamma \left\| \overline{\phi}(s', g_s') -\overline{\phi}(t', g_t') \right\|_2 
&\biggr)^2
\end{aligned}
$$


A bisimulation objective can be appended with a forward model
error-based intrinsic reward to improve exploration. This enhances
exploration in the latent space for learning a bisimulation-based
representation and has been empirically shown to learn robust state
representations for downstream goal-conditioned tasks
([11]).

### Learning from Offline Data

Bisimulation-based representations can be learnt using an offline
dataset. Learning from an offline dataset is notoriously difficult using
function approximation. The representations learnt using the
bisimulation metric can result in a value function that does not
diverge. These representations stabilize TD-learning and are Bellman
complete ([18]). [18] defines a
$\pi$-bisimilarity kernel $k^{\pi_e} : S \times S \to \mathbb{R}$ for
pairs of state-actions and under $\pi_e$: 

$$
k^{\pi_{e}}(s,a_s; t,a_t) =
$$

$$
\quad k_1(s,a_s; t,a_t)
\quad + \gamma \quad k_2\big(k^{\pi_{e}}\big)
\quad \big(
P^{\pi_{e}}(\cdot|s,a_s), \;
P^{\pi_{e}}(\cdot|t,a_t)
\big)
$$


Here, $k_1$ measures short-term similarity based on the
rewards received. $k_2$ measures long-term similarity between
probability distributions by evaluating similarity between samples of
the distributions according to $k^{\pi_{e}}$.

To use the bisimulation metric for offline datasets, it must learn from
an incomplete dataset with missing transitions. While Implicit Q
learning can be used to mitigate this problem
([10]), the error in the estimated value of
bisimulation operator can be reduced by direct use of an expectile
operator
([27]). 

Finally, for real-world deployment of bisimulation-based
representations, naively training with adversarial states and actions
does not transfer well to the downstream tasks
([25]). Learn robust representation with perturbed
states and goals using a contrastive objective.

### Action Invariance

Bisimulation-based metrics enable capturing invariance in behavior by
learning action representations. Such long-term action representations
can be learnt in a self-supervised way using bisimulation
([21]). Such representations are
effective for solving complex tasks and this is experimentally shown
with 7DOF ARM Control ([21]).

Learning action representations is not restricted to online interactions
but can also be achieved from an offline dataset. To stabilize learning
close to the behavioral policy and mitigate distribution mismatch, trainable model predicting out-of-distribution actions.

While the method above explicitly handles the behavioral distribution, a
similar objective for single-step action representation, denoted by
$\phi$, can be learnt.

Bisimulation effectively captures the similarity between states. It can
be adapted for offline datasets, online exploration, and learning action
representations.

## Contrastive Learning based Representations

Contrastive Learning objectives have been used for unsupervised RL. It
is a framework to learn representations by exploiting the structure
between similar and dissimilar pairs of input. This can be achieved in
an unsupervised way by performing a dictionary lookup task wherein the
positives and negatives represent a set of keys with respect to a query.
There are many ways in which such an objective can be captured, and one
of the most adapted ones is the InfoNCE loss function.

Contrastive Learning can be used to train an RL agent over
representations learnt from image-based observations
([12]). It can be used to improve the robustness of
bisimulation representation for transfer to downstream tasks by using it
to perturb the negative samples ([25]).

Finally, Contrastive Learning is effective for representations over
offline image data ([15]). Learn representation over states
using a goal-conditioned offline pre-training objective. The objective
minimizes the distance between the goal-conditioned state-occupancy
distribution of the policy and the data distribution and has been effective for zero-shot generalization in
goal-conditioned reinforcement learning ([15]).


### Conclusion

This article provides an overview of recent research trends in RL. It
looks at different works using unsupervised and self-supervised methods
of learning in RL and provides a unification of the core objectives.
Curiosity-based approaches capture a notion of
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

ChatGPT (OpenAI) has been used to assist in
finding word phrasing. The text was proofread using Grammarly's basic
grammar and style suggestions, without the use of AI-generated content.

### Citation 
If you found this useful in your academic work, please cite this using:

Kaur, Jasmeet. Pre-Training in Reinforcement Learning: Data, Curiosity and Structure. https://jasmeetkaur9.github.io/blog/mentalnotes/.


### References 
1. Bellman R. Dynamic programming and stochastic control processes. 
Information and control. 1958 

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

28. Siddhant Agarwal, Harshit Sikchi, Peter Stone, and Amy Zhang. Proto
successor measure: Representing the behavior space of an rl agent.
*arXiv preprint arXiv:2411.19418*, 2024.

