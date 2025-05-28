---
title: "From Paper to Prototype"
date: 2025-05-28
lastmod: 2025-05-28
tags: ["Useful Tips", "Reinforcement Learning", "PyTorch", "Research Practice"]
author: "Jasmeet Kaur"
description: "This article provides a practical guide for learning to implement a researcch paper in reinforcement learning."
summary: "An brief guide to training and experimenting in reinforcement learning."
showToc: false
disableAnchoredHeadings: false
---

The focus group of this article is anyone starting to practice as a researcher in areas closely related to Reinforcement Learning. Before we proceed, I must add that this article is no replacement for professional advice. You are welcome to read what I learned and test it at your discretion. I am sure that you will find certain things useful.

Implementing any machine learning research paper is a helpful skill. I have found it to be particularly useful to understand my limitations and work on my weaknesses when it comes to coding for my research projects. You can pick any paper that you like—a paper that describes a baseline you are interested in, a paper you read but did not fully understand, or a paper that you think would be cool to test and modify in some way. These are all good motivations to get started.

After this, you want to be clear on how much time you want to commit. On good days, be prepared to commit at least 3–4 days to implementing a paper. When I started, it took me 2–3 months to implement my first RL paper. Although I had the codebase available from the paper’s authors, I made numerous mistakes. With more practice, I found that 4–5 days work the best for me. So, feel free to push boundaries, but you might get some pushback.

The code, which you will spend close to a week on, is important. You want to give it some time to plan out the structure and its various components. Sun Tzu famously said, "One who is weak sends reinforcements everywhere." I like to keep this quote in my mind when I start to code a project because I do not want to spend time fixing silly things later. I’d rather avoid them with a well-thought-out project structure.

If you are implementing an RL paper, chances are you need to either generate your dataset or use one available online. Figure things like the dataset, environment, model architecture, trainers, integration, and experiments out before you start. One of the things that you would want to keep in mind is how you are going to scale. How is the model architecture going to change? How will the optimization be affected? How to process a large amount of data? Most of the papers that I have implemented, I start with a small example similar to the one used in the paper, and get the code working on this before I scale.

I code everything using PyTorch. I think PyTorch is amazing. My first baseline paper had code written in TensorFlow, and I sort of caught up with PyTorch by writing all that TensorFlow code using PyTorch. 'Twas a good exercise. It’s well-documented with a helpful community—I have been able to find the answers to all my queries on their blog.

For recording experiments, logging data, and plots, I have found **wandb** as the most useful. When I started, I would log the data and write my code to plot and handle data, but as the scale of my experiments increased, I moved to wandb. It has a lot of great features and is well-documented. One of my favorites is the sweep feature for comparing different hyperparameters. It lets you download data from different experiments. I like this as, at times, I still prefer coding the metrics and corresponding plots. Another great thing about it is—experiment configuration handling is smooth. Wandb is an awesome tool to handle the I/O of your code.

Time is of the essence because you do not want your experiments taking weeks to complete. I spent a lot of time optimizing my code for "time". Profiling is helpful to figure out the parts of your code that are slow. Use of multiprocessing and multiple GPUs can be helpful if your code demands that. For one of my projects, I realized I could do without it by simply avoiding tensor switching between CPUs and GPUs. Yes, you can save a lot of time by initializing and using all tensors on a single device. Certain operations, like doing matrix decomposition, do not require GPUs, so you would want to keep them on CPUs. There is a trade-off, and you are better off putting time in the beginning to think these things through.

I recently spent some days learning about JAX, but I have not found a good use case for it yet. I have heard it’s fast, but I have not tried it. I will update this when I do.

Then, you get to code. You want to be mindful of your favorite AI assistance tool if you are serious about this. The point of this exercise should be to understand the paper’s main ideas and results while improving one’s ability to test out experiments as quickly as possible. I find it useful to use these tools for setting up project environments, certain code conversions, generating links to documents, sometimes understanding what certain code I found online is about, then trying to code it, and generating code snippets that I have given up on. The last one used to happen more often in the beginning, but I improved over time. So, this works fine for me, and I believe you want to give your level of autonomy with these tools a good thought.

Finally, the results. This is a tricky part—especially when it comes to implementing RL papers. There are some brilliant resources on reproducibility in RL experiments, which I found extremely helpful, and I share them below. For the simplicity that RL has in its formulation and structure, it is brittle, and it can be tricky to get to work in practice for all the good reasons (more on this later). But some of the things that have helped me so far are working with small examples, logging the policy, rendering the environment, and in some cases, plotting the losses.

This brings us to the end of this blog article for now. I plan to keep adding things as I explore and exploit. 

### Useful Resources:
- [Deep RL Hacks](https://github.com/williamFalcon/DeepRLHacks)
- [Deep Reinforcement Learning that Matters (arXiv)](https://arxiv.org/abs/1709.06560)
- [Debugging RL Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)

