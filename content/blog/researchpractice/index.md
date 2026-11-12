---
title: "From Paper to Prototype"
date: 2025-05-28
lastmod: 2025-05-28
tags: ["Reinforcement Learning", "PyTorch", "Research Practice"]
author: "Jasmeet Kaur"
description: "This article is my account on implementing an reinforcement learning paper."
summary: "An brief guide to training and experimenting in reinforcement learning."
showToc: false
disableAnchoredHeadings: false
---

The focus group of this short article is anyone starting to study and practice in areas closely related to Reinforcement Learning. It is an account of my experience in implementing some ml papers. Before we proceed, I must add that this article is no replacement for professional advice. You are welcome to read what I learned and test it at your discretion. I am sure that you will find certain things useful. Implementing any machine learning research paper is a helpful skill. I have found it to be particularly useful for understanding my limitations and working on my weaknesses when it comes to coding for my research projects. 

To start, you can pick any paper that you like - a paper that describes a baseline you are interested in, a paper you read but did not understand thoroughly, or a paper that you think would be cool to test and improve in some way. These are all good reasons to get started. You might also want to be clear on how much time you want to commit. When I started, it took me 2-3 months to implement my first RL paper. I had the codebase available from the paper's authors, but I made numerous mistakes when trying to implement it on my own. With more practice, I found that 4-5 days works the best for me. So find what works for you. Feel free to push boundaries, but you might get some pushback.

Another thing that has helped me is structuring the code and planning out its various components. I have found a well-thought-out project structure is helpful. If you are implementing an RL paper, chances are you need to either generate your dataset or use one available online. Figure things like the dataset, environment, model architecture, trainers, integration, and experiments out before you start. Some questions that I like to keep in mind is - How to test with changing model architecture? How will the optimization be affected? How to process a large amount of data? I usually start with a small example similar to the one used in the paper, and get the code working on this before I run any big experiments. 

My favorite framework to use is **PyTorch**. I think PyTorch is amazing. My first baseline paper had code written in Tensorflow, and I sort of caught up with PyTorch by writing all that Tensorflow code using PyTorch. T'was a good exercise. It's well documented with a helpful community - I have been able to find all the answers to my queries on their blog. For recording experiments, logging data, and plots, I have been using a small logger with a set of files that log data and plot it. 

When I started, I spent a lot of effort optimizing my code for "time". Looking back, I realised that it can be a good metric to optimize. **Profiling** is helpful to figure out the parts of your code that are slow. Multi-processing and multiple GPUs can be useful if your code demands that. For one of my projects, I realised I could do without it by simply avoiding tensor switching between CPUs and GPUs. Yes, you can save a lot of time by initializing and using all tensors on a single device. Certain operations, like doing matrix decomposition, do not require GPUs, so you would want to keep them on CPUs. Recently, I also spent some days learning about JAX, but I have not used it yet. I will update this when I do. 

When you finally get to code, you want to be mindful of your favorite AI assistance tool to make the most of this exercise. The point should be to understand the paper's main ideas and results while improving one's ability to test out experiments as quickly as possible. I find it useful to use these tools for setting up project environments, certain code conversions, generating links to documents, sometimes understanding what certain code I found online is about, then trying to code it, and generating code snippets that I have given up on. The last one used to happen more often in the beginning, but I improved over time. So, this works fine for me, and I believe you want to give your level of autonomy with these tools, a good thought. 

Finally, the results. This is a tricky part. Specifically, when it comes to implementing RL papers, I think. There are some brilliant resources on reproducibility in RL experiments, which I found extremely helpful, and I share them below. For the simplicity that RL has in its formulation and structure, it is brittle, and it can be tricky to get to work in practice. Some of the things that have helped me so far are - working with small examples, logging the policy, being careful around stop-gradients, rendering the environment, and in some cases, plotting the losses. 

And, this brings to the end of this article for now. I plan to keep adding things as I explore and exploit. Thank you, if you made it this far! **Happy Tinkering with RL!**  

### Resources:
- [Deep RL Hacks](https://github.com/williamFalcon/DeepRLHacks)
- [Deep Reinforcement Learning that Matters (arXiv)](https://arxiv.org/abs/1709.06560)
- [Debugging RL Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html)

