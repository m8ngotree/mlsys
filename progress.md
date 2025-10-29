# Progress

## October 6

Spent time learning CUDA / systems basics from the following blogs:
- https://pyspur.dev/blog/introduction_cuda_programming
- https://horace.io/brrr_intro.html

I realized I don't have a solid understanding of the transformer / CUDA code in the first blog, so I'll spend tomorrow really trying to understand it.

## October 7

I kept working on understanding the PyTorch/CUDA code for Flash Attention in [this link](https://pyspur.dev/blog/introduction_cuda_programming). I feel like I got a pretty good understanding of the PyTorch code now. 

**Learning Prompt:** Using this prompt with Claude really helps with learning:

> I want you to teach me so that I can understand this, go step-by-step and build my knowledge up from first principles, I want you to present information for quiz me at each step to ensure I have actually learned, don't immediately quiz, ask me if i have any questions, i will ask questions until i don't and then i will tell you to quiz me on everything you presented at each step and we can go from there, the end goal is for me to understand this code at a foundational level

## October 8

I finished going through and understanding the CUDA implementation of Flash Attention in https://www.pyspur.dev/blog/introduction_cuda_programming. I plan on watching Umar Jamil's [Flash Attention from First Principles video](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=4844s) to gain a full understanding.

## October 14

I waffled around a bit through different video tutorials, but I just decided to grind the *Programming Massively Parallel Processors 4th edition* textbook to learn the fundamentals. I finished Chapter 1 today. I plan on doing a chapter or so each day and spending my remaining time doing applied work on inference or other technical LLM stuff.

## October 15

I read through Chapters 2 & 3 of PMPP today. I skimmed a little bit but I think I understand the main ideas pretty well.

## October 16 - Day 6 of GPU Programming

I read through Chapter 4 of PMPP today. I also wrote my first vector addition kernel on LeetGPU. I spent some time really making sure I understanding the fundamental syntax.

## October 17

**Learning Tip:** One thing that really helps when reading PMPP is to summarize each paragraph in my own words after I read it by talking out loud. Sometimes I tend to glaze/skim over when reading text and this helps to make sure I actually understand what's going on.

## October 21 - Day 7 of GPU Programming

I read most of Chapter 5 of PMPP focusing on the various memory types and tiling. Planning on finishing Chapter 5 & 6 tomorrow.

## October 22 - Day 8 of GPU Programming

I finished Chapter 5 of PMPP. I got kind of confused by the tiled matmul kernel so I spent a lot of time understanding it. Probably will read Chapter 6 of PMPP and dive more into matmul kernels next.

## October 23 - Day 9 of GPU Programming

I spent today using Claude to deeply understand the naive matmul kernel. I was kind of unsure on the dimension calculations, but using my Claude prompt (listed below) helped me understand it from first principles.

**Learning Prompt:**
> I want you to teach me so that I can understand this, go step-by-step and build my knowledge up from first principles, I want you to present information for quiz me at each step to ensure I have actually learned, don't immediately quiz, ask me if i have any questions, i will ask questions until i don't and then i will tell you to quiz me on everything you presented at each step and we can go from there, the end goal is for me to understand this code at a foundational level

## October 27
This is combining the last few days because I didn't get to spend much time each day, but I mainly worked on understanding the tiled matmul kernel with dynamic shared memory in PMPP Chapter 5 using Claude. I also read through PMPP Chapter 6 and learned things like thread coarsening. I think I got a pretty good understanding of all the concepts from Claude and testing them out on LeetGPU. I'm planning on reading through Simon Boehm's blog next (https://siboehm.com/articles/22/CUDA-MMM) to learn more about optimizing matmul kernels.

## October 28
Day 11 of GPU Programming: I read through the first part of Simon Boehm's blog on optimizing matmul kernels. It was mainly just a refresher of what I've learned from PMPP.