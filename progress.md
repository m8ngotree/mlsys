# Progress

## January 4
Day 37 of ML Systems: I only had two hours today because of a long drive, but I just used Claude to walk me through the Miles codebase. I learned about MoE architectures and how Miles uses Ray for distributed computing. I'll keep learning the codebase tomorrow.

## January 3
Day 36 of ML Systems: I spent a few hours today doing a deep dive into speculative decoding by learning from Claude. I learned how speculative decoding works along with the math behind the verification algorithm - Claude was really helpful/useful in teaching me this. I briefly started doing a deep dive into the Miles codebase but I'll do more on this tomorrow.

## January 2
Day 35 of ML Systems: I spent today looking into RadixArk's Miles repo and reading documentation. I think I will focus on making open source contributions to this since it is integrated with SGLang and Megatron-LM. I did some reading on speculative decoding - I'll try to make contributions related to this.

## January 1
Day 34 of ML Systems: I pivoted again and decided to learn concepts I'm interested in related to inference and make open source contributions to things like SGLang. I did some studying today by reviewing attention mechanisms and just scrolling through resources for SGLang. Tomorrow I plan on reading through a lot of documentation.

## 2026

## December 19
Day 33 of ML Systems: I finished writing code and testing for the naive fused GEMM / MXFP8 dequantization kernel. I plan on doing the optimized and tensor core versions tomorrow.

## December 17
Day 32 of ML Systems: I mostly finished writing the fused GEMM / MXFP8 dequantization kernel. There is still some stuff I need to look into for the dequantization, but I plan on doing that and writing some tests to compare performance to PyTorch and separate kernels.

## December 16
Day 31 of ML Systems: I started working on a fused GEMM / MXFP8 dequantization kernel. I spent time writing out the tile loading and learning how to manage the scales while tiling.

## December 9
Day 30 of ML Systems: I started working on my MXFP / NVFP quantization/dequentization kernels.

## December 8
Day 29 of ML Systems: I finished learning about MXFP/NVFP. I plan on implementing quantization/dequantization, fused GEMM, and conversion kernels for these formats in the coming days.

## November 28
Day 28 of ML Systems: I learned about bit packing/unpacking and implemented a naive kernel. I didn't get as much done as I wanted to, but I decided to hop into learning about MXFP/NVFP. I don't really know if I have all the fundamentals yet, but I decided I'd rather learn top-down than bottom-up.

## November 27
Day 27 of ML Systems: I basically spent all day learning about INT8 quantization and created a series of quantization kernels. There was a lot of stuff I learned in between about numerics, but a lot of the concepts such as warp shuffling, shared memory, coalescing, vectorizing, etc. carried over. I was able to achieve a 5x speedup compared to PyTorch and verified the correctness of my kernels compared to PyTorch. Tomorrow, I plan on learning about bit packing and writing some related kernels.

## November 25
Day 26 of ML Systems: I finished reading Alpindale's top-k kernel blog. There was a lot to digest especially because I wasn't too familiar with PTX and some of the optimizations. I decided to start doing my own stuff from tomorrow onwards as I feel like I've gotten the basics down and probably spent too much time reading other people's stuff instead of doing my own projects. I'll mainly be focusing on systems work for quantization, training, & inference. I plan on starting a deep dive into quantization tomorrow.

## November 24
Day 25 of ML Systems: I read the first few parts of Alpindale's top-k kernel blog (https://blog.alpindale.net/posts/top_k_cuda/). It's a really good read - the optimizations are really interesting / unique, and I spent a lot of time learning about the various floating-point formats and concepts such as denormal/subnormal numbers. I'll keep reading through it tomorrow.

## November 23
Day 24 of ML Systems: I read and coded out the kernels from @aryagxr 's CUDA layernorm blog. It was fairly similar to the SGEMV/softmax kernels I did, but it was good to practice writing kernels using techniques I've learned such as reductions, warp shuffling, and vectorized memory loads.

## November 22
Day 23 of ML Systems: I coded out the kernels from @maharshii 's SGEMV blog. I learned about a lot of different things like cuBLAS, various CUDA library features, Makefiles, and how to structure .cuh/.cu files.

## November 21
Day 22 of ML Systems: I read through @maharshii 's SGEMV blog and started coding out some of the kernels. I plan on finishing the rest of the kernels tomorrow.

## November 20
Day 22 of ML Systems: I studied the final 2 softmax kernels from @maharshii 's softmax blog and learned about warp-level primitives like shuffling, function templates for block/warp reducing, and vectorized memory loads. I coded out the kernels as well. This is technically for today and yesterday but I didn't get that much done yesterday. I spent about 3 hours today.

## November 18
Day 21 of ML Systems: I coded out the first 3 kernels of @maharshii 's softmax blog and learned more about the various optimizations such as the online algorithm and reductions as well as CUDA runtime functions. I spent about 3 hours total.

## November 12
Day 20 of ML Systems: I was a little inefficient with my studying today, but I mainly went through @maharshii ‘s softmax and understood the optimizations at a high level. Tomorrow, I’ll go through and start coding things out.

## November 11
Day 19 of ML Systems: I finished up studying the rest of the kernels from Simon Boehm's blog and learned about warp tiling. I'll start something new tomorrow.

## November 10
Day 18 of ML Systems: I worked on understanding Kernel 7 from Simon Boehm's blog in the evening. I initially had trouble understanding the linearization of the shared memory for the B matrix as well as the access patterns but I got Claude to explain it with visualizations which really helped me figure it out. I have the day off tomorrow so planning on finishing studying the rest of the kernels and moving on to something else.

## November 9
Day 17 of ML Systems: I worked on understanding Kernel 6 from Simon Boehm's blog and implemented my own version of it. I worked on a matmul kernel with more optimizations such as transposition of shared memory, vectorized memory loading, and register blocking. I achieved 90.7% percentile performance on LeetGPU for B200. One of my tweets got liked by @therealkmans who works on LeetGPU.

## November 6
Day 16 of GPU Programming: I learned about vectorizing memory accesses by transposing shared memory matrices and studied Simon Boehm's kernel that implements this technique.

## November 5
Day 15 of GPU Programming: I spent some time implementing my own 2D Blocktiling matmul kernel. On LeetGPU, I achieved 93.5% percentile for Tesla T4 and 81.3% for B200. I also spent time learning about vectorizing memory accesses from Simon Boehm's blog.

## November 4
Day 14 of GPU Programming: I spent time understanding the 2D tiling matmul kernel in Simon Boehm's blog post and got a pretty good understanding of it.

## October 30
Day 13 of GPU Programming: I just kept reading Simon Boehm's blog post. I had to spend more time understanding the kernels because they are in a slightly different format.

## October 29
Day 12 of GPU programming: I read through the 4th kernel of Simon Boehm's blog post. I had to spend some time understanding the code even though I already learned the concepts from PMPP because it was in a different format than the PMPP code.

## October 28
Day 11 of GPU Programming: I read through the first part of Simon Boehm's blog on optimizing matmul kernels. It was mainly just a refresher of what I've learned from PMPP.

## October 27
This is combining the last few days because I didn't get to spend much time each day, but I mainly worked on understanding the tiled matmul kernel with dynamic shared memory in PMPP Chapter 5 using Claude. I also read through PMPP Chapter 6 and learned things like thread coarsening. I think I got a pretty good understanding of all the concepts from Claude and testing them out on LeetGPU. I'm planning on reading through Simon Boehm's blog next (https://siboehm.com/articles/22/CUDA-MMM) to learn more about optimizing matmul kernels.

## October 23 - Day 9 of GPU Programming

I spent today using Claude to deeply understand the naive matmul kernel. I was kind of unsure on the dimension calculations, but using my Claude prompt (listed below) helped me understand it from first principles.

**Learning Prompt:**
> I want you to teach me so that I can understand this, go step-by-step and build my knowledge up from first principles, I want you to present information for quiz me at each step to ensure I have actually learned, don't immediately quiz, ask me if i have any questions, i will ask questions until i don't and then i will tell you to quiz me on everything you presented at each step and we can go from there, the end goal is for me to understand this code at a foundational level

## October 22 - Day 8 of GPU Programming

I finished Chapter 5 of PMPP. I got kind of confused by the tiled matmul kernel so I spent a lot of time understanding it. Probably will read Chapter 6 of PMPP and dive more into matmul kernels next.

## October 21 - Day 7 of GPU Programming

I read most of Chapter 5 of PMPP focusing on the various memory types and tiling. Planning on finishing Chapter 5 & 6 tomorrow.

## October 17

**Learning Tip:** One thing that really helps when reading PMPP is to summarize each paragraph in my own words after I read it by talking out loud. Sometimes I tend to glaze/skim over when reading text and this helps to make sure I actually understand what's going on.

## October 16 - Day 6 of GPU Programming

I read through Chapter 4 of PMPP today. I also wrote my first vector addition kernel on LeetGPU. I spent some time really making sure I understanding the fundamental syntax.

## October 15

I read through Chapters 2 & 3 of PMPP today. I skimmed a little bit but I think I understand the main ideas pretty well.

## October 14

I waffled around a bit through different video tutorials, but I just decided to grind the *Programming Massively Parallel Processors 4th edition* textbook to learn the fundamentals. I finished Chapter 1 today. I plan on doing a chapter or so each day and spending my remaining time doing applied work on inference or other technical LLM stuff.

## October 8

I finished going through and understanding the CUDA implementation of Flash Attention in https://www.pyspur.dev/blog/introduction_cuda_programming. I plan on watching Umar Jamil's [Flash Attention from First Principles video](https://www.youtube.com/watch?v=zy8ChVd_oTM&t=4844s) to gain a full understanding.

## October 7

I kept working on understanding the PyTorch/CUDA code for Flash Attention in [this link](https://pyspur.dev/blog/introduction_cuda_programming). I feel like I got a pretty good understanding of the PyTorch code now. 

**Learning Prompt:** Using this prompt with Claude really helps with learning:

> I want you to teach me so that I can understand this, go step-by-step and build my knowledge up from first principles, I want you to present information for quiz me at each step to ensure I have actually learned, don't immediately quiz, ask me if i have any questions, i will ask questions until i don't and then i will tell you to quiz me on everything you presented at each step and we can go from there, the end goal is for me to understand this code at a foundational level

## October 6

Spent time learning CUDA / systems basics from the following blogs:
- https://pyspur.dev/blog/introduction_cuda_programming
- https://horace.io/brrr_intro.html

I realized I don't have a solid understanding of the transformer / CUDA code in the first blog, so I'll spend tomorrow really trying to understand it.

## 2025