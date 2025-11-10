TODO:
PMPP
LEETGPU
MATMULS

IMPLEMENT:
MATMULS
https://maharshi.bearblog.dev/optimizing-softmax-cuda/
https://aryagxr.com/blogs/cuda-optimizing-layernorm
https://andrewkchan.dev/posts/yalm.html#section-1.1
ATTENTION
KERNELS
MOE

READ:
https://cursor.com/blog/kernels
https://andrewkchan.dev/posts/yalm.html
https://www.together.ai/blog/adaptive-learning-speculator-system-atlas
https://github.com/vllm-project/vllm
https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html
https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html

Attention:
https://www.youtube.com/watch?v=zy8ChVd_oTM
https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
https://www.stephendiehl.com/posts/flash_attention/
https://gau-nernst.github.io/fa-5090/
https://tridao.me/
https://maharshi.bearblog.dev/optimizing-softmax-cuda/
https://hamdi.bearblog.dev/understanding-flash-attention-forward-with-cuda/
https://github.com/tspeterkim/flash-attention-minimal
https://medium.com/@damienjose/flash-attention-with-cuda-c45d9167e8dc
https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax
https://pytorch.org/blog/flexattention/
https://github.com/pytorch-labs/attention-gym
https://github.com/shi-labs/natten
https://github.com/AlpinDale?tab=repositories

Inference:
https://bentoml.com/llm/ - ✅
https://andrewkchan.dev/posts/yalm.html#section-1.1
https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
https://github.com/vllm-project/vllm
https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html
https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
https://pytorch.org/blog/flash-decoding/
https://bench.flashinfer.ai/
https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html
https://flashinfer.ai/2024/02/02/cascade-inference.html
https://github.com/xlite-dev/Awesome-LLM-Inference
https://rentry.org/samplers

Speculative Decoding
https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/
https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/

MoE:
https://github.com/woct0rdho/transformers-qwen3-moe-fused
https://bit-ml.github.io/blog/post/fused-swiglu-kernel/
https://pytorch.org/blog/accelerating-moe-model/

MatMul:
https://siboehm.com/articles/22/CUDA-MMM,
https://www.aleksagordic.com/blog/matmul
https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
https://www.spatters.ca/mma-matmul
https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
https://salykova.github.io/sgemm-gpu
https://www.youtube.com/watch?v=ErTmTCRP1_U
https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html
https://maharshi.bearblog.dev/optimizing-sgemv-cuda/

Kernels:
https://www.youtube.com/watch?v=IpHjDoW4ffw
https://cursor.com/blog/kernels
https://aryagxr.com/blogs/cuda-optimizing-layernorm
https://github.com/HazyResearch/ThunderKittens
https://www.together.ai/blog/thunderkittens-nvidia-blackwell-gpus
https://research.perplexity.ai/articles/enabling-trillion-parameter-models-on-aws-efa
https://research.colfax-intl.com/
https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/
https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf
https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
http://www.kapilsharma.dev/posts/triton-kernels-softmax/
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
https://github.com/fattorib/CudaSoftmax
https://danfu.org/
https://sandyresearch.github.io/
https://gpu.camp/

GPU:
https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu
https://leetarxiv.substack.com/p/learning-cuda-on-a-budget-on-google
https://feldmann.nyc/blog/smem-microbenchmarks

Twitter:
Bookmarks
https://irhum.github.io/blog/pjit/
https://github.com/MekkCyber/TritonAcademy
https://docs.unsloth.ai/
https://www.deep-ml.com/problems
https://hazyresearch.stanford.edu/
https://hamzaelshafie.bearblog.dev/
https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey
https://x.com/TheAhmadOsman/status/1966780033206264100
https://gaurigupta19.github.io/llms/distributed%20ml/optimization/2025/10/02/efficient-ml.htmlhttps://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

https://leetgpu.com/
https://tensara.org/
https://github.com/julienokumu/100DaysOfGPUProgramming/tree/main/day%200
https://github.com/Maharshi-Pandya/cudacodes
https://www.youtube.com/watch?si=Pz4IX1zHfoASqgxV&v=NBqHVjyDFfQ&feature=youtu.be
https://x.com/fleetwood___/status/1968716580621271076
https://x.com/SemiAnalysis_/status/1978602446386520400
https://x.com/mrsiipa/status/1888632883738550294
https://x.com/mrsiipa/status/1885286750555430988
https://x.com/hy3na_xyz/status/1976753174389424588
https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?pli=1&tab=t.0#heading=h.2dzgqxiwa5xd
https://blog.ezyang.com/
https://simonguo.tech/blog/2025-10-automated-gpu-kernels.html
https://cognition.ai/blog/swe-grep#fast-context-as-the-first-step-to-fast-agents
https://stuartsul.com/
https://publish.obsidian.md/ueaj/Welcome
https://x.com/Kimi_Moonshot/status/1983937694360322136
https://x.com/gaunernst/status/1984993034078179665
https://x.com/mrsiipa/status/1985004396791681307
https://x.com/mrsiipa/status/1985002945306583140
https://x.com/mrsiipa/status/1985003644199256114
https://www.modular.com/
https://x.com/mrsiipa/status/1985002945306583140
https://x.com/mrsiipa/status/1985004396791681307
https://x.com/gaunernst/status/1984993034078179665
https://x.com/mrsiipa/status/1986152319004856491
https://x.com/gm8xx8/status/1985961647664410714
https://x.com/samsja19/status/1986232036055785698
https://x.com/Grad62304977/status/1986219468465303703
https://inference.net/blog/logic
https://blog.alpindale.net/posts/top_k_cuda/
https://kalomaze.bearblog.dev/rl-lora-ddd/
https://x.com/StefanoErmon/status/1986477376835047740
https://x.com/sadernoheart/status/1987491712374038970

General:
https://horace.io/brrr_intro.html
https://github.com/karpathy/llm.c
https://siboehm.com/
https://blog.ezyang.com/2019/05/pytorch-internals/
https://www.amansanger.com/

C/C++:
https://www.learncpp.com/

Nsight:
https://docs.nvidia.com/nsight-systems/
https://docs.nvidia.com/nsight-compute/

CUDA:
https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf
https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/
https://docs.nvidia.com/cuda/cuda-c-programming-guide/
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
https://www.amazon.com/CUDA-Example-Introduction-General-Purpose-Programming/dp/0131387685
https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311
https://github.com/gpu-mode/lectures
https://modal.com/gpu-glossary/readme
https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf

Triton:
https://openai.com/index/triton/
https://github.com/srush/Triton-Puzzles/tree/main
https://github.com/SiriusNEO/Triton-Puzzles-Lite/tree/main

Jax:

Pallas:

XLA:

PTX:

SASS:

Quantization:
https://franciscormendes.github.io/2024/05/16/quantization-1/
https://leimao.github.io/article/Neural-Networks-Quantization/

Drug Discovery:
https://fabianfuchsml.github.io/alphafold2/
https://developer.nvidia.com/blog/accelerating-se3-transformers-training-using-an-nvidia-open-source-model-implementation/
https://openfold.readthedocs.io/en/latest/
https://arxiv.org/html/2404.11068v1
https://supercomputing-system-ai-lab.github.io/blogs/blog/megafold-an-open-sourced-alphafold-3-training-system/
https://lupoglaz.github.io/OpenFold2/iterativeSE3Transformer.html
