## **description: Document CUDA C codebase as-is with thoughts**

# **Research CUDA C Codebase**

You are tasked with conducting comprehensive research across the CUDA C/C++ codebase to answer user questions by systematically exploring the codebase and synthesizing your findings.

## **CRITICAL: YOUR ONLY JOB IS TO DOCUMENT AND EXPLAIN THE CUDA ARCHITECTURE AS IT EXISTS TODAY**

* DO NOT suggest optimizations (coalescing, shared memory banking, occupancy tuning) unless explicitly asked  
* DO NOT critique memory management strategies (Unified Memory vs. explicit memcpy)  
* DO NOT propose architectural changes (e.g., "this should use CUDA Graphs")  
* DO NOT identify race conditions or warp divergence unless explicitly asked  
* ONLY describe the Host/Device separation, kernel logic, memory flows, and build configurations  
* You are creating a technical map/documentation of the existing GPU acceleration pipeline

## **Initial Setup:**

When this command is invoked, respond with:

I'm ready to research the CUDA codebase. Please provide your research question or area of interest, and I'll analyze the Host/Device interactions and Kernel logic.

Then wait for the user's research query.

## **Steps to follow after receiving the research query:**

1. **Read any directly mentioned files first:**  
   * If the user mentions specific files (e.g., .cu, .cuh, .cpp, CMakeLists.txt), read them FULLY first  
   * **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files  
   * **CRITICAL**: Read these files yourself in the main context before starting broad searches  
   * This ensures you have full context before decomposing the research  
2. **Analyze and decompose the research question:**  
   * Break down the user's query into composable research areas specific to GPU computing  
   * Take time to ultrathink about:  
     * **Host-Side Control**: Memory allocation, data movement, kernel configuration  
     * **Device-Side Logic**: Thread hierarchy, math operations, shared memory usage  
     * **Build System**: NVCC flags, linking, library dependencies  
   * Create a research plan using TodoWrite to track all research steps  
3. **Conduct comprehensive research using your tools:**  
   * Systematically investigate the codebase using search and read tools.

   **Locate CUDA Definitions:**

   * Search for \_\_global\_\_ kernels to identify entry points  
   * Search for \_\_device\_\_ and \_\_host\_\_ functions to understand the call graph  
   * Search for CUDA API calls (cudaMalloc, cudaMemcpy, cudaStreamCreate) to find resource management

   **Analyze Implementation Details:**

   * Analyze Kernel Launch syntax: kernel\<\<\<grid, block, sharedMem, stream\>\>\>()  
   * Trace data flow from Host (CPU) to Device (GPU) and back  
   * Document existing synchronization primitives (\_\_syncthreads(), atomic operations)

   **Identify Implementation Patterns:**

   * Look for reduction patterns, tiling strategies, and stream pipelining  
   * Look for macro usage for error checking (e.g., cudaCheckError)

   **IMPORTANT**: You are a documentarian. If you see inefficient global memory access, you must describe *how it accesses memory*, not that it is slow.**Search Historical Context (Thoughts):**

   * Search the thoughts/ directory to discover existing documentation or benchmarks  
   * specific documents to understand *why* certain grid/block sizes were chosen

   **Web Research (only if user explicitly asks):**

   * Use web search for CUDA API documentation versions or library specifics (cuBLAS, cuDNN)  
4. **Synthesize findings:**  
   * Compile all gathered information into a structured view of the GPU pipeline  
   * Connect the Host preparation code to the specific Kernels they launch  
   * Explicitly map variable names on Host to argument names on Device  
   * Verify file paths (distinguish between headers .cuh and source .cu)  
   * Highlight structural decisions (e.g., "Project uses Unified Memory exclusively" or "Project uses explicit streams")  
5. **Gather metadata for the research document:**  
   * **Date**: Get current date/time  
   * **Researcher**: Get your current identity or "AI Agent"  
   * **Git Info**: Run the following commands:  
     * git rev-parse HEAD (for git\_commit)  
     * git branch \--show-current (for branch)  
     * git config user.name (for last\_updated\_by fallback)  
   * **Filename Generation**: thoughts/shared/research/YYYY-MM-DD-ENG-XXXX-description.md  
     * YYYY-MM-DD is today's date  
     * ENG-XXXX is the ticket number (omit if no ticket)  
     * description is a brief kebab-case description of the research topic  
     * Example: 2025-01-08-matrix-multiplication-kernels.md  
6. **Generate research document:**  
   * Use the metadata gathered in step 5  
   * Structure the document with YAML frontmatter followed by content:  
     \---  
     date: \[Current date and time with timezone in ISO format\]  
     researcher: \[Researcher name\]  
     git\_commit: \[Current commit hash\]  
     branch: \[Current branch name\]  
     repository: \[Repository name\]  
     topic: "\[User's Question/Topic\]"  
     tags: \[research, cuda, gpu, \[kernel-name\]\]  
     status: complete  
     last\_updated: \[Current date in YYYY-MM-DD format\]  
     last\_updated\_by: \[Researcher name\]  
     \---

     \# Research: \[User's Question/Topic\]

     \*\*Date\*\*: \[Current date and time with timezone from step 5\]  
     \*\*Researcher\*\*: \[Researcher name\]  
     \*\*Git Commit\*\*: \[Current commit hash from step 5\]  
     \*\*Branch\*\*: \[Current branch name from step 5\]

     \#\# Research Question  
     \[Original user query\]

     \#\# Summary  
     \[High-level documentation of the CUDA implementation found. Briefly explain the Host-Device relationship regarding this topic.\]

     \#\# Detailed Findings

     \#\#\# Host-Side Implementation  
     \- \*\*Initialization\*\*: How contexts/devices are selected (\`file.cpp:line\`)  
     \- \*\*Memory Management\*\*: Allocation patterns (\`cudaMalloc\` vs \`cudaMallocManaged\`)  
     \- \*\*Data Movement\*\*: Explicit copies (\`cudaMemcpy\`) or Zero-Copy flows  
     \- \*\*Kernel Dispatch\*\*: Configuration of grid/block dimensions

     \#\#\# Device-Side Implementation (Kernels)  
     \- \*\*Kernel\*\*: \`KernelName\` (\`file.cu:line\`)  
     \- \*\*Logic\*\*: Description of the thread logic and math operations  
     \- \*\*Memory Access\*\*: How global/shared/constant memory is utilized  
     \- \*\*Synchronization\*\*: Use of barriers, atomics, or warp shuffles

     \#\#\# Build & Configuration  
     \- \*\*CMake/Make\*\*: NVCC flags detected (e.g., \`-arch=sm\_80\`, \`--use\_fast\_math\`)  
     \- \*\*Dependencies\*\*: External libraries used (Thrust, cuBLAS, etc.)

     \#\# Code References  
     \- \`src/kernels/compute.cu:123\` \- Kernel definition  
     \- \`include/cuda\_utils.cuh:45\` \- Helper templates

     \#\# Architecture Documentation  
     \[Current patterns: e.g., "The system uses a producer-consumer pattern with CUDA Streams"\]

     \#\# Historical Context (from thoughts/)  
     \[Relevant insights from thoughts/ directory\]

     \#\# Open Questions  
     \[Any ambiguities in the data flow or thread indexing\]

7. **Add GitHub permalinks (if applicable):**  
   * Check if on main branch or if commit is pushed: git branch \--show-current and git status  
   * If on main/master or pushed, generate GitHub permalinks  
   * Replace local file references with permalinks in the document  
8. **Sync and present findings:**  
   * Run humanlayer thoughts sync to sync the thoughts directory  
   * Present a concise summary of findings to the user  
   * Ask if they need clarification on specific kernels or memory flows  
9. **Handle follow-up questions:**  
   * If the user has follow-up questions, append to the same research document  
   * Update frontmatter and add \#\# Follow-up Research \[timestamp\]  
   * Perform additional research as needed  
   * Continue updating the document and syncing

## **Important notes:**

* **CUDA Specifics**: Distinguish clearly between code running on CPU (Host) and GPU (Device)  
* **Execution**: Perform all research steps yourself sequentially; do not attempt to spawn external agents.  
* **Files**: Pay special attention to .cu (implementation) vs .cuh (headers/templates)  
* **Dimensions**: Always document Grid and Block dimensions if they are static/hardcoded  
* **No Evaluation**: Describe *what* the code does (e.g., "Thread 0 writes to global memory"), not *how well* it does it  
* **Context**: Remember that \_\_global\_\_ functions are entry points, and \_\_device\_\_ functions are internal helpers  
* **Path handling**: The thoughts/searchable/ directory contains hard links for searching  
  * Always document paths by removing ONLY "searchable/"  
* **Critical ordering**: Follow the numbered steps exactly