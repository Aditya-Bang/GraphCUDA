edge case

```bash
ncu --import ncu_profiles/accurate/ncu_triton_edge_case_new.ncu-rep
[1013347] python3.11@127.0.0.1
  fused_spmm_gemm_relu_small_n_kernel (64, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         7.59
    SM Frequency                    Ghz         1.41
    Elapsed Cycles                cycle      166,261
    Memory Throughput                 %         3.29
    DRAM Throughput                   %         3.29
    Duration                         us       117.98
    L1/TEX Cache Throughput           %        26.71
    L2 Cache Throughput               %         1.27
    SM Active Cycles              cycle     3,717.88
    Compute (SM) Throughput           %         0.42
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.8 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel        
          Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details  
          on roofline analysis.                                                                                         

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        25.17
    Dropped Samples                sample            0
    Maximum Sampling Interval          us            1
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.39
    Executed Ipc Elapsed  inst/cycle         0.01
    Issue Slots Busy               %         9.85
    Issued Ipc Active     inst/cycle         0.39
    SM Busy                        %        13.12
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 86.88%                                                                                    
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.             

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s        24.00
    Mem Busy                               %         1.27
    Max Bandwidth                          %         3.29
    L1/TEX Hit Rate                        %         0.64
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        30.71
    Mem Pipes Busy                         %         0.42
    ---------------------------- ----------- ------------

    Section: Memory Workload Analysis Tables
    OPT   Est. Speedup: 0.6487%                                                                                         
          The memory access pattern for global loads from DRAM might not be optimal. On average, only 24.0 of the 32    
          bytes transmitted per sector are utilized by each thread. This applies to the 79.3% of sectors missed in L2.  
          This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced  
          global loads.                                                                                                 
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 3.554%                                                                                          
          The memory access pattern for shared stores might not be optimal and causes on average a 2.4 - way bank       
          conflict across all 3616 shared store requests.This results in 1144 bank conflicts,  which represent 13.31%   
          of the overall 8597 wavefronts for shared stores. Check the Source Counters section for uncoalesced shared    
          stores.                                                                                                       

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         9.93
    Issued Warp Per Scheduler                        0.10
    No Eligible                            %        90.07
    Active Warps Per Scheduler          warp         1.00
    Eligible Warps Per Scheduler        warp         0.10
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 90.07%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 10.1 cycles. This might leave hardware resources underutilized and may lead to    
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 1.00 active warps per scheduler, but only an average of 0.10 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, avoid possible load imbalances due to highly different execution durations per      
          warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.      

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        10.07
    Warp Cycles Per Executed Instruction           cycle        10.29
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    31.02
    ---------------------------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst       358.62
    Executed Instructions                           inst      120,496
    Avg. Issued Instructions Per Scheduler          inst       366.27
    Issued Instructions                             inst      123,066
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 1.207%                                                                                          
          This kernel executes 0 fused and 1024 non-fused FP32 instructions. By converting pairs of non-fused           
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP32 instructions.         

    Section: Launch Statistics
    -------------------------------- --------------- -----------------
    Metric Name                          Metric Unit      Metric Value
    -------------------------------- --------------- -----------------
    Block Size                                                     128
    Function Cache Configuration                     CachePreferShared
    Grid Size                                                       64
    Registers Per Thread             register/thread               111
    Shared Memory Configuration Size           Kbyte            102.40
    Driver Shared Memory Per Block       Kbyte/block              1.02
    Dynamic Shared Memory Per Block      Kbyte/block             67.58
    Static Shared Memory Per Block        byte/block                 0
    # SMs                                         SM                84
    Stack Size                                                   1,024
    Threads                                   thread             8,192
    # TPCs                                                          42
    Enabled TPC IDs                                                all
    Uses Green Context                                               0
    Waves Per SM                                                  0.76
    -------------------------------- --------------- -----------------

    OPT   Est. Speedup: 23.81%                                                                                          
          The grid for this launch is configured to execute only 64 blocks, which is less than the GPU's 84             
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            1
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp            4
    Theoretical Occupancy                     %         8.33
    Achieved Occupancy                        %         8.30
    Achieved Active Warps Per SM           warp         3.99
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 90.07%                                                                                          
          The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of       
          shared memory.                                                                                                

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    29,490.67
    Total DRAM Elapsed Cycles        cycle   10,743,808
    Average L1 Active Cycles         cycle     3,717.88
    Total L1 Elapsed Cycles          cycle   13,930,740
    Average L2 Active Cycles         cycle    45,051.56
    Total L2 Elapsed Cycles          cycle    7,555,968
    Average SM Active Cycles         cycle     3,717.88
    Total SM Elapsed Cycles          cycle   13,930,740
    Average SMSP Active Cycles       cycle     3,686.80
    Total SMSP Elapsed Cycles        cycle   55,722,960
    -------------------------- ----------- ------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.01
    Branch Instructions              inst        1,340
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 0.07417%                                                                                        
          This kernel has uncoalesced shared accesses resulting in a total of 2560 excessive wavefronts (3% of the      
          total 77376 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations.     
          The CUDA Best Practices Guide                                                                                 
           (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
          -ab) has an example on optimizing shared memory accesses.
```

regular
```bash
ncu --import ncu_profiles/accurate/ncu_triton_regular_new.ncu-rep
[1013213] python3.11@127.0.0.1
  fused_spmm_gemm_relu_small_n_kernel (64, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         7.58
    SM Frequency                    Ghz         1.38
    Elapsed Cycles                cycle       27,701
    Memory Throughput                 %        36.74
    DRAM Throughput                   %        21.13
    Duration                         us        19.90
    L1/TEX Cache Throughput           %        36.73
    L2 Cache Throughput               %        36.74
    SM Active Cycles              cycle    17,704.77
    Compute (SM) Throughput           %        18.47
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.8 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel        
          Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details  
          on roofline analysis.                                                                                         

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        25.17
    Dropped Samples                sample            0
    Maximum Sampling Interval          us            1
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.54
    Executed Ipc Elapsed  inst/cycle         0.36
    Issue Slots Busy               %        13.56
    Issued Ipc Active     inst/cycle         0.54
    SM Busy                        %        22.03
    -------------------- ----------- ------------

    INF   Tensor is the highest-utilized pipeline (22.0%) based on active cycles, taking into account the rates of its  
          different instructions. It is the logical aggregation of individual tensor pipelines. It's dominated by its   
          Tensor (FP) sub-pipeline. It is well-utilized, but should not be a bottleneck.                                

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s       153.81
    Mem Busy                               %        36.74
    Max Bandwidth                          %        36.44
    L1/TEX Hit Rate                        %         0.66
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        84.67
    Mem Pipes Busy                         %        18.47
    ---------------------------- ----------- ------------

    Section: Memory Workload Analysis Tables
    OPT   Est. Speedup: 11.63%                                                                                          
          The memory access pattern for global loads from L2 might not be optimal. On average, only 21.8 of the 32      
          bytes transmitted per sector are utilized by each thread. This applies to the 99.3% of sectors missed in      
          L1TEX. This could possibly be caused by a stride between threads. Check the Source Counters section for       
          uncoalesced global loads.                                                                                     
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 6.16%                                                                                           
          The memory access pattern for shared stores might not be optimal and causes on average a 5.8 - way bank       
          conflict across all 21760 shared store requests.This results in 21139 bank conflicts,  which represent        
          16.77% of the overall 126031 wavefronts for shared stores. Check the Source Counters section for uncoalesced  
          shared stores.                                                                                                

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        13.42
    Issued Warp Per Scheduler                        0.13
    No Eligible                            %        86.58
    Active Warps Per Scheduler          warp         0.99
    Eligible Warps Per Scheduler        warp         0.13
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 63.26%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 7.5 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 0.99 active warps per scheduler, which already limits the scheduler to less than a warp per instruction.   

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         7.41
    Warp Cycles Per Executed Instruction           cycle         7.44
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    30.80
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 33.27%                                                                                          
          On average, each warp of this workload spends 2.5 cycles being stalled waiting on a fixed latency execution   
          dependency. Typically, this stall reason should be very low and only shows up as a top contributor in         
          already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the       
          number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to         
          lower-latency instructions, e.g. by making use of fast math compiler options. This stall type represents      
          about 33.3% of the total average of 7.4 cycles between issuing two instructions.                              
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst     2,391.62
    Executed Instructions                           inst      803,584
    Avg. Issued Instructions Per Scheduler          inst     2,401.21
    Issued Instructions                             inst      806,805
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 1.734%                                                                                          
          This kernel executes 0 fused and 1024 non-fused FP32 instructions. By converting pairs of non-fused           
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP32 instructions.         

    Section: Launch Statistics
    -------------------------------- --------------- -----------------
    Metric Name                          Metric Unit      Metric Value
    -------------------------------- --------------- -----------------
    Block Size                                                     128
    Function Cache Configuration                     CachePreferShared
    Grid Size                                                       64
    Registers Per Thread             register/thread               111
    Shared Memory Configuration Size           Kbyte            102.40
    Driver Shared Memory Per Block       Kbyte/block              1.02
    Dynamic Shared Memory Per Block      Kbyte/block             67.58
    Static Shared Memory Per Block        byte/block                 0
    # SMs                                         SM                84
    Stack Size                                                   1,024
    Threads                                   thread             8,192
    # TPCs                                                          42
    Enabled TPC IDs                                                all
    Uses Green Context                                               0
    Waves Per SM                                                  0.76
    -------------------------------- --------------- -----------------

    OPT   Est. Speedup: 23.81%                                                                                          
          The grid for this launch is configured to execute only 64 blocks, which is less than the GPU's 84             
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            1
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp            4
    Theoretical Occupancy                     %         8.33
    Achieved Occupancy                        %         8.37
    Achieved Active Warps Per SM           warp         4.02
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 63.26%                                                                                          
          The 1.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (8.3%) is limited by the required amount of       
          shared memory.                                                                                                

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    31,890.67
    Total DRAM Elapsed Cycles        cycle    1,811,456
    Average L1 Active Cycles         cycle    17,704.77
    Total L1 Elapsed Cycles          cycle    2,234,184
    Average L2 Active Cycles         cycle    18,164.35
    Total L2 Elapsed Cycles          cycle    1,272,768
    Average SM Active Cycles         cycle    17,704.77
    Total SM Elapsed Cycles          cycle    2,234,184
    Average SMSP Active Cycles       cycle    17,895.49
    Total SMSP Elapsed Cycles        cycle    8,936,736
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 18.37%                                                                                          
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 27.60% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 20.06%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 29.82% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 18.37%                                                                                          
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 27.60% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.01
    Branch Instructions              inst        5,120
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 0.2995%                                                                                         
          This kernel has uncoalesced global accesses resulting in a total of 2026 excessive sectors (0% of the total   
          463365 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations.    
          The CUDA Programming Guide                                                                                    
          (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional      
          information on reducing uncoalesced device memory accesses.                                                   
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 0.2915%                                                                                         
          This kernel has uncoalesced shared accesses resulting in a total of 2560 excessive wavefronts (0% of the      
          total 584624 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations.    
          The CUDA Best Practices Guide                                                                                 
           (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
          -ab) has an example on optimizing shared memory accesses. 
```

switch loop
```bash
ncu --import ncu_profiles/accurate/ncu_triton_switch_loop_new.ncu-rep
[1012457] python3.11@127.0.0.1
  fused_spmm_gemm_relu_small_n_switch_loop_kernel (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         7.59
    SM Frequency                    Ghz         1.40
    Elapsed Cycles                cycle       47,401
    Memory Throughput                 %        29.75
    DRAM Throughput                   %        12.53
    Duration                         us        33.79
    L1/TEX Cache Throughput           %        42.65
    L2 Cache Throughput               %        21.57
    SM Active Cycles              cycle    33,607.45
    Compute (SM) Throughput           %        25.45
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.4 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: GPU Speed Of Light Roofline Chart
    INF   The ratio of peak float (fp32) to double (fp64) performance on this device is 64:1. The workload achieved     
          close to 0% of this device's fp32 peak performance and 0% of its fp64 peak performance. See the Kernel        
          Profiling Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details  
          on roofline analysis.                                                                                         

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        25.17
    Dropped Samples                sample            0
    Maximum Sampling Interval          us            1
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.94
    Executed Ipc Elapsed  inst/cycle         0.66
    Issue Slots Busy               %        23.65
    Issued Ipc Active     inst/cycle         0.95
    SM Busy                        %        23.65
    -------------------- ----------- ------------

    INF   ALU is the highest-utilized pipeline (20.7%) based on active cycles, taking into account the rates of its     
          different instructions. It executes integer and logic operations. It is well-utilized, but should not be a    
          bottleneck.                                                                                                   

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s        91.27
    Mem Busy                               %        29.75
    Max Bandwidth                          %        25.45
    L1/TEX Hit Rate                        %        24.62
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        84.77
    Mem Pipes Busy                         %        25.45
    ---------------------------- ----------- ------------

    Section: Memory Workload Analysis Tables
    OPT   Est. Speedup: 5.777%                                                                                          
          The memory access pattern for global loads from L1TEX might not be optimal. On average, only 25.8 of the 32   
          bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between   
          threads. Check the Source Counters section for uncoalesced global loads.                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 5.356%                                                                                          
          The memory access pattern for shared stores might not be optimal and causes on average a 3.7 - way bank       
          conflict across all 92160 shared store requests.This results in 42982 bank conflicts,  which represent        
          12.56% of the overall 342266 wavefronts for shared stores. Check the Source Counters section for uncoalesced  
          shared stores.                                                                                                

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        23.51
    Issued Warp Per Scheduler                        0.24
    No Eligible                            %        76.49
    Active Warps Per Scheduler          warp         1.99
    Eligible Warps Per Scheduler        warp         0.29
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 70.25%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 4.3 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 1.99 active warps per scheduler, but only an average of 0.29 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         8.48
    Warp Cycles Per Executed Instruction           cycle         8.50
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    30.26
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 31.83%                                                                                          
          On average, each warp of this workload spends 2.7 cycles being stalled waiting for a scoreboard dependency on 
          a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited     
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the        
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by        
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently     
          used data to shared memory. This stall type represents about 31.8% of the total average of 8.5 cycles         
          between issuing two instructions.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason.                                                                                         

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Avg. Executed Instructions Per Scheduler        inst     7,926.86
    Executed Instructions                           inst    2,663,424
    Avg. Issued Instructions Per Scheduler          inst     7,948.95
    Issued Instructions                             inst    2,670,848
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 2.6%                                                                                            
          This kernel executes 0 fused and 2048 non-fused FP32 instructions. By converting pairs of non-fused           
          instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point),           
          higher-throughput equivalent, the achieved FP32 performance could be increased by up to 50% (relative to its  
          current performance). Check the Source page to identify where this kernel executes FP32 instructions.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     64
    Registers Per Thread             register/thread             102
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block           24.58
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              84
    Stack Size                                                 1,024
    Threads                                   thread          16,384
    # TPCs                                                        42
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.38
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 23.81%                                                                                          
          The grid for this launch is configured to execute only 64 blocks, which is less than the GPU's 84             
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            2
    Block Limit Shared Mem                block            2
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %        33.33
    Achieved Occupancy                        %        16.66
    Achieved Active Warps Per SM           warp         7.99
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 50.03%                                                                                          
          The difference between calculated theoretical (33.3%) and measured achieved occupancy (16.7%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 66.67%                                                                                          
          The 4.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (33.3%) is limited by the number of required      
          registers, and the required amount of shared memory.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    32,126.67
    Total DRAM Elapsed Cycles        cycle    3,076,096
    Average L1 Active Cycles         cycle    33,607.45
    Total L1 Elapsed Cycles          cycle    4,047,228
    Average L2 Active Cycles         cycle    26,184.27
    Total L2 Elapsed Cycles          cycle    2,164,032
    Average SM Active Cycles         cycle    33,607.45
    Total SM Elapsed Cycles          cycle    4,047,228
    Average SMSP Active Cycles       cycle    33,816.45
    Total SMSP Elapsed Cycles        cycle   16,188,912
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 17.4%                                                                                           
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum   
          instance value is 24.95% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 17.52%                                                                                          
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum 
          instance value is 24.97% above the average, while the minimum instance value is 100.00% below the average.    
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 17.4%                                                                                           
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 24.95% above the average, while the minimum instance value is 100.00% below the     
          average.                                                                                                      

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.01
    Branch Instructions              inst       33,280
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------

    OPT   Est. Speedup: 2.887%                                                                                          
          This kernel has uncoalesced global accesses resulting in a total of 30320 excessive sectors (5% of the total  
          610016 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations.    
          The CUDA Programming Guide                                                                                    
          (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional      
          information on reducing uncoalesced device memory accesses.                                                   
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 0.1833%                                                                                         
          This kernel has uncoalesced shared accesses resulting in a total of 2560 excessive wavefronts (0% of the      
          total 974256 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations.    
          The CUDA Best Practices Guide                                                                                 
           (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c
          -ab) has an example on optimizing shared memory accesses.
```
