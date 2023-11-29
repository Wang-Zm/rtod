# RTOD

## Prerequisites
Your GPU must support NVIDIA RTX (hardware raytracing acceleration). For consumer GPUs, this applies to NVIDIA RTX 2000, 3000, and 4000 series, but there are workstation and data-center GPUs that also support RTX.

## Configuration

- Edit script/run.py to set experiment parameters (dataset, window, slide, R, K, etc.)

- Run all experiments
    ```
    $ cd .
    $ mkdir build
    $ python script/run.py 
    ```

- Example output for STK dataset under default parameters with two techniques
    ```
    DIM[0]: 0, 9930
    [Mem] data_h2d: 4
    [Mem-make_gas] kGenAABB: 4
    [Mem-make_gas] d_temp_buffer_gas: 8
    [Mem-make_gas] d_gas_output_buffer: 4
    [Mem-make_gas] optixAccelBuild: 0
    Final GAS size: 2.395630 MB
    [Mem] make_gas: 16
    [Mem] make_module: 0
    [Mem] make_program_groups: 0
    [Mem] make_pipeline: 0
    [Mem] make_sbt: 2
    [Mem] initialize_params: 4
    Dimension: 1
    Compaction: 0
    Update GAS type: Rebuild
    data num: 1048572
    R: 0.45, R2: 0.2025
    K: 50
    Window size: 100000
    Slide size: 5000
    Input file: /home/wzm/rtod/data/stock.txt
    Optimization: 2

    ###########   Time  ##########
    [Time] copy new points h2d: 0.0128129 ms
    [Time] copy filtered points h2d: 0.00242849 ms
    [Time] copy outlier d2h: 0.005933 ms
    [Time] prepare cell: 0.0846664 ms
    [Time] build BVH: 0.188458 ms
    [Time] detect outlier: 0.0846083 ms
    [Time] total time for a slide: 0.382371 ms
    ##############################

    [Mem] Host cpu memery used(MB): 3.00781
    [Mem] Device memory used for data(MB): 26
    BVH Node / slide: 8953
    ```