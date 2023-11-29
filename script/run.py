import time
import os
import sys, getopt
import numpy as np

data_dir = '/home/wzm/rtod/data/'
ugt_dict = {0:'update', 1:'rebuild'}
# build_type = 'Debug'
build_type = 'Release'
def gau_1d(compaction = 0, update_gas_type = 1, optimization = 2):
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/{ugt_dict[update_gas_type]}/{logtime}-GAU.log"
    args = f'--n 1000000 --R 0.028 --K 50 --window 100000 --slide 5000'
    cmd = f"./build/bin/optixScan {args} -f {data_dir}gaussian.txt >> {output_file}" # set args --n 1000000 --R 0.028 --K 50 --window 100000 --slide 5000 -f data/gaussian.txt
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION=1 \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION={optimization} && \
                make')
    os.system(cmd)
    
def stk_1d(compaction = 0, update_gas_type = 1, optimization = 2):
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/{ugt_dict[update_gas_type]}/{logtime}-STK.log"
    args = f'--n 1048572 --R 0.45 --K 50 --window 100000 --slide 5000'
    cmd = f"./build/bin/optixScan {args} -f {data_dir}stock.txt >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION=1 \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION={optimization} && \
                make')
    os.system(cmd)

def tao_3d(compaction = 0, update_gas_type = 1, optimization = 2):
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/{ugt_dict[update_gas_type]}/{logtime}-TAO.log"
    args = f'--n 575468 --R 1.9 --K 50 --window 10000 --slide 500'
    cmd = f"./build/bin/optixScan {args} -f {data_dir}tao.txt >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION=3 \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION={optimization} && \
                make')
    os.system(cmd)

# def tao_3d_vary_points_num_building_bvh(compaction = 0, update_gas_type = 1):
#     logtime = time.strftime("%y%m%d-%H%M%S")
#     output_file = f"log/{ugt_dict[update_gas_type]}/fast_build/vary_points_num_build_bvh/{logtime}-TAO.log"
#     points_num_list = [2500, 5000, 10000, 20000, 40000]
#     os.system(f'cd build/ && cmake ../src/ -D DIMENSION=3 \
#                 -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
#                 -D CMAKE_BUILD_TYPE={build_type} && \
#                 make')
#     for points_num in points_num_list:
#         args = f'--n 575468 --R 1.9 --K 50 --window {points_num} --slide 500 --launch_ray_num 2500' # window 变小时，一部分 ray 的相交的 sphere 少于 K 个。实验方法无问题，这恰说明构成 BVH 的 point 少时的影响
#         cmd = f"./build/bin/optixScan {args} -f {data_dir}tao.txt >> {output_file}"
#         print(cmd)
#         os.system(cmd)

# def tao_3d_vary_rays_istests_per_ray(compaction = 0, update_gas_type = 1):
#     logtime = time.strftime("%y%m%d-%H%M%S")
#     output_file = f"log/{ugt_dict[update_gas_type]}/fast_build/vary_rays_istests_per_ray/{logtime}-TAO.log"
#     rays_num_list = [2000, 4000, 6000, 8000, 10000]
#     k_list = [10, 30, 50, 70]
#     os.system(f'cd build/ && cmake ../src/ -D DIMENSION=3 \
#                 -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
#                 -D CMAKE_BUILD_TYPE={build_type} && \
#                 make')
#     for k in k_list:
#         for rays_num in rays_num_list:
#             args = f'--n 575468 --R 1.9 --K {k} --window 10000 --slide 500 --launch_ray_num {rays_num}'
#             cmd = f"./build/bin/optixScan {args} -f {data_dir}tao.txt >> {output_file}"
#             print(cmd)
#             os.system(cmd)
            
def gau_1d_vary_parameters(Wo=True, So=True, Ro=True, Ko=True):
    logtime = time.strftime("%y%m%d-%H%M%S")
    gau_R       = 0.028
    gau_N       = 1000000
    gau_dim     = 1
    dataset     = 'gaussian'
    window_list = [10000, 50000, 100000, 150000, 200000]
    slide_list  = [0.05, 0.1, 0.2, 0.5, 1.0]
    R_list      = [0.25, 0.5, 1.0, 5.0, 10.0]
    K_list      = [10, 30, 50, 70, 100]
    
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION={gau_dim} \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION=2 && \
                make')
    if Wo:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/window/{logtime}-GAU.log"
        for window in window_list:
            args = f'--n {gau_N} --R {gau_R} --K 50 --window {window} --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if So:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/slide/{logtime}-GAU.log"
        for slide in slide_list:
            args = f'--n {gau_N} --R {gau_R} --K 50 --window 100000 --slide {int(100000 * slide)}'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ro:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/R/{logtime}-GAU.log"
        for R in R_list:
            args = f'--n {gau_N} --R {gau_R * R} --K 50 --window 100000 --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ko:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/K/{logtime}-GAU.log"
        for K in K_list:
            args = f'--n {gau_N} --R {gau_R} --K {K} --window 100000 --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
        
def stk_1d_vary_parameters(Wo=True, So=True, Ro=True, Ko=True):
    logtime = time.strftime("%y%m%d-%H%M%S")
    stk_R       = 0.45
    stk_N       = 1048572
    stk_dim     = 1
    dataset     = 'stock'
    window_list = [10000, 50000, 100000, 150000, 200000]
    slide_list  = [0.05, 0.1, 0.2, 0.5, 1.0]
    R_list      = [0.25, 0.5, 1.0, 5.0, 10.0]
    K_list      = [10, 30, 50, 70, 100]
    
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION={stk_dim} \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION=2 && \
                make')
    if Wo:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/window/{logtime}-STK.log"
        for window in window_list:
            args = f'--n {stk_N} --R {stk_R} --K 50 --window {window} --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if So:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/slide/{logtime}-STK.log"
        for slide in slide_list:
            args = f'--n {stk_N} --R {stk_R} --K 50 --window 100000 --slide {int(100000 * slide)}'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ro:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/R/{logtime}-STK.log"
        for R in R_list:
            args = f'--n {stk_N} --R {stk_R * R} --K 50 --window 100000 --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ko:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/K/{logtime}-STK.log"
        for K in K_list:
            args = f'--n {stk_N} --R {stk_R} --K {K} --window 100000 --slide 5000'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
        
def tao_3d_vary_parameters(Wo=True, So=True, Ro=True, Ko=True):
    logtime = time.strftime("%y%m%d-%H%M%S")
    tao_R       = 1.9
    tao_N       = 575468
    tao_dim     = 3
    dataset     = 'tao'
    window_list = [1000, 5000, 10000, 15000, 20000]
    slide_list  = [0.05, 0.1, 0.2, 0.5, 1.0]
    R_list      = [0.25, 0.5, 1.0, 5.0, 10.0]
    K_list      = [10, 30, 50, 70, 100]
    
    os.system(f'cd build/ && cmake ../src/ -D DIMENSION={tao_dim} \
                -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION=2 && \
                make')
    if Wo:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/window/{logtime}-TAO.log"
        for window in window_list:
            args = f'--n {tao_N} --R {tao_R} --K 50 --window {window} --slide 500'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if So:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/slide/{logtime}-TAO.log"
        for slide in slide_list:
            args = f'--n {tao_N} --R {tao_R} --K 50 --window 10000 --slide {int(10000 * slide)}'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ro:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/R/{logtime}-TAO.log"
        for R in R_list:
            args = f'--n {tao_N} --R {tao_R * R} --K 50 --window 10000 --slide 500'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)
    if Ko:
        output_file = f"log/{ugt_dict[update_gas_type]}/vary_params/K/{logtime}-TAO.log"
        for K in K_list:
            args = f'--n {tao_N} --R {tao_R} --K {K} --window 10000 --slide 500'
            cmd = f"./build/bin/optixScan {args} -f {data_dir}{dataset}.txt >> {output_file}"
            print(cmd)
            os.system(cmd)


def stk_1d_vary_ray_load(compaction = 0, update_gas_type = 1):
    logtime = time.strftime("%y%m%d-%H%M%S")
    output_file = f"log/{ugt_dict[update_gas_type]}/fast_build/vary_ray_load/{logtime}-STK-vary_ray_load.log"
    K_list = [1, 4, 9, 29, 49]
    
    for K in K_list:
        args = f'--n 1048572 --R 0.45 --K {K} --window 100000 --slide 5000 --launch_ray_num 800'
        cmd = f"./build/bin/optixScan {args} -f {data_dir}stock.txt >> {output_file}"
        print(cmd)
        os.system(f'cd build/ && cmake ../src/ -D DIMENSION=1 \
                    -D COMPACTION={compaction} -D UPDATE_GAS_TYPE={update_gas_type} \
                    -D CMAKE_BUILD_TYPE={build_type} -D OPTIMIZATION=0 && \
                    make')
        os.system(cmd)


compaction = 0
update_gas_type = 1

stk_1d_vary_ray_load() # Figure 8: [Time] detect outlier in log <-> ray traversal


gau_1d(compaction, update_gas_type, 0) # Figure 15
stk_1d(compaction, update_gas_type, 0) # Figure 15
tao_3d(compaction, update_gas_type, 0) # Figure 15

gau_1d(compaction, update_gas_type, 1) # Figure 15
stk_1d(compaction, update_gas_type, 1) # Figure 15
tao_3d(compaction, update_gas_type, 1) # Figure 15

gau_1d(compaction, update_gas_type, 2) # Figure 9, 10, 15, 16
stk_1d(compaction, update_gas_type, 2) # Figure 9, 10, 15, 16
tao_3d(compaction, update_gas_type, 2) # Figure 9, 10, 15, 16


gau_1d_vary_parameters() # Figure 11, 12, 13, 14
stk_1d_vary_parameters() # Figure 11, 12, 13, 14
tao_3d_vary_parameters() # Figure 11, 12, 13, 14