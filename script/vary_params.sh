#!/bin/bash

dir_path=log/vary_params
data_dir=data
check=check_ # NULL or "check_"
compaction=0
update_gas_type=1
build_type=Release

# TAO
function run_tao() {
    window_list=(1000 5000 10000 15000 20000)
    slide_list=(0.05 0.1 0.2 0.5 1.0)
    R_list=(0.25 0.5 1.0 5.0 10.0)
    K_list=(10 30 50 70 100)
    tao_R=1.9
    tao_N=575468
    tao_dim=3
    dataset=tao
    cd build/ && cmake ../src/ -D DIMENSION=${tao_dim} \
        -D COMPACTION=${compaction} -D UPDATE_GAS_TYPE=${update_gas_type} \
        -D CMAKE_BUILD_TYPE=${build_type} -D OPTIMIZATION=2 && \
        make
    cd ..
    # for w in ${window_list[*]} 
    # do
    #     echo "processing tao, vary window, window = ${w}"
    #     args="--n ${tao_N} --R ${tao_R} --K 50 --window ${w} --slide 500 -f ${data_dir}/${dataset}.txt"
    #     ./build/bin/optixScan ${args} > ${dir_path}/${check}w_tao_${w}.log
    # done
    # for s in ${slide_list[*]}
    # do
    #     real_s=`echo "scale=0; ${s}*10000/1" | bc`
    #     echo "processing tao, vary slide, slide = ${real_s}"
    #     args="--n ${tao_N} --R ${tao_R} --K 50 --window 10000 --slide ${real_s} -f ${data_dir}/${dataset}.txt"
    #     ./build/bin/optixScan ${args} > ${dir_path}/${check}s_tao_${real_s}.log
    # done
    for r in ${R_list[*]}
    do
        real_r=`echo "scale=3; ${r}*${tao_R}" | bc`
        echo "processing tao, vary r, r = ${real_r}"
        args="--n ${tao_N} --R ${real_r} --K 50 --window 10000 --slide 500 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}r_tao_${real_r}.log
    done
    for k in ${K_list[*]}
    do
        echo "processing tao, vary k, k = ${k}"
        args="--n ${tao_N} --R ${tao_R} --K ${k} --window 10000 --slide 500 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}k_tao_${k}.log
    done
}

# GAU
function run_gau() {
    window_list=(10000 50000 100000 150000 200000)
    slide_list=(0.05 0.1 0.2 0.5 1.0)
    R_list=(0.25 0.5 1.0 5.0 10.0)
    K_list=(10 30 50 70 100)
    gau_R=0.028
    gau_N=1000000
    gau_dim=1
    dataset=gaussian
    cd build/ && cmake ../src/ -D DIMENSION=${gau_dim} \
        -D COMPACTION=${compaction} -D UPDATE_GAS_TYPE=${update_gas_type} \
        -D CMAKE_BUILD_TYPE=${build_type} -D OPTIMIZATION=2 && \
        make
    cd ..
    for w in ${window_list[*]} 
    do
        echo "processing gau, vary window, window = ${w}"
        args="--n ${gau_N} --R ${gau_R} --K 50 --window ${w} --slide 5000 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}w_gau_${w}.log
    done
    for s in ${slide_list[*]}
    do
        real_s=`echo "scale=0; ${s}*100000/1" | bc`
        echo "processing gau, vary slide, slide = ${real_s}"
        args="--n ${gau_N} --R ${gau_R} --K 50 --window 100000 --slide ${real_s} -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}s_gau_${real_s}.log
    done
    for r in ${R_list[*]}
    do
        real_r=`echo "scale=3; ${r}*${gau_R}" | bc`
        echo "processing gau, vary r, r = ${real_r}"
        args="--n ${gau_N} --R ${real_r} --K 50 --window 100000 --slide 5000 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}r_gau_${real_r}.log
    done
    for k in ${K_list[*]}
    do
        echo "processing gau, vary k, k = ${k}"
        args="--n ${gau_N} --R ${gau_R} --K ${k} --window 100000 --slide 5000 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}k_gau_${k}.log
    done
}

# STK
function run_stk() {
    window_list=(10000 50000 100000 150000 200000)
    slide_list=(0.05 0.1 0.2 0.5 1.0)
    R_list=(0.25 0.5 1.0 5.0 10.0)
    K_list=(10 30 50 70 100)
    stk_R=0.45
    stk_N=1048572
    stk_dim=1
    dataset=stock
    cd build/ && cmake ../src/ -D DIMENSION=${stk_dim} \
        -D COMPACTION=${compaction} -D UPDATE_GAS_TYPE=${update_gas_type} \
        -D CMAKE_BUILD_TYPE=${build_type} -D OPTIMIZATION=2 && \
        make
    cd ..
    # for w in ${window_list[*]} 
    # do
    #     echo "processing stk, vary window, window = ${w}"
    #     args="--n ${stk_N} --R ${stk_R} --K 50 --window ${w} --slide 5000 -f ${data_dir}/${dataset}.txt"
    #     ./build/bin/optixScan ${args} > ${dir_path}/${check}w_stk_${w}.log
    # done
    # for s in ${slide_list[*]}
    # do
    #     real_s=`echo "scale=0; ${s}*100000/1" | bc`
    #     echo "processing stk, vary slide, slide = ${real_s}"
    #     args="--n ${stk_N} --R ${stk_R} --K 50 --window 100000 --slide ${real_s} -f ${data_dir}/${dataset}.txt"
    #     ./build/bin/optixScan ${args} > ${dir_path}/${check}s_stk_${real_s}.log
    # done
    for r in ${R_list[*]}
    do
        real_r=`echo "scale=4; ${r}*${stk_R}" | bc`
        echo "processing stk, vary r, r = ${real_r}"
        args="--n ${stk_N} --R ${real_r} --K 50 --window 100000 --slide 5000 -f ${data_dir}/${dataset}.txt"
        ./build/bin/optixScan ${args} > ${dir_path}/${check}r_stk_${real_r}.log
        break
    done
    # for k in ${K_list[*]}
    # do
    #     echo "processing stk, vary k, k = ${k}"
    #     args="--n ${stk_N} --R ${stk_R} --K ${k} --window 100000 --slide 5000 -f ${data_dir}/${dataset}.txt"
    #     ./build/bin/optixScan ${args} > ${dir_path}/${check}k_stk_${k}.log
    # done
}

run_tao
# run_gau
# run_stk