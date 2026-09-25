FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ccache \
    build-essential \
    cppcheck \
    gdb \
    git \
    libffi-dev \
    ninja-build \
    ocl-icd-libopencl1 \
    pkg-config \
    python3 \
    python3-pip \
    python-is-python3 \
    zstd \
    unzip \
    libnuma-dev \
    wget \
    cmake \
    lsb-release software-properties-common gnupg \
    libboost-all-dev \
    libzmq3-dev \
&& rm -rf /var/lib/apt/lists/*


############################
#       Install ROCm 6.2   #
############################
RUN wget --progress=bar:force -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
&& echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2 noble main" | tee /etc/apt/sources.list.d/rocm.list \
&& printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600 \
&& apt update -y \
&& apt install -y rocm-dev

RUN mkdir -p /opt/sycl/source \
&& git clone https://github.com/intel/llvm.git \
&& cd llvm \
&& git checkout 2023-WW27 \
&& mv /llvm /opt/sycl/source

############################
#Install OpenCL CPU runtime to get backend available in DPCPP
RUN cd /opt \
&& wget https://github.com/intel/llvm/releases/download/2023-WW27/oclcpuexp-2023.16.6.0.28_rel.tar.gz

RUN cd /opt \
&& wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.10.0/oneapi-tbb-2021.10.0-lin.tgz

RUN mkdir -p /opt/intel/oclcpuexp_2023.16.6.0.28_rel \
&& cd /opt/intel/oclcpuexp_2023.16.6.0.28_rel \
&& mv /opt/oclcpuexp-2023.16.6.0.28_rel.tar.gz /opt/intel/oclcpuexp_2023.16.6.0.28_rel \
&& tar -zxvf oclcpuexp-2023.16.6.0.28_rel.tar.gz

RUN echo /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64/libintelocl.so > /etc/OpenCL/vendors/intel_expcpu.icd

RUN mkdir -p /opt/intel \
&& cd /opt/intel && mv /opt/oneapi-tbb-2021.10.0-lin.tgz /opt/intel \
&& tar -zxvf oneapi-tbb*lin.tgz

RUN rm /opt/intel/oclcpuexp_2023.16.6.0.28_rel/oclcpuexp-2023.16.6.0.28_rel.tar.gz \
&& rm /opt/intel/oneapi-tbb-2021.10.0-lin.tgz

RUN ln -s /opt/intel/oneapi-tbb-2021.10.0/lib/intel64/gcc4.8/libtbb.so /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64 \
&& ln -s /opt/intel/oneapi-tbb-2021.10.0/lib/intel64/gcc4.8/libtbbmalloc.so /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64 \
&& ln -s /opt/intel/oneapi-tbb-2021.10.0/lib/intel64/gcc4.8/libtbb.so.12 /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64 \
&& ln -s /opt/intel/oneapi-tbb-2021.10.0/lib/intel64/gcc4.8/libtbbmalloc.so.2 /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64 

RUN echo /opt/intel/oclcpuexp_2023.16.6.0.28_rel/x64 >> /etc/ld.so.conf.d/libintelopenclexp.conf \
&& ldconfig -f /etc/ld.so.conf.d/libintelopenclexp.conf

############################
#        Build DPC++       #
############################
ENV CUDA_LIB_PATH=/usr/local/cuda/lib64/stubs
RUN mkdir /opt/rocm/hsa && ln -s /opt/rocm/include/hsa /opt/rocm/hsa/include && ln -s /opt/rocm/lib /opt/rocm/hsa/lib \
    && mkdir /opt/rocm/hip && ln -s /opt/rocm/include/hip /opt/rocm/hip/include && ln -s /opt/rocm/lib /opt/rocm/hip/lib \
    && mkdir -p /opt/sycl/source/llvm/build \
    && python3 /opt/sycl/source/llvm/buildbot/configure.py \
      --cuda \
      --cmake-gen "Ninja" \
      --cmake-opt="-DCMAKE_INSTALL_PREFIX=/opt/sycl" \
      --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" \
      -t release \
      -o /opt/sycl/source/llvm/build \
    && python3 /opt/sycl/source/llvm/buildbot/compile.py -o /opt/sycl/source/llvm/build -j 4 \
    && rm -rf /opt/sycl/source/llvm/

RUN mkdir /opt/dpcpp \
&& mv /opt/sycl/* /opt/dpcpp \
&& mv /opt/dpcpp /opt/sycl

############################
#       Install LLVM       #
############################
RUN wget --progress=bar:force https://apt.llvm.org/llvm.sh \
&& chmod +x llvm.sh \
&& ./llvm.sh 17 \
&& apt install -y libclang-17-dev clang-tools-17 libomp-17-dev \
&& rm llvm.sh

############################
#       Install acpp       #
############################ 
# RUN cd /opt/sycl \
# && git clone --recurse-submodules https://github.com/AdaptiveCpp/AdaptiveCpp.git -b v24.02.0 \
# && cd AdaptiveCpp \
# && mkdir build && cd build \
# && cmake -G Ninja \
#     -DCMAKE_CXX_COMPILER=/usr/bin/clang++-17 \
#     -DCLANG_EXECUTABLE_PATH=/usr/bin/clang++-17 \
#     -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm \
#     -DWITH_CUDA_BACKEND=ON -DWITH_ROCM_BACKEND=ON \
#     -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
#     -DROCM_PATH=/opt/rocm \
#     -DCMAKE_INSTALL_PREFIX=/opt/sycl/acpp \
#     -DWITH_OPENCL_BACKEND=OFF \
#     .. \
# && ninja install \
# && cd /opt \
# && rm -rf /opt/sycl/AdaptiveCpp

#Env variables
# ENV PATH=/opt/sycl/acpp/bin:/opt/sycl/dpcpp/bin:$PATH
# ENV LD_LIBRARY_PATH=/opt/sycl/dpcpp/lib:/opt/sycl/acpp/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/sycl/dpcpp/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/sycl/dpcpp/lib:$LD_LIBRARY_PATH

############################
#        Add QSYCL         #
############################ 

RUN git clone https://github.com/Nicofero/QSYCL.git /opt/QSYCL

WORKDIR /opt/QSYCL
RUN cmake -B build \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DENABLE_GPU_BACKEND=ON \
    -DENABLE_CUNQA_BACKEND=ON \
    -DENABLE_QPU_BACKEND=ON \
    && cmake --build build

# When using the container you need to compile using clang++