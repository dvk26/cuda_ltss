## Cách chạy

**Đôi với CPU**
1. Biên dịch chương trình: 
```
!g++ -std=c++17 -O3 -march=native -ffast-math -DNDEBUG src/cpu/*.cpp -o autoencoder_cpu
```

2. Chạy chương trình: 
```
!./autoencoder_cpu cifar-10-batches-bin [--keep-partial]
```
Trong đó, `--keep-partial` để sử dụng chỉ 4% dữ liệu (2000 mẫu train, 400 mẫu test).
Nếu không có flag này thì sử dụng toàn bộ dữ liệu.

3. Dữ liệu sẽ được đưa vào thư mục `out-cpu/`.

**Đối với GPU**
1. Biên dịch chương trình: 
```
# GPU version
!nvcc -std=c++17 -O2 \
    src/gpu/main.cu \
    src/gpu/gpu_autoencoder.cu \
    -I./src/gpu \
    -arch=sm_75 \
    -o autoencoder_gpu
```

2. Chạy chương trình: 
```
!./autoencoder_gpu cifar-10-batches-bin [--keep-partial]
```
`--keep-partial` có tác dụng tương tự như bản CPU.

3. Dữ liệu sẽ được đưa vào thư mục `out-gpu/`.