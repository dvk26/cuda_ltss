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
    src/gpu_<version>/main.cu \
    src/gpu_<version>/gpu_autoencoder.cu \
    -I./src/gpu_<version> \
    -arch=sm_75 \
    -o autoencoder_gpu_<version>
```
trong đó `<version>` là tên phiên bản, bao gồm v1.1, v1.3 và v1.4.

2. Chạy chương trình: 
```
!./autoencoder_gpu_<version> cifar-10-batches-bin [--keep-partial]
```
`--keep-partial` có tác dụng tương tự như bản CPU.

3. Dữ liệu sẽ được đưa vào thư mục `out-gpu/`.