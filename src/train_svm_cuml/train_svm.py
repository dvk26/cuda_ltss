import numpy as np
import struct

def load_features_bin(filename):
    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        # Đọc Header: 2 số integer (num_samples, dim)
        header = f.read(8)
        num_samples, dim = struct.unpack('ii', header)
        
        print(f"  Samples: {num_samples}, Dim: {dim}")
        
        # Tính kích thước 1 mẫu: 4 bytes (label) + dim * 4 bytes (features)
        sample_size = 4 + dim * 4
        
        # Đọc toàn bộ dữ liệu còn lại vào numpy array byte
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Reshape dữ liệu thô để xử lý
    # Cấu trúc bộ nhớ: [Label (4B) | Feature (Dim * 4B)] lặp lại
    dt = np.dtype([('label', 'i4'), ('features', 'f4', (dim,))])
    data = np.frombuffer(raw_data, dtype=dt)
    
    # Tách ra X và y
    y = data['label']
    X = data['features']
    
    return X, y

# --- SỬ DỤNG ---
X_train, y_train = load_features_bin('train_features.bin')
X_test, y_test = load_features_bin('test_features.bin')

# --- CHẠY CUML SVM ---
from cuml.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Scale dữ liệu (Rất quan trọng cho SVM)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM trên GPU
print("Training cuML SVM...")
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train_scaled, y_train)

# Đánh giá
acc = model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")