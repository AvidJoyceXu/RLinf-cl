import numpy as np

def calculate_fwt(matrix):
    """计算 Forward Transfer (FWT)"""
    N = matrix.shape[0]
    upper_triangle_sum = 0
    count = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            upper_triangle_sum += matrix[i, j]
            count += 1
    
    fwt = upper_triangle_sum / (N * (N - 1) / 2)
    return fwt


def calculate_nbt(matrix):
    """计算 Negative Backward Transfer (NBT)"""
    N = matrix.shape[0]
    total_sum = 0
    count = 0
    
    for i in range(1, N):
        for j in range(i):
            total_sum += (matrix[i, j] - matrix[j, j])
            count += 1
    
    nbt = total_sum / (N * (N - 1) / 2)
    return nbt

def calculate_auc(matrix):
    # 3. AUC (下三角平均值：全生命周期平均表现)
    N = matrix.shape[0]
    lower_triangle_sum = 0
    count = 0
    for i in range(N):
        for j in range(i):
            lower_triangle_sum += matrix[i, j]
            count += 1
    auc = lower_triangle_sum / count
    return auc

# 输入数据
# matrix = np.array([ # NOTE: libero spatial w/ merge
#     [0.90625, 0.59375, 0.65625, 0.53125, 0.75000],
#     [0.96875, 0.93750, 0.53125, 0.96875, 0.90625],
#     [0.96875, 0.93750, 0.90625, 0.96875, 0.90625],
#     [0.96875, 0.93750, 0.90625, 0.96875, 0.90625],
#     [0.96875, 0.93750, 0.90625, 0.96875, 0.90625]
# ])


# matrix = np.array([ # NOTE: libero object w/ merge
#     [0.84375, 0.93750, 0.00000, 0.31250, 0.84375],
#     [0.84375, 0.93750, 0.00000, 0.31250, 0.84375],
#     [0.84375, 0.93750, 0.96875, 0.75000, 0.84375],
#     [0.84375, 0.8750,  0.96875, 0.75,    0.4375],
#     [0.84375, 0.8750,  0.96875, 0.75,    0.90625]
# ])

# matrix = np.array([ # NOTE: libero object w/o merge
#     [0.84375, 0.93750, 0.00000, 0.31250, 0.84375],
#     [0.84375, 0.93750, 0.00000, 0.31250, 0.84375],
#     [0.84375, 0.93750, 0.96875, 0.75000, 0.84375],
#     [0.84375, 0.93750, 0.96875, 0.90625, 0.84375],
#     [0.84375, 0.93750, 0.96875, 0.93750, 0.90625]
# ])

# matrix = np.array([ # NOTE: libero spatial w/o merge
#     [0.90625, 0.59375, 0.65625, 0.53125, 0.75000],
#     [0.90625, 0.62500, 0.65625, 0.53125, 0.81250],
#     [0.90625, 0.62500, 0.65625, 0.53125, 0.81250],
#     [0.90625, 0.62500, 0.65625, 0.96875, 0.96875],
#     [0.90625, 0.96875, 0.65625, 0.96875, 0.96875]
# ])

# matrix = np.array([ # NOTE: Seq. FT, libero-spatial
#     [0.96875, 0, 0,     0, 0],  # policy 0
#     [0,       1, 0,     0, 0],  # policy 2 (ing)
#     [0,       0, 0.375, 0, 0],  # policy 3 (ing)
#     [0,       0, 0,     0, 0],  # policy 6
#     [0,       0, 0,     0, 0]   # policy 7
# ])

# matrix = np.array([
#     [1, 0,       0, 0,      0],  # policy 1 (done)
#     [0, 0.84375, 0, 0.8125, 0],  # policy 6 (ing) succ=53
#     [0, 0,       0, 0,      0],  # policy 7 (ing)
#     [0, 0,       0, 0,      0],  # policy 8
#     [0, 0,       0, 0,      0]   # policy 9
# ])

# matrix = np.array([ # NOTE: Seq. FT, Libero-obj, seed=0 & seed=23
#     [1, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

# matrix = np.array([ # NOTE: Seq. FT, Libero-spatial, seed=0
#     [1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 0.15625, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

# matrix = np.array([ # NOTE: Seq. FT, Libero-spatial,  seed=23
#     [1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 0.25, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

# matrix = np.array([
#     [0.875, 0.9375, 0,       0.40625, 0.90625],
#     [0.875, 0.9375, 0,       0.40625, 0.90625],
#     [0.875, 0.9375, 0.90625, 0.53125, 0.90625],
#     [0.75,  0.8125, 0.90625, 0.96875, 0.40625],
#     [0.75,  0.8125, 0.90625, 0.96875, 0.96875]
# ])

matrix = np.array([
    [0.875, 0.9375, 0,       0.40625, 0.90625],
    [0.875, 0.9375, 0,       0.40625, 0.90625],
    [0.875, 0.9375, 0.90625, 0.53125, 0.90625],
    [0.875, 0.9375, 0.90625, 0.9375,  0.90625],
    [0.875, 0.9375, 0.90625, 0.9375,  0.96875]
])

print("矩阵:")
print(matrix)
print()

# 计算 FWT 和 NBT
fwt = calculate_fwt(matrix)
nbt = calculate_nbt(matrix)
auc = calculate_auc(matrix)

print("="*70)
print(f"Forward Transfer (FWT): {fwt:.6f}")
print(f"Negative Backward Transfer (NBT): {nbt:.6f}")
print(f"AUC: {auc:.6f}")
print("="*70)