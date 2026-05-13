import numpy as np
import torch

def Small_Domain_Struct_Contact_pro_batch(contact_matrix, index, scale):
    # 输入：contact_matrix:[batch,n,n]--torch, index=n, scale:int [1,2,3...]
    batchsize = contact_matrix.shape[0]
    new_matrix = torch.zeros((batchsize, index + 2 * scale, index + 2 * scale), dtype=torch.float32).to(contact_matrix.device)
    SICP = []
    chr_total = sum_matrix_batch(contact_matrix)
    
    # 填充新矩阵
    new_matrix[:, scale:scale + index, scale:scale + index] = contact_matrix[:]
    # 计算每个 bin 的值
    for i in range(index):
        bin_idx = i + scale
        sub_matrix = new_matrix[:, bin_idx - scale:bin_idx + scale + 1, bin_idx - scale:bin_idx + scale + 1]
        a = sum_matrix_batch(sub_matrix)
        # 修改后的 if-else 逻辑，逐个处理每个批次的结果
        # 防止 chr_total 为 0 的情况
        bin_sicp = torch.zeros_like(chr_total)  # 初始化为 0
        nonzero_mask = chr_total != 0          # 找出 chr_total 非零的位置
        bin_sicp[nonzero_mask] = a[nonzero_mask] / chr_total[nonzero_mask]  # 仅对非零位置进行除法计算
        # bin_sicp = a / chr_total
        SICP.append(bin_sicp)
    SICP = torch.stack(SICP,dim=1)
    
    return SICP

def sum_matrix_batch(matrix):
    # 输入：[batch,n,n]--torch
    upper_triangular_sum = torch.triu(matrix).sum(dim=(1,2))
    return upper_triangular_sum


def Small_Domain_Struct_Contact_pro_batch_numpy(contact_matrix, index, scale):
    """
    用 numpy 实现 Small_Domain_Struct_Contact_pro_batch 函数，避免除零。
    输入：
        - contact_matrix: [batch, n, n] 的 numpy 数组
        - index: 矩阵的大小 (n)
        - scale: 整数范围 [1, 2, 3, ...]
    输出：
        - SICP: [batch, n] 的 numpy 数组
    """
    batchsize = contact_matrix.shape[0]
    new_matrix = np.zeros((batchsize, index + 2 * scale, index + 2 * scale), dtype=np.float32)
    SICP = []
    chr_total = sum_matrix_numpy(contact_matrix)

    # 填充新矩阵
    new_matrix[:, scale:scale + index, scale:scale + index] = contact_matrix[:]

    # 计算每个 bin 的值
    for i in range(index):
        bin_idx = i + scale
        sub_matrix = new_matrix[:, bin_idx - scale:bin_idx + scale + 1, bin_idx - scale:bin_idx + scale + 1]
        a = sum_matrix_numpy(sub_matrix)

        # 避免除以零的逻辑：如果 chr_total 为零，则结果为零
        bin_sicp = np.zeros_like(chr_total)  # 初始化为零
        nonzero_mask = chr_total != 0       # 找出 chr_total 非零的位置
        bin_sicp[nonzero_mask] = a[nonzero_mask] / chr_total[nonzero_mask]

        SICP.append(bin_sicp)

    # 将结果堆叠为 [batch, n]
    SICP = np.stack(SICP, axis=1)
    return SICP


def sum_matrix_numpy(matrix):
    """
    用 numpy 计算矩阵的上三角部分和对角线元素之和。
    输入：
        - matrix: [batch, n, n] 或 [n, n] 的 numpy 数组
    输出：
        - 每个矩阵的和，返回值的形状为 [batch] 或 [1]
    """
    # 获取上三角部分（包含对角线）
    upper_triangular = np.triu(matrix)
    return np.sum(upper_triangular, axis=(1, 2))  # 对最后两个维度求和


def Small_Domain_Struct_Contact_pro(contact_matrix, index, scale):
    contact_matrix = np.array(contact_matrix)
    new_matrix = np.zeros((index+2*scale, index+2*scale))
    SICP = []
    chr_total = sum_matrix(contact_matrix)
    for i in range(index):
        for j in range(index):
            new_matrix[i+scale, j+scale] = contact_matrix[i, j]

    for i in range(index):
        bin = i+scale
        a = sum_matrix(new_matrix[bin-scale:bin+scale+1, bin-scale:bin+scale+1])
        if a == 0:
            SICP.append(float(0))
        else:
            SICP.append(float(a/chr_total))
    return SICP


def sum_matrix(matrix):
    U = np.triu(matrix, 1)
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D)) + 1e-10

if __name__ == '__main__':
    device = torch.device("cuda:0")
    batch_size = 5000
    test_data = np.random.random((batch_size,250,250))
    import time
    t1 = time.time()
    for _ in range(10):
        # for data in test_data:
        #     Small_Domain_Struct_Contact_pro(data, 250, 1)
        Small_Domain_Struct_Contact_pro_batch(torch.from_numpy(test_data).to(device),250,1)
        # Small_Domain_Struct_Contact_pro_batch_numpy(test_data,250,1)
    print(f'use time: {round(time.time()-t1, 2)} seconds')

    # import GPUtil
    
    # gpus = GPUtil.getGPUs()
    # for gpu in gpus:
    #     if gpu.id == 0:
    #         print(f"GPU {gpu.id} ({gpu.name}):")
    #         print(f"  Total Memory: {gpu.memoryTotal:.2f} MB")
    #         print(f"  Used Memory: {gpu.memoryUsed:.2f} MB")
    #         print(f"  Free Memory: {gpu.memoryFree:.2f} MB")
    


    



