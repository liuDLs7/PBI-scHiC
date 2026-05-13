'''
The code on this page defines the functions used to calculate the BCP feature set
'''
import numpy as np
import torch

def find_contact_list_batch(list):
    # [batch, bin-len]
    con = torch.sum(list,dim=1)
    return con

# This function is used to calculate the probability of bin contact for each chromosome
# The input contact_matrix is the contact matrix corresponding to chromosome number chr under the current cell, and index is the number of blocks chromosome number chr can cut out
def Bin_contact_pro_batch(contact_matrix, index):
    # 输入：contact_matrix:[batch,n,n]--torch, index=n, scale:int [1,2,3...]
    batchsize = contact_matrix.shape[0]

    Bc = []
    # chr_total is the total number of contacts in the contact matrix corresponding to chromosome chr number under the previous cell.
    # In short it is to find the total number of contacts on the current chromosome.
    chr_total = sum_matrix_batch(contact_matrix)
    for i in range(index):
        # contact_matrix[i] is the i-th row of the chromosome contact matrix
        con_list = find_contact_list_batch(contact_matrix[:,i])
        con_pro = con_list/chr_total
        # Each con_pro is the contact probability corresponding to each chromosome segment
        Bc.append(con_pro)
    BCP = torch.stack(Bc,dim=1)
    return BCP


def sum_matrix_batch(matrix):
    # 输入：[batch,n,n]--torch
    upper_triangular_sum = torch.triu(matrix).sum(dim=(1,2))
    return upper_triangular_sum + 1e-10



if __name__ == '__main__':
    a = np.random.randint(1, 10, (5, 30, 30))
    a = torch.from_numpy(a)
    print(a.shape)
    b = Bin_contact_pro_batch(a, 30)
    print(type(b))
    print(len(b))
    print(b)


