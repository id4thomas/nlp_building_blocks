# Lecture 02 - Primitives, Efficiency (Memory/Compute)
Overview
- Memory Accounting: tensors, memory
- Compute Accounting: Operations, FLOPs
- Models: Parameters, optimizer, train_loop

## Motivating Questions (napkin math)
**Q1.** Time to train **70B** model on **15T tokens** on 1024 H100s
- total flops: `6 * 70e9 * 15e12*
    - **6: typically 6 FLOP per token per param**
    - 70e9: 70B
    - 15e12: 15T tokens
- h100_flop_per_sec: (1979e12)/2
- mfu (model flop utilization): 0.5 (approximately 50%)
- -> flops_per_day = h100_fps * mfu * 1024 GPU * (60*60*24)
- -> days = total_flops / flops_per_day


**Q2.** largest model you can train on 8*H100s with AdamW (native)
- h100_bytes = 80e9 (e9: 1GB)
- bytes_per_parameter = 4 + 4 + (4+4)
    - 4: parameters (float32)
    - 4: gradients (float32)
    - 4+4: AdamW optimizer state (1st, 2nd moment estimate per param)
- num_parameters = (h100_bytes*8) / bytes_per_param

Caveat: activations are not accounted for

## Memory Accounting
### Tensors Memory
Almost everything is stored as floating point number

| precision | size (bytes) | notes |
| --- | --- | --- |
| fp32 | 4 | baseline |
| float16 (fp16) | 2 | dynamic **range (exponent)** isn't great -> can get instability |
| bfloat16 (bf16) | 2 | same dynamic range as fp32, but **resolution (mantissa) is worse** |
| fp8 | 1 | Standardiezed in 2022, H100s support 2 variants of FP8 (E4M3, E5M2) |

Why does **resolution matter less** for deep learning?
```
- fp16: exponent = 5 bits → smallest positive ≈ 6e-8, largest ≈ 6e+4.
- bf16: exponent = 8 bits (like fp32) → smallest positive ≈ 1e-38, largest ≈ 3e+38.

- bf16, at around the value 1.0, the spacing between representable numbers is ~0.0078 (coarse).
- fp16, it’s ~0.00098 (finer)
```
- resolution means 'spacing', doesn't mean 'fine' small numbers can't be represented
- ex. fp16 can underflow at 1e-8 (small gradient)

## Compute Accounting
### tensor_storage
PyTorch tensors are **pointers** into allocated memory (+metadata)
- metadata example: `x.stride(0) -> 4`: skip 4 elements to go to next row
- indexing: `row: 1, col: 2 -> r*x.stride(0) + c*x.stride(1)`

### tensor_slicing
Many operations provide a different **view** of the tensor (no copy)

Contiguous
- for non-contiguous entries -> further views aren't possible (ex. transpose)
- ex. `x.transpose(1,0).view(2,3)` -> `x.transpost(1,0).contiguous().view(2,3)`

### tensor_elementwise
Operations apply some operation to **each element** of tensor, return tensor of same shape

### tensor_matmul
Example:
```
x (4,8,16,32)
y (32,2)

for dim0 in ...
    for dim1 in ...
        x[dim0][dim1] -> shape of (16,32)
        x[dim0][dim1]@y -> shape of (16,2)
        out[dim0][dim1] -> shape of (16,2)

-> Final output (4, 8, 16, 2)
```
- matmul iterates over values of first 2 dimensions

### tensor_einops
einops: library for manipulating tensors where dimensions are named
- inspired by einstein summation notation

example (matmul):
```
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2,3,4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2,3,4)

x@y.transpose(-2,-1)

->
einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```
- dimensions that are not named in output are summed over

example (broadcasting)
```
einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

-> out[..., i, j] = sum over hidden(x[...,i, k] * y[...,j, k])
-> for each element of hidden of seq1 -> mult with seq2 -> sum
```

### tensor_operations_flops
Calculating FLOP of operations

matmul X(B,D)@W(D,K): 2 * B * D * K
- 2: one for multiplication, one for addition per (i,j,k) triple
    - multiplication: x[i][j]*w[j][k]
    - addition: summing across X row i, W col k -> j elements (i*j*k sums)

Elementwise operation (m,n): O(mn)

Addition of 2 matrices (m,n): m*n

-> In general matmul is most expensive

### gradients_flops


## Models
