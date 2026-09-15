# Implementation Checklist

## For Each Algorithm Implementation:

### Pre-Implementation
- [ ] Review CUDA implementation in nx-cugraph/cugraph
- [ ] Identify parallelization strategy
- [ ] Design WGSL compute shader approach
- [ ] Plan buffer layouts and bindings

### Implementation
- [ ] Create CPU reference implementation
- [ ] Implement WGSL compute shader
- [ ] Set up buffer management
- [ ] Handle edge cases

### Testing
- [ ] Unit tests with small graphs
- [ ] Correctness validation against CPU version
- [ ] Performance benchmarks
- [ ] Memory usage profiling

### Documentation
- [ ] Algorithm explanation
- [ ] API documentation
- [ ] Usage examples
- [ ] Performance characteristics

## WebGPU/WGSL Conversion Patterns

### Thread Mapping
```
CUDA                          →  WebGPU/WGSL
blockIdx.x * blockDim.x + threadIdx.x → global_invocation_id.x
__shared__ memory             →  var<workgroup> memory
atomicAdd()                   →  atomicAdd() 
__syncthreads()              →  workgroupBarrier()
```

### Memory Access
```
CUDA                          →  WebGPU/WGSL
global memory                 →  var<storage, read_write>
constant memory              →  var<uniform>
texture memory               →  texture_2d<f32>
```

### Kernel Launch
```
CUDA                          →  WebGPU/WGSL
kernel<<<blocks, threads>>>() →  computePass.dispatchWorkgroups(x, y, z)
```

## Performance Optimization Checklist

- [ ] Coalesced memory access patterns
- [ ] Minimize divergent branching
- [ ] Optimize workgroup size
- [ ] Use shared memory effectively
- [ ] Reduce global memory transactions
- [ ] Profile with browser DevTools

## Common Pitfalls to Avoid

1. **Buffer Alignment**: WebGPU requires specific alignments
2. **Workgroup Limits**: Check device limits
3. **Atomic Operations**: Limited compared to CUDA
4. **Dynamic Allocation**: Not available in WGSL
5. **Recursion**: Not supported in shaders