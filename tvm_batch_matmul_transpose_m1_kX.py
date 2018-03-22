from __future__ import absolute_import, print_function
import sys
import tvm
import topi
import numpy as np

def simplified_batch_matmul_transpose(batch_size, features, N, K):
  # computation representation
  A = tvm.placeholder((batch_size, features, 1, K), name='A')
  B = tvm.placeholder((batch_size, features, K, N), name='B')
  k = tvm.reduce_axis((0, K), 'k')
  C = tvm.compute(
    (batch_size, 1, features, N),
    lambda yb, silent, yf, x: tvm.sum(A[yb, yf, silent, k] * B[yb, yf, k, x], axis = k),
    name='C')

  # schedule optimization
  s = tvm.create_schedule(C.op)

  # memory hierarchy
  CS = s.cache_write(C, 'local')

  # schedule paramters
  num_thread_y = 8
  num_thread_x = 32
  vthread_y = 1
  vthread_x = 1

  # thread indices
  block_y = tvm.thread_axis("blockIdx.y")
  block_x = tvm.thread_axis("blockIdx.x")
  thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
  thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
  thread_yz = tvm.thread_axis((0, vthread_y), "vthread", name="vy")
  thread_xz = tvm.thread_axis((0, vthread_x), "vthread", name="vx")

  # block partitioning
  BB, MM, FF, PP = s[C].op.axis
  BBMMFF = s[C].fuse(BB, MM, FF)
  by, ty_block = s[C].split(BBMMFF, factor=num_thread_y * vthread_y)
  bx, tx_block = s[C].split(PP, factor=num_thread_x * vthread_x)
  s[C].bind(by, block_y)
  s[C].bind(bx, block_x)
  vty, ty = s[C].split(ty_block, nparts=vthread_y)
  vtx, tx = s[C].split(tx_block, nparts=vthread_x)
  s[C].reorder(by, bx, vty, vtx, ty, tx)
  s[C].reorder(by, bx, ty, tx)
  s[C].bind(ty, thread_y)
  s[C].bind(tx, thread_x)
  s[C].bind(vty, thread_yz)
  s[C].bind(vtx, thread_xz)

  # schedule CS writes
  s[CS].compute_at(s[C], tx)

  # dump something readable ...
  print("-----the generated IR-----")
  print(tvm.lower(s, [A, B, C], simple_mode=True))

  # build the model
  kernel_name = 'batch_matmul_transpose_%d_%d_%d_%d_%d_0213' %\
                (batch_size, features, 1, N, K)
  matmul_func = tvm.build(s, [A, B, C], 'cuda',\
              target_host='llvm', name=kernel_name)

  # verification
  ctx = tvm.gpu(0)
  a_np = np.random.rand(batch_size, features, 1, K).astype(np.float32)
  b_np = np.random.rand(batch_size, features, K, N).astype(np.float32)
  c_np = np.zeros((batch_size, features, 1, N), dtype=np.float32)
  for bs in range(batch_size):
    for fs in range(features):
      c_np[bs, fs, :, :] = np.dot(a_np[bs, fs, :, :], b_np[bs, fs, :, :])
  c_np = np.transpose(c_np, (0, 2, 1, 3))

  a = tvm.nd.array(a_np, ctx)
  b = tvm.nd.array(b_np, ctx)
  c = tvm.nd.array(c_np, ctx)
  evaluator = matmul_func.time_evaluator(\
     matmul_func.entry_name, ctx, number=1000)
  print('generated kernel time: %fus' % (evaluator(a, b, c).mean * 1e6))
  matmul_func(a, b, c)
  np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
  dev_module = matmul_func.imported_modules[0]
  print("-----the generated CUDA code-----")
  print(dev_module.get_source())

# test
batch_size = 64
features = 8
N = 128

if __name__ == '__main__':
  # accepted one input as parameter K for code generation
  assert len(sys.argv) == 2
  K = int(sys.argv[1])
  simplified_batch_matmul_transpose(batch_size, features, N, K)
