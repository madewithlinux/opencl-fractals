#!/usr/bin/env python3

import pyopencl as cl
import numpy as np

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

a_np = np.random.rand(5, 5).astype(np.float32)
b_np = np.random.rand(5, 5).astype(np.float32)
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)


x = 5120
y = 5120

outbuf_np = np.empty(int(x * y * 3)).astype(np.uint8)
outbuf_g = cl.Buffer(ctx, mf.WRITE_ONLY, outbuf_np.nbytes)

with open("fractal.cl", 'r') as f:
	if x < 100 and y < 100:
		prog = cl.Program(ctx, "#define DEBUG 1\n"+f.read()).build()
	else:
		prog = cl.Program(ctx, f.read()).build()

print(outbuf_g)

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prog.sum(queue, (x, y), None,
	np.float64(-2), np.float64(-2), np.float64(4/x), np.float64(4/y),
	outbuf_g)
# queue.finish()
cl.enqueue_copy(queue, outbuf_np, outbuf_g)

print(outbuf_np)

with open("test.ppm", 'wb') as f:
	f.write(("P6\n%d %d\n255\n" % (x,y)).encode('ascii'))
	outbuf_np.tofile(f)
