#!/usr/bin/env python3

import pyopencl as cl
import numpy as np
from collections import namedtuple
from time import time
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as rawGlBufferData

mf = cl.mem_flags

Render = namedtuple('Render',
    [ 'x'        #
    , 'y'        #
    , 'iter'     #
    , 'type'     # julia or mandelbrot
    , 'center_r' # 
    , 'center_i' # 
    , 'zoom'     #
    , 'output'
    ])
# set defaults
Render.__new__.__defaults__ = Render(400, 400, 512, "mandelbrot", 0, 0, 1, "output.ppm")

def calc_width(x, y, zoom):
    dz = 4 / zoom;
    if x > y:
        return (1 * x / y * dz, dz)
    else:
        return (dz, 1 * x / y * dz)


def run_render(render, debug=False):
    debug=True
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    x = render.x
    y = render.y

    outbuf_np = np.empty(int(x * y * 4)).astype(np.uint8)
    # outbuf_g = cl.Buffer(ctx, mf.WRITE_ONLY, outbuf_np.nbytes)
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    outbuf_g = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(x, y))

    (xwidth, ywidth) = calc_width(render.x, render.y, render.zoom);
    xscale = xwidth / render.x
    yscale = ywidth / render.y
    xmin   = render.center_r - xwidth / 2
    ymin   = render.center_i - ywidth / 2

    if debug:
        print("xwidth: " + str(xwidth))
        print("ywidth: " + str(ywidth))
        print("xscale: " + str(xscale))
        print("yscale: " + str(yscale))
        print("xmin:   " + str(xmin))
        print("ymin:   " + str(ymin))
        print("(x,y):   " + str((x,y)))
        print("outbuf_g.size:    " + str(outbuf_g.size))
        print("outbuf_g.int_ptr: " + str(outbuf_g.int_ptr))

    with open("fractal.cl", 'r') as f:
        prog_src = f.read()
    # do not let the GPU do too much debug logging because that lags the system
    if debug and x < 500 and y < 500:
        prog_src = "#define DEBUG 1\n" + prog_src
    prog = cl.Program(ctx, prog_src).build()

    t = time()
    prog.sum(queue, (x, y), None,
        outbuf_g,
        np.float64(xmin), np.float64(ymin), np.float64(xscale), np.float64(yscale)
        )
    queue.finish()
    t = time() - t
    print('time:', t, 's')
    print(1/t, 'Hz')
    cl.enqueue_copy(queue, outbuf_np, outbuf_g, origin=(0,0), region=(x, y))
    queue.finish()
    img = Image.fromarray(outbuf_np.reshape((y,x,4)), 'RGBA')
    img.save(render.output)
    print('wrote', render.output)


if __name__ == '__main__':
    # run_render(Render())
    run_render(Render(x=1920,y=1080,center_r=-0.743643135, center_i=0.131825963, zoom=210350))
    # run_render(Render(x=2560*2,y=1440*2,center_r=-0.743643135, center_i=0.131825963, zoom=210350))
    # run_render(Render(x=2000,y=2000,center_r=-0.743643135, center_i=0.131825963, zoom=210350))
