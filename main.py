#!/usr/bin/env python3

import pyopencl as cl
import numpy as np
from collections import namedtuple

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
Render.__new__.__defaults__ = Render(1280, 1280, 512, "mandelbrot", 0, 0, 1, "output.ppm")

def calc_width(x, y, zoom):
    dz = 4 / zoom;
    if x > y:
        return (1 * x / y * dz, dz)
    else:
        return (dz, 1 * x / y * dz)


def run_render(render, debug=False):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    x = render.x
    y = render.y

    outbuf_np = np.empty(int(x * y * 3)).astype(np.uint8)
    outbuf_g = cl.Buffer(ctx, mf.WRITE_ONLY, outbuf_np.nbytes)

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

    with open("fractal.cl", 'r') as f:
        prog_src = f.read()
    # do not let the GPU do too much debug logging because that lags the system
    if debug and x < 500 and y < 500:
        prog_src = "#define DEBUG 1\n" + prog_src
    prog = cl.Program(ctx,prog_src).build()

    prog.sum(queue, (x, y), None,
        np.float64(xmin), np.float64(ymin), np.float64(xscale), np.float64(yscale),
        outbuf_g)
    cl.enqueue_copy(queue, outbuf_np, outbuf_g)
    queue.finish()
    with open(render.output, 'wb') as f:
        f.write(("P6\n%d %d\n255\n" % (x,y)).encode('ascii'))
        outbuf_np.tofile(f)
    print("wrote " + render.output)


if __name__ == '__main__':
    # run_render(Render())
    run_render(Render(x=2560*2,y=1440*2,center_r=-0.743643135, center_i=0.131825963, zoom=210350))