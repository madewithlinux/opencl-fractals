// -*- mode: c++ -*-
#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef NUMBER_OF_ITERATIONS
#define NUMBER_OF_ITERATIONS 1024
#endif

#ifndef FRAC_EXPRESSION
#define FRAC_EXPRESSION cadd(cmul(z,z), c)
#endif

#ifndef FRACTAL_FUNC
#define FRACTAL_FUNC mandelbrot
#endif


double2 cadd(const double2 a, const double2 b) { return (double2)(a.x + b.x, a.y + b.y); }
double2 csub(const double2 a, const double2 b) { return (double2)(a.x - b.x, a.y - b.y); }
double2 cmul(const double2 a, const double2 b) { return (double2)(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y); }
double  cmag2(const double2 a) { return a.x * a.x + a.y * a.y; }
double2 cdiv(const double2 a, double2 b) {
    const double denom = cmag2(b);
    return (double2)((a.x*b.x + a.y*b.y)/denom, (a.y*b.x + a.x*b.y)/denom);
}

double fractal(const double2 c, const double2 z_) {
    double2 z = z_;
    for (int iteration = 0; iteration < NUMBER_OF_ITERATIONS; iteration++) {
        z = FRAC_EXPRESSION;
        if (cmag2(z) > NUMBER_OF_ITERATIONS) {
            double sl = iteration - log2(log2(z.x * z.x + z.y * z.y)) + 4.0;
            return sl;
        }
    }
    return -1.0;
}

double mandelbrot(const double2 coordinate,const double2 offset) {
    double2 c = (double2)(coordinate.x, coordinate.y);
    double2 z = (double2)(0.0, 0.0);

    return fractal(c, z);
}

double julia(const double2 coordinate, const double2 offset) {
    double2 c = (double2)(offset.x, offset.y);
    double2 z = (double2)(coordinate.x, coordinate.y);

    return fractal(c, z);
}

kernel void sum(
    write_only image2d_t dest,
    double r_min,
    double i_min,
    double r_step,
    double i_step
    // TODO: c, multipler, offset
) {
    private double r;
    private double i;
    private double fractalValue;
    private double3 color;
    private uchar3 color_byte;

    r = r_min + r_step * convert_double(get_global_id(0));
    i = i_min + i_step * convert_double(get_global_id(1));

    fractalValue = FRACTAL_FUNC((double2)(r, i), (double2)(0, 0));

    if (fractalValue > 0.0) {
        // normalize
        fractalValue *= 4;
        fractalValue = log2(fractalValue + 1.0)*0.5;

        fractalValue = fmod(fractalValue, 1+1.1e-16);
        if (fractalValue < 0.5) {
            fractalValue *= 2;
        } else {
            fractalValue = 2.0 - 2.0*fractalValue;
        }

        // colorize
        color = 0.5 + 0.5*cos( 3.0 + fractalValue*2*M_PI + (double3)(0.0,0.6,1.0));
        color_byte = convert_uchar3(color*256);
    } else {
        color_byte = (uchar3)(0,0,0);
    }

    int2 pos = (int2)(get_global_id(0), get_global_size(1) - get_global_id(1) - 1);
    uint4 pix = (uint4) (color_byte.x, color_byte.y, color_byte.z, 255);
    write_imageui(dest, pos, pix);

#if DEBUG
    printf("%6llu %6llu | %6llu | %9f %9f %9f %9f | %9f %9f | %9f \n",
        (size_t) get_global_id(0),
        (size_t) get_global_id(1),
        (double) r_min,
        (double) i_min,
        (double) r_step,
        (double) i_step,
        (double) r,
        (double) i,
        (double) fractalValue
        );
#endif
}
