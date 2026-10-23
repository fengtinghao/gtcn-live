/* Real-time streaming kernel for the gTCN renderer.
 *
 * All causal convs are done as one GEMM over stacked dilated taps; the
 * gated activation uses tanh/sigmoid via a single vectorised tanh:
 *   sigmoid(x) = 0.5 + 0.5 * tanh(x / 2)   (the x/2 is folded into the weights
 * by the Python side). No allocation happens in gtcn_process().
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
static void vtanh(float *x, int n) { vvtanhf(x, x, &n); }
#else
#include <cblas.h>
static void vtanh(float *x, int n) { for (int i = 0; i < n; i++) x[i] = tanhf(x[i]); }
#endif

typedef struct {
    int cin, cout, k, d, pad;
    const float *W;   /* cout x (k*cin), tap-major */
    const float *b;   /* cout */
    float *state;     /* cin x pad */
} Conv;

typedef struct {
    int rc, sc, L, maxn, ldim;
    Conv pe1, pe2;
    Conv *blocks;
    const float **Wr, **br;      /* (rc+sc) x rc, rc+sc */
    const float *o1W, *o1b, *o2W, *o2b;
    float *full, *taps, *g, *h, *skip, *rs, *tmp;
} Engine;

static void conv_init(Conv *c, int cin, int cout, int k, int d, const float *W, const float *b) {
    c->cin = cin; c->cout = cout; c->k = k; c->d = d; c->pad = (k - 1) * d;
    c->W = W; c->b = b;
    c->state = (float *)calloc((size_t)cin * (c->pad > 0 ? c->pad : 1), sizeof(float));
}

static void bias_fill(float *out, const float *b, int rows, int n) {
    for (int r = 0; r < rows; r++) {
        float v = b[r];
        float *o = out + (size_t)r * n;
        for (int i = 0; i < n; i++) o[i] = v;
    }
}

static void gemm(int m, int n, int k, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 1.0f, C, n);
}

/* out (cout x n) = conv(x (cin x n)); `full`/`taps` are scratch. */
static void conv_run(Engine *e, Conv *c, const float *x, int n, float *out) {
    int pad = c->pad, cin = c->cin, w = pad + n;
    for (int ch = 0; ch < cin; ch++) {
        float *f = e->full + (size_t)ch * w;
        if (pad) memcpy(f, c->state + (size_t)ch * pad, pad * sizeof(float));
        memcpy(f + pad, x + (size_t)ch * n, n * sizeof(float));
        if (pad) memcpy(c->state + (size_t)ch * pad, f + n, pad * sizeof(float));
    }
    for (int j = 0; j < c->k; j++)
        for (int ch = 0; ch < cin; ch++)
            memcpy(e->taps + ((size_t)j * cin + ch) * n,
                   e->full + (size_t)ch * w + (size_t)j * c->d, n * sizeof(float));
    bias_fill(out, c->b, c->cout, n);
    gemm(c->cout, n, c->k * cin, c->W, e->taps, out);
}

void *gtcn_create(int rc, int sc, int L, int maxn,
                  int pe_k, const float *pe1W, const float *pe1b, const float *pe2W, const float *pe2b,
                  int blk_k, const int *dil, const float **gW, const float **gb,
                  const float **Wr, const float **br,
                  const float *o1W, const float *o1b, const float *o2W, const float *o2b) {
    Engine *e = (Engine *)calloc(1, sizeof(Engine));
    e->rc = rc; e->sc = sc; e->L = L; e->maxn = maxn;
    conv_init(&e->pe1, 1, rc, pe_k, 1, pe1W, pe1b);
    conv_init(&e->pe2, rc, rc, pe_k, 1, pe2W, pe2b);
    e->blocks = (Conv *)calloc(L, sizeof(Conv));
    int max_pad = e->pe2.pad, max_tapsrows = pe_k * rc;
    for (int i = 0; i < L; i++) {
        conv_init(&e->blocks[i], rc, 2 * rc, blk_k, dil[i], gW[i], gb[i]);
        if (e->blocks[i].pad > max_pad) max_pad = e->blocks[i].pad;
        if (blk_k * rc > max_tapsrows) max_tapsrows = blk_k * rc;
    }
    e->Wr = Wr; e->br = br;
    e->o1W = o1W; e->o1b = o1b; e->o2W = o2W; e->o2b = o2b;
    size_t m = (size_t)maxn;
    e->full = (float *)malloc(sizeof(float) * rc * (max_pad + m));
    e->taps = (float *)malloc(sizeof(float) * max_tapsrows * m);
    e->g = (float *)malloc(sizeof(float) * 2 * rc * m);
    e->h = (float *)malloc(sizeof(float) * rc * m);
    e->skip = (float *)malloc(sizeof(float) * sc * m);
    e->rs = (float *)malloc(sizeof(float) * (rc + sc) * m);
    e->tmp = (float *)malloc(sizeof(float) * (rc > sc ? rc : sc) * m);
    return e;
}

void gtcn_reset(void *p) {
    Engine *e = (Engine *)p;
    Conv *cs[2] = {&e->pe1, &e->pe2};
    for (int i = 0; i < 2; i++) memset(cs[i]->state, 0, sizeof(float) * cs[i]->cin * cs[i]->pad);
    for (int i = 0; i < e->L; i++)
        memset(e->blocks[i].state, 0, sizeof(float) * e->blocks[i].cin * e->blocks[i].pad);
}

void gtcn_process(void *p, const float *x, float *y, int n) {
    Engine *e = (Engine *)p;
    int rc = e->rc, sc = e->sc;
    size_t cnt = (size_t)rc * n;

    conv_run(e, &e->pe1, x, n, e->tmp);
    for (size_t i = 0; i < cnt; i++) e->tmp[i] = e->tmp[i] > 0 ? e->tmp[i] : 0;
    conv_run(e, &e->pe2, e->tmp, n, e->h);
    for (size_t i = 0; i < cnt; i++) e->h[i] = e->h[i] > 0 ? e->h[i] : 0;

    memset(e->skip, 0, sizeof(float) * sc * n);
    for (int l = 0; l < e->L; l++) {
        conv_run(e, &e->blocks[l], e->h, n, e->g);
        vtanh(e->g, 2 * rc * n);
        float *tf = e->g, *tg = e->g + cnt;
        for (size_t i = 0; i < cnt; i++) tf[i] = tf[i] * (0.5f + 0.5f * tg[i]);
        bias_fill(e->rs, e->br[l], rc + sc, n);
        gemm(rc + sc, n, rc, e->Wr[l], tf, e->rs);
        const float *res = e->rs, *sk = e->rs + cnt;
        for (size_t i = 0; i < cnt; i++) e->h[i] = (e->h[i] + res[i]) * 0.70710678118654752f;
        size_t scnt = (size_t)sc * n;
        for (size_t i = 0; i < scnt; i++) e->skip[i] += sk[i];
    }
    size_t scnt = (size_t)sc * n;
    for (size_t i = 0; i < scnt; i++) e->skip[i] = e->skip[i] > 0 ? e->skip[i] : 0;
    bias_fill(e->tmp, e->o1b, sc, n);
    gemm(sc, n, sc, e->o1W, e->skip, e->tmp);
    for (size_t i = 0; i < scnt; i++) e->tmp[i] = e->tmp[i] > 0 ? e->tmp[i] : 0;
    bias_fill(y, e->o2b, 1, n);
    gemm(1, n, sc, e->o2W, e->tmp, y);
    vtanh(y, n);
}

void gtcn_destroy(void *p) {
    Engine *e = (Engine *)p;
    free(e->pe1.state); free(e->pe2.state);
    for (int i = 0; i < e->L; i++) free(e->blocks[i].state);
    free(e->blocks); free(e->full); free(e->taps); free(e->g); free(e->h);
    free(e->skip); free(e->rs); free(e->tmp); free(e);
}
