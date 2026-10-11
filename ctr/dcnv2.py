"""DCN-V2 (Deep & Cross Network V2) in Keras 3 / TensorFlow.

Reference: Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons
for Web-scale Learning to Rank Systems" (2021).

Architecture
    cats  -> one shared embedding table (per-feature offset) -> flatten --+
    dense ----------------------------------------------------------------+-> x0
    cross tower :  x_{l+1} = x0 * (W_l x_l + b_l) + x_l      (explicit feature crosses)
    deep tower  :  MLP on x0                                  (implicit interactions)
    "parallel"  :  logit = Dense([cross_out ; deep_out])
    "stacked"   :  logit = Dense(MLP(cross_out))

`cross_rank=None` uses the full-rank W (d x d). A small integer uses the low-rank
form W = U V^T, which cuts cross-layer parameters from d^2 to 2*d*rank.
"""
import keras
from keras import layers, ops

from ctr.config import N_CAT, N_INT, NUM_BUCKETS


@keras.saving.register_keras_serializable(package="ctr")
class CrossLayer(layers.Layer):
    """One DCN-V2 cross layer. Call with [x0, x_l]; returns x_{l+1}."""

    def __init__(self, rank: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank

    def build(self, input_shape):
        d = input_shape[0][-1]
        if self.rank is None:
            self.w = layers.Dense(d, name="w")                       # W x + b
            self.w.build((None, d))
        else:
            self.v = layers.Dense(self.rank, use_bias=False, name="v")  # V^T x
            self.u = layers.Dense(d, name="u")                          # U (.) + b
            self.v.build((None, d))
            self.u.build((None, self.rank))
        super().build(input_shape)

    def call(self, inputs):
        x0, xl = inputs
        proj = self.w(xl) if self.rank is None else self.u(self.v(xl))
        return x0 * proj + xl

    def get_config(self):
        return {**super().get_config(), "rank": self.rank}


@keras.saving.register_keras_serializable(package="ctr")
class OffsetCats(layers.Layer):
    """Add a per-feature offset so all categorical features index one shared embedding table."""

    def __init__(self, n_cat: int, num_buckets: int, **kwargs):
        super().__init__(**kwargs)
        self.n_cat, self.num_buckets = n_cat, num_buckets

    def call(self, cats):
        offsets = ops.arange(self.n_cat, dtype=cats.dtype) * self.num_buckets
        return cats + offsets[None, :]

    def get_config(self):
        return {**super().get_config(), "n_cat": self.n_cat, "num_buckets": self.num_buckets}


def build_dcnv2(num_buckets: int = NUM_BUCKETS, n_dense: int = N_INT, n_cat: int = N_CAT,
                emb_dim: int = 16, n_cross: int = 3, cross_rank: int | None = None,
                mlp_units=(256, 128), dropout: float = 0.1, structure: str = "parallel",
                emb_l2: float = 0.0) -> keras.Model:
    if structure not in ("parallel", "stacked"):
        raise ValueError("structure must be 'parallel' or 'stacked'")

    dense_in = keras.Input((n_dense,), dtype="float32", name="dense")
    cats_in = keras.Input((n_cat,), dtype="int32", name="cats")

    ids = OffsetCats(n_cat, num_buckets, name="offset")(cats_in)
    emb = layers.Embedding(
        n_cat * num_buckets, emb_dim,
        embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=keras.regularizers.l2(emb_l2) if emb_l2 else None,
        name="embedding")(ids)                                       # [B, n_cat, emb_dim]
    x0 = layers.Concatenate(name="x0")([dense_in, layers.Flatten()(emb)])

    def mlp(x, prefix):
        for i, u in enumerate(mlp_units):
            x = layers.Dense(u, activation="relu", name=f"{prefix}_{i}")(x)
            if dropout:
                x = layers.Dropout(dropout)(x)
        return x

    xl = x0
    for i in range(n_cross):
        xl = CrossLayer(rank=cross_rank, name=f"cross_{i}")([x0, xl])

    if structure == "parallel":
        top = layers.Concatenate(name="merge")([xl, mlp(x0, "deep")])
    else:
        top = mlp(xl, "deep")
    logit = layers.Dense(1, name="logit")(top)
    prob = layers.Activation("sigmoid", name="ctr")(logit)
    return keras.Model({"dense": dense_in, "cats": cats_in}, prob, name=f"dcnv2_{structure}")
