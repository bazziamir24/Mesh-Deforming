import enum
import tensorflow as tf

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


import tensorflow as tf

def tetrahedra_to_edges(cells):
    tf.print("ENTERED tetrahedra_to_edges")

    v0, v1, v2, v3 = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]

    edges = tf.concat([
        tf.stack([v0, v1], axis=1),
        tf.stack([v0, v2], axis=1),
        tf.stack([v0, v3], axis=1),
        tf.stack([v1, v2], axis=1),
        tf.stack([v1, v3], axis=1),
        tf.stack([v2, v3], axis=1),
    ], axis=0)

    senders = tf.minimum(edges[:, 0], edges[:, 1])
    receivers = tf.maximum(edges[:, 0], edges[:, 1])
    senders_bi = tf.concat([senders, receivers], axis=0)
    receivers_bi = tf.concat([receivers, senders], axis=0)

    tf.print("senders.shape =", tf.shape(senders_bi))
    tf.print("senders sample =", senders_bi[:10])

    return senders_bi, receivers_bi
