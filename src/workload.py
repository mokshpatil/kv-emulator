import random
from dataclasses import dataclass


@dataclass
class Operation:
    op_type: str        # "get", "put", "delete"
    key: bytes
    key_size: int
    value_size: int     # 0 for get/delete


# workload profiles from KVPack Table V
WORKLOAD_PROFILES = {
    "ETC":     {"source": "Facebook",    "key_size": 41, "value_size": 358},
    "UDB":     {"source": "Facebook",    "key_size": 27, "value_size": 127},
    "ZippyDB": {"source": "Facebook",    "key_size": 48, "value_size": 43},
    "Cache":   {"source": "Twitter",     "key_size": 42, "value_size": 188},
    "Cache15": {"source": "Twitter",     "key_size": 38, "value_size": 38},
    "VAR":     {"source": "Facebook",    "key_size": 35, "value_size": 115},
    "Crypto1": {"source": "BlockStream", "key_size": 76, "value_size": 50},
    "Crypto2": {"source": "Trezor",      "key_size": 37, "value_size": 110},
    "Dedup":   {"source": "IBM",         "key_size": 20, "value_size": 44},
    "RTDATA":  {"source": "Microsoft",   "key_size": 24, "value_size": 10},
}


def synthetic_workload(
    workload_name: str,
    num_keys: int,
    num_ops: int,
    read_ratio: float = 0.5,
    delete_ratio: float = 0.0,
    seed: int = 42,
):
    """Generate a synthetic workload matching a KVPack profile."""
    profile = WORKLOAD_PROFILES[workload_name]
    key_size = profile["key_size"]
    value_size = profile["value_size"]
    rng = random.Random(seed)

    # pre-generate key pool
    keys = [rng.randbytes(key_size) for _ in range(num_keys)]

    # phase 1: populate with PUTs
    for key in keys:
        yield Operation("put", key, key_size, value_size)

    # phase 2: mixed read/write/delete
    write_ratio = 1.0 - read_ratio - delete_ratio
    for _ in range(num_ops):
        key = rng.choice(keys)
        r = rng.random()
        if r < read_ratio:
            yield Operation("get", key, key_size, 0)
        elif r < read_ratio + write_ratio:
            yield Operation("put", key, key_size, value_size)
        else:
            yield Operation("delete", key, key_size, 0)


def uniform_workload(
    num_keys: int,
    num_ops: int,
    key_size: int = 32,
    value_size: int = 64,
    read_ratio: float = 0.5,
    seed: int = 42,
):
    """Generate a uniform workload with custom key/value sizes."""
    rng = random.Random(seed)
    keys = [rng.randbytes(key_size) for _ in range(num_keys)]

    for key in keys:
        yield Operation("put", key, key_size, value_size)

    for _ in range(num_ops):
        key = rng.choice(keys)
        if rng.random() < read_ratio:
            yield Operation("get", key, key_size, 0)
        else:
            yield Operation("put", key, key_size, value_size)


def trace_workload(trace_path: str, max_ops: int = 0):
    """Replay a Twitter cache trace file (CSV, optionally zstd-compressed)."""
    if trace_path.endswith(".zst"):
        import io
        import zstandard
        with open(trace_path, "rb") as f:
            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(f)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            yield from _parse_trace_lines(text, max_ops)
    else:
        with open(trace_path, "r") as f:
            yield from _parse_trace_lines(f, max_ops)


def _parse_trace_lines(source, max_ops=0):
    """Parse Twitter trace CSV lines into Operations."""
    count = 0
    for line in source:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue

        key = parts[1].strip().encode()
        key_size = int(parts[2].strip())
        value_size = int(parts[3].strip())
        op = parts[5].strip().lower()

        if op in ("get", "gets"):
            yield Operation("get", key, key_size, 0)
            count += 1
        elif op in ("set", "add", "replace"):
            yield Operation("put", key, key_size, value_size)
            count += 1
        elif op == "delete":
            yield Operation("delete", key, key_size, 0)
            count += 1
        # skip cas, append, prepend, incr, decr

        if max_ops > 0 and count >= max_ops:
            return
