import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.workload import (
    synthetic_workload, uniform_workload, _parse_trace_lines,
    WORKLOAD_PROFILES,
)


def test_synthetic_correct_sizes():
    ops = list(synthetic_workload("RTDATA", num_keys=10, num_ops=20, seed=1))
    profile = WORKLOAD_PROFILES["RTDATA"]

    for op in ops:
        assert op.key_size == profile["key_size"]
        if op.op_type == "put":
            assert op.value_size == profile["value_size"]
        else:
            assert op.value_size == 0


def test_synthetic_population_phase():
    # first num_keys ops should all be puts (population phase)
    ops = list(synthetic_workload("ETC", num_keys=20, num_ops=50, seed=1))
    for op in ops[:20]:
        assert op.op_type == "put"


def test_synthetic_delete_ratio():
    ops = list(synthetic_workload(
        "RTDATA", num_keys=10, num_ops=1000,
        read_ratio=0.3, delete_ratio=0.3, seed=1,
    ))
    # skip population phase
    mixed = ops[10:]
    deletes = sum(1 for op in mixed if op.op_type == "delete")
    reads = sum(1 for op in mixed if op.op_type == "get")
    # ratios should be roughly correct (within 5%)
    assert abs(deletes / len(mixed) - 0.3) < 0.05
    assert abs(reads / len(mixed) - 0.3) < 0.05


def test_synthetic_all_profiles():
    for name in WORKLOAD_PROFILES:
        ops = list(synthetic_workload(name, num_keys=5, num_ops=10, seed=1))
        assert len(ops) == 15  # 5 population + 10 mixed


def test_uniform_workload_sizes():
    ops = list(uniform_workload(num_keys=5, num_ops=20,
                                key_size=16, value_size=128, seed=1))
    for op in ops:
        assert op.key_size == 16
        if op.op_type == "put":
            assert op.value_size == 128


def test_trace_parser_valid_lines():
    lines = [
        "1000,key1,32,128,0,get,0\n",
        "1001,key2,24,64,0,set,0\n",
        "1002,key3,16,0,0,delete,0\n",
    ]
    ops = list(_parse_trace_lines(lines))
    assert len(ops) == 3
    assert ops[0].op_type == "get"
    assert ops[0].key == b"key1"
    assert ops[1].op_type == "put"
    assert ops[2].op_type == "delete"


def test_trace_parser_skips_malformed():
    lines = [
        "short,line\n",        # too few fields
        "",                     # empty
        "1,key1,32,128,0,get,0\n",  # valid
        "1,key2,32,128,0,cas,0\n",  # unsupported op (cas)
    ]
    ops = list(_parse_trace_lines(lines))
    assert len(ops) == 1
    assert ops[0].op_type == "get"


def test_trace_parser_max_ops():
    lines = [f"1,key{i},32,64,0,set,0\n" for i in range(100)]
    ops = list(_parse_trace_lines(lines, max_ops=10))
    assert len(ops) == 10


if __name__ == "__main__":
    test_synthetic_correct_sizes()
    test_synthetic_population_phase()
    test_synthetic_delete_ratio()
    test_synthetic_all_profiles()
    test_uniform_workload_sizes()
    test_trace_parser_valid_lines()
    test_trace_parser_skips_malformed()
    test_trace_parser_max_ops()
    print("all workload tests passed")
