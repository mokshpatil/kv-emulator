import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import small_config
from src.metrics import Metrics
from src.flash import Flash


def test_read_write_counting():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)

    p1 = flash.allocate_page()
    flash.write_page(p1, "data")
    flash.read_page(p1, "data")
    flash.read_page(p1, "translation")

    assert metrics.data_reads == 1
    assert metrics.tp_reads == 1
    assert metrics.flash_writes == 1
    assert metrics.total_flash_reads == 2


def test_allocation():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)

    p1 = flash.allocate_page()
    p2 = flash.allocate_page()
    assert p1 != p2
    assert flash.allocated_pages == 0  # not written yet

    flash.write_page(p1, "data")
    flash.write_page(p2, "data")
    assert flash.allocated_pages == 2


def test_free_page():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)

    p1 = flash.allocate_page()
    flash.write_page(p1, "data")
    assert flash.allocated_pages == 1

    flash.free_page(p1)
    assert flash.allocated_pages == 0


def test_erase_block():
    config = small_config()
    metrics = Metrics()
    flash = Flash(config, metrics)

    flash.erase_block(0)
    assert metrics.flash_erases == 1


if __name__ == "__main__":
    test_read_write_counting()
    test_allocation()
    test_free_page()
    test_erase_block()
    print("all flash tests passed")
