from src.config import SSDConfig
from src.metrics import Metrics


class Flash:
    def __init__(self, config: SSDConfig, metrics: Metrics):
        self.config = config
        self.metrics = metrics
        self.page_size = config.flash.page_size
        self.pages_per_block = config.flash.pages_per_block
        self.total_pages = config.capacity_bytes // config.flash.page_size
        self.total_blocks = self.total_pages // self.pages_per_block

        # track allocated pages and block erase counts
        self._allocated = set()
        self._next_page = 0
        self._erase_counts = [0] * self.total_blocks

    def read_page(self, _page_id, page_type="data"):
        # count the flash read, distinguishing translation vs data pages
        self.metrics.record_flash_read(page_type)

    def write_page(self, page_id, _page_type="data"):
        self.metrics.flash_writes += 1
        self._allocated.add(page_id)

    def allocate_page(self):
        # simple sequential allocator
        while self._next_page in self._allocated:
            self._next_page += 1
            if self._next_page >= self.total_pages:
                raise RuntimeError("flash full")
        page_id = self._next_page
        self._next_page += 1
        return page_id

    def free_page(self, page_id):
        self._allocated.discard(page_id)

    def erase_block(self, block_id):
        self.metrics.flash_erases += 1
        self._erase_counts[block_id] += 1
        start = block_id * self.pages_per_block
        for pid in range(start, start + self.pages_per_block):
            self._allocated.discard(pid)

    @property
    def allocated_pages(self):
        return len(self._allocated)
