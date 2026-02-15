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

        # _occupied: physically written pages (cleared only by block erase)
        # _valid: logically live pages (cleared by free_page or erase)
        self._occupied = set()
        self._valid = set()
        self._page_types = {}  # page_id -> "data" or "translation"
        self._next_page = 0
        self._erase_counts = [0] * self.total_blocks

    def read_page(self, _page_id, page_type="data"):
        self.metrics.record_flash_read(page_type)

    def write_page(self, page_id, page_type="data"):
        self.metrics.flash_writes += 1
        self._occupied.add(page_id)
        self._valid.add(page_id)
        self._page_types[page_id] = page_type

    def allocate_page(self):
        # find a page not physically occupied (NAND requires erase before rewrite)
        start = self._next_page
        while self._next_page in self._occupied:
            self._next_page = (self._next_page + 1) % self.total_pages
            if self._next_page == start:
                raise RuntimeError("flash full")
        page_id = self._next_page
        self._next_page = (self._next_page + 1) % self.total_pages
        return page_id

    def free_page(self, page_id):
        # invalidate: data is stale but page is physically still written
        self._valid.discard(page_id)

    def erase_block(self, block_id):
        self.metrics.flash_erases += 1
        self._erase_counts[block_id] += 1
        start = block_id * self.pages_per_block
        for pid in range(start, start + self.pages_per_block):
            self._occupied.discard(pid)
            self._valid.discard(pid)
            self._page_types.pop(pid, None)

    def get_block_id(self, page_id):
        return page_id // self.pages_per_block

    def valid_pages_in_block(self, block_id):
        start = block_id * self.pages_per_block
        result = []
        for pid in range(start, start + self.pages_per_block):
            if pid in self._valid:
                result.append((pid, self._page_types.get(pid, "data")))
        return result

    def invalid_count_in_block(self, block_id):
        # pages that are physically occupied but logically invalid
        start = block_id * self.pages_per_block
        count = 0
        for pid in range(start, start + self.pages_per_block):
            if pid in self._occupied and pid not in self._valid:
                count += 1
        return count

    @property
    def allocated_pages(self):
        # count of logically valid pages
        return len(self._valid)

    @property
    def utilization(self):
        # physical occupancy -- drives GC triggering
        return len(self._occupied) / self.total_pages if self.total_pages > 0 else 0.0
