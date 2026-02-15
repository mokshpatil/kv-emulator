from src.flash import Flash
from src.metrics import Metrics


class GarbageCollector:
    def __init__(self, flash: Flash, metrics: Metrics, threshold: float = 0.85):
        self.flash = flash
        self.metrics = metrics
        # trigger GC when flash utilization exceeds this threshold
        self.threshold = threshold
        # callbacks set by KVSSD for relocating pages
        self._relocate_data_page = None
        self._relocate_translation_page = None

    def set_relocators(self, data_fn, translation_fn):
        # KVSSD provides these callbacks to handle page relocation
        self._relocate_data_page = data_fn
        self._relocate_translation_page = translation_fn

    def should_run(self):
        return self.flash.utilization >= self.threshold

    def run(self, max_rounds=10, force=False):
        # run GC until utilization drops below threshold or no victim found
        rounds = 0
        while rounds < max_rounds:
            if not force and not self.should_run():
                break
            victim = self._select_victim()
            if victim is None:
                break
            self._collect_block(victim)
            rounds += 1
            force = False  # only force the first round
        return rounds

    def _select_victim(self):
        # greedy: pick block with most invalid pages
        best_block = None
        best_invalid = 0
        for block_id in range(self.flash.total_blocks):
            invalid = self.flash.invalid_count_in_block(block_id)
            if invalid > best_invalid:
                best_invalid = invalid
                best_block = block_id
        # only collect if there are actually invalid pages to reclaim
        if best_invalid == 0:
            return None
        return best_block

    def _collect_block(self, block_id):
        self.metrics.gc_invocations += 1
        valid_pages = self.flash.valid_pages_in_block(block_id)

        # copy valid pages to new locations
        for old_page_id, page_type in valid_pages:
            self.metrics.gc_pages_copied += 1
            # read old page
            self.flash.read_page(old_page_id, page_type)
            # allocate new page and write
            new_page_id = self.flash.allocate_page()
            self.flash.write_page(new_page_id, page_type)

            # notify KVSSD to update its mappings
            if page_type == "translation" and self._relocate_translation_page:
                self._relocate_translation_page(old_page_id, new_page_id)
            elif page_type == "data" and self._relocate_data_page:
                self._relocate_data_page(old_page_id, new_page_id)

        # erase the victim block
        self.flash.erase_block(block_id)
