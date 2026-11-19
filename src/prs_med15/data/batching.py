"""Source-balanced batching that never mixes native 2D and 3D examples."""
from __future__ import annotations

from collections import defaultdict
import random

import torch.distributed as dist
from torch.utils.data import Sampler

from .schema import Dialogue


class SourceBalancedBucketBatchSampler(Sampler[list[int]]):
    """Build homogeneous global batches, then assign disjoint batches to ranks."""

    def __init__(
        self,
        rows,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        balance_by_turns: bool = False,
        match_ranks: bool = False,
        match_window: int = 32,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.start_batch = 0  # one-shot offset used when resuming mid-epoch
        self.balance_by_turns = balance_by_turns
        # Give the ranks of each micro-step batches of the same bucket and similar cost, so the
        # fast rank does not idle at the gradient all-reduce while the other runs a 3D volume.
        self.match_ranks = match_ranks
        self.match_window = max(1, match_window)
        self.bucket_of: dict[int, str] = {}
        self.cost: dict[int, float] = {}
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if world_size is None:
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = rank
        self.world_size = world_size
        self.groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.turn_counts: dict[int, int] = {}
        for index, row in enumerate(rows):
            dialogue = row if hasattr(row, "scans") else None
            if dialogue is None:
                dialogue = Dialogue.from_json(row)
            kind = dialogue.scans[0].input_kind
            bucket = "2d" if kind == "native_2d" else "3d"
            self.groups[(bucket, dialogue.source)].append(index)
            self.turn_counts[index] = max(1, len(dialogue.turns))
            self.bucket_of[index] = bucket
            # Cost proxy: turns x actions-to-decode (each action runs the mask decoder per slice).
            actions = sum(len(turn.actions) for turn in dialogue.turns)
            self.cost[index] = self.turn_counts[index] + actions

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _bucket_batches(self, rng: random.Random) -> dict[str, list[list[int]]]:
        queues = {key: list(values) for key, values in self.groups.items()}
        for values in queues.values():
            rng.shuffle(values)
        by_bucket: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in queues:
            by_bucket[key[0]].append(key)
        output: dict[str, list[list[int]]] = defaultdict(list)
        for bucket, keys in by_bucket.items():
            source_cursor = 0
            while any(queues[key] for key in keys):
                active = [key for key in keys if queues[key]]
                batch: list[int] = []
                while len(batch) < self.batch_size and active:
                    key = active[source_cursor % len(active)]
                    batch.append(queues[key].pop())
                    active = [candidate for candidate in active if queues[candidate]]
                    source_cursor += 1
                if len(batch) == self.batch_size or not self.drop_last:
                    output[bucket].append(batch)
        return output

    def _balanced_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        by_bucket = self._bucket_batches(rng)
        if self.balance_by_turns and len(by_bucket) > 1:
            available_turns = {
                bucket: sum(self.turn_counts[index] for batch in batches for index in batch)
                for bucket, batches in by_bucket.items()
            }
            target = min(available_turns.values())
            for bucket, batches in list(by_bucket.items()):
                kept: list[list[int]] = []
                turns = 0
                for batch in batches:
                    if kept and turns >= target:
                        break
                    kept.append(batch)
                    turns += sum(self.turn_counts[index] for index in batch)
                by_bucket[bucket] = kept

        output: list[list[int]] = []
        cursors = {bucket: 0 for bucket in by_bucket}
        emitted_turns = {bucket: 0 for bucket in by_bucket}
        while any(cursors[bucket] < len(by_bucket[bucket]) for bucket in by_bucket):
            active = [bucket for bucket in by_bucket if cursors[bucket] < len(by_bucket[bucket])]
            bucket = min(active, key=lambda value: (emitted_turns[value], value))
            batch = by_bucket[bucket][cursors[bucket]]
            cursors[bucket] += 1
            emitted_turns[bucket] += sum(self.turn_counts[index] for index in batch)
            output.append(batch)
        return output

    def _matched_groups(self, batches: list[list[int]]) -> list[list[int]]:
        """Reorder into consecutive groups of world_size same-bucket, cost-sorted batches."""
        world = self.world_size
        rng = random.Random(self.seed + self.epoch + 7919)
        per_bucket: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
        for position, batch in enumerate(batches):
            per_bucket[self.bucket_of[batch[0]]].append((position, batch))
        groups: list[tuple[float, list[list[int]]]] = []
        window = self.match_window * world
        for entries in per_bucket.values():
            usable = len(entries) - len(entries) % world
            for start in range(0, usable, window):
                chunk = sorted(entries[start:min(start + window, usable)],
                               key=lambda value: sum(self.cost[i] for i in value[1]))
                for g in range(0, len(chunk), world):
                    members = chunk[g:g + world]
                    rng.shuffle(members)  # no rank always gets the heavier batch of a pair
                    order = sum(position for position, _ in members) / world
                    groups.append((order, [batch for _, batch in members]))
        groups.sort(key=lambda value: value[0])  # keep the bucket/turn interleaving of the base order
        return [batch for _, group in groups for batch in group]

    def _rank_batches(self) -> list[list[int]]:
        batches = self._balanced_batches()
        if self.world_size > 1 and self.match_ranks:
            batches = self._matched_groups(batches)
        if self.world_size > 1:
            usable = len(batches) - (len(batches) % self.world_size)
            batches = batches[:usable]
        return batches[self.rank::self.world_size]

    def __iter__(self):
        start, self.start_batch = self.start_batch, 0
        yield from self._rank_batches()[start:]

    def __len__(self) -> int:
        return len(self._rank_batches())
