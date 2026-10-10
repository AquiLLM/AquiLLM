"""CPU oracle for a frozen device policy; never called during graph replay."""
from .contracts import SplitPlan


def _context(prior_len: int) -> None:
    if type(prior_len) is not int or prior_len < 0:
        raise ValueError("prior_len must be a nonnegative integer")


def select_active_splits(prior_len: int, plan: SplitPlan) -> int:
    _context(prior_len)
    for upper, splits in plan.buckets:
        if prior_len <= upper:
            return splits
    return plan.buckets[-1][1] if plan.buckets else plan.max_splits


def committed_interval(prior_len: int, active_splits: int, split_id: int) -> tuple[int, int]:
    """Mathematical partition oracle, including empty splits shorter than the grid."""
    _context(prior_len)
    if type(active_splits) is not int or active_splits <= 0:
        raise ValueError("active_splits must be a positive integer")
    if type(split_id) is not int or not 0 <= split_id < active_splits:
        raise ValueError("split_id must address an active committed split")
    step = (prior_len + active_splits - 1) // active_splits
    return min(split_id * step, prior_len), min((split_id + 1) * step, prior_len)
