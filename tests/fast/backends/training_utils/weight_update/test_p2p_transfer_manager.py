import threading

import pytest


def _blocking_task(started: threading.Event, release: threading.Event):
    def run() -> None:
        started.set()
        assert release.wait(timeout=30.0)

    return run


class TestSubmitReturningFuture:
    """Submission of P2P writes to the background executor."""

    def test_a_future_handed_to_the_caller_is_still_tracked_for_the_bulk_wait(self, p2p_transfer_utils) -> None:
        """The last engine rank never calls result() itself, so its write must still be awaited by wait_transfers."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        started, release = threading.Event(), threading.Event()

        future = manager.submit_returning_future(_blocking_task(started, release))
        assert started.wait(timeout=30.0)

        assert manager.transfer_futures == [future]
        assert not future.done()

        release.set()
        manager.wait_transfers()

        assert future.done()
        assert manager.transfer_futures == []

    def test_every_submitted_write_runs_before_the_bulk_wait_returns(self, p2p_transfer_utils) -> None:
        """More writes than workers are submitted per bucket, so the queued ones must finish too."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        lock = threading.Lock()
        finished: list[int] = []

        for index in range(6):

            def run(index: int = index) -> None:
                with lock:
                    finished.append(index)

            manager.submit_returning_future(run)

        manager.wait_transfers()

        assert sorted(finished) == list(range(6))


class TestWaitTransfers:
    """Collection of the background P2P writes at the end of a weight update."""

    def test_a_completed_round_of_transfers_finishes_quietly(self, p2p_transfer_utils) -> None:
        """The happy path must stay silent and forget the futures it already collected."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)
        manager.submit_returning_future(lambda: None)

        manager.wait_transfers()

        assert manager.transfer_futures == []

    def test_a_failed_transfer_is_raised_to_the_caller(self, p2p_transfer_utils) -> None:
        """A silently logged RDMA failure would publish a half-written weight version, so it must propagate."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=30.0)

        def failing() -> None:
            raise RuntimeError("[P2P-Shared] Transfer failed for session s0, error: -1")

        manager.submit_returning_future(failing)

        with pytest.raises(RuntimeError, match="error: -1"):
            manager.wait_transfers()

    def test_every_transfer_is_awaited_even_after_one_of_them_failed(self, p2p_transfer_utils) -> None:
        """One broken target must not stop the wait, otherwise later writes stay in flight over freed buffers."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=1, transfer_timeout=30.0)
        finished: list[str] = []

        def failing(name: str) -> None:
            raise RuntimeError(f"boom-{name}")

        def succeeding(name: str) -> None:
            finished.append(name)

        manager.submit_returning_future(failing, "first")
        manager.submit_returning_future(succeeding, "second")
        manager.submit_returning_future(failing, "third")

        with pytest.raises(RuntimeError) as raised:
            manager.wait_transfers()

        assert "boom-first" in str(raised.value)
        assert "boom-third" in str(raised.value)
        assert finished == ["second"]

    def test_a_timed_out_transfer_raises_and_stays_tracked_until_it_really_finishes(self, p2p_transfer_utils) -> None:
        """A timed-out write is still reading the shared buffer, so it must not be dropped as if it had completed."""
        manager = p2p_transfer_utils.P2PTransferManager(num_workers=2, transfer_timeout=0.05)
        started, release = threading.Event(), threading.Event()

        try:
            future = manager.submit_returning_future(_blocking_task(started, release))
            assert started.wait(timeout=30.0)

            with pytest.raises(RuntimeError, match="1 of 1 transfers failed"):
                manager.wait_transfers()

            assert manager.transfer_futures == [future]
        finally:
            release.set()

        manager.transfer_timeout = 30.0
        manager.wait_transfers()

        assert future.done()
        assert manager.transfer_futures == []
