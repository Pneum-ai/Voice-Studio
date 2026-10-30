"""Verify admission and decode priority against simulated playable-audio timelines."""

from app.backends.mlx.playback import PlaybackPolicy


def test_admission_waits_for_playable_audio_from_every_listener() -> None:
  """Silence removed by the caller cannot fund a new prefill or starve another row."""
  policy = PlaybackPolicy()
  assert policy.can_admit([], rows=8, now=10, until_decode=6)
  policy.received(1, 0, now=10)
  policy.received(2, 1, now=10)
  assert not policy.can_admit([1, 2], rows=1, now=10, until_decode=6)
  policy.received(1, 0.5, now=10)
  assert policy.can_admit([1, 2], rows=1, now=10, until_decode=6)
  assert not policy.can_admit([1, 2], rows=1, now=10.2, until_decode=6)


def test_new_prefill_cost_cannot_consume_existing_playback_reserve() -> None:
  """A slow observed prefill delays newcomers until the consumer has more audio."""
  policy = PlaybackPolicy()
  policy.received(1, 0.5, now=10)
  assert policy.can_admit([1], rows=2, now=10, until_decode=6)
  policy.observe("prefill", 0.8, rows=2)
  assert not policy.can_admit([1], rows=2, now=10, until_decode=6)
  policy.received(1, 0.8, now=10.1)
  assert policy.can_admit([1], rows=2, now=10.1, until_decode=6)


def test_single_newcomer_reserves_fixed_prefill_cost() -> None:
  """A large batch's average row cost cannot underprice the next single admission."""
  policy = PlaybackPolicy()
  policy.observe("prefill", 0.8, rows=8)
  policy.received(1, 0.6, now=10)
  assert not policy.can_admit([1], rows=1, now=10, until_decode=1)
  policy.received(1, 0.5, now=10.1)
  assert policy.can_admit([1], rows=1, now=10.1, until_decode=1)


def test_decode_preempts_cadence_before_playback_runs_out() -> None:
  """Buffered rows wait for batching while nearly empty rows receive immediate audio."""
  policy = PlaybackPolicy()
  policy.received(1, 0.5, now=10)
  assert not policy.decode_due(1, pending=2, decoded=6, now=10, until_decode=4)
  assert policy.decode_due(1, pending=2, decoded=6, now=10.4, until_decode=4)
  assert not policy.decode_due(1, pending=0, decoded=6, now=11, until_decode=4)


def test_startup_retries_after_silent_prefix_without_waiting_for_full_chunk() -> None:
  """A trimmed first decode remains a startup request until actual playback starts."""
  policy = PlaybackPolicy()
  assert not policy.decode_due(1, pending=3, decoded=0, now=10, until_decode=3)
  assert policy.decode_due(1, pending=4, decoded=0, now=10, until_decode=2)
  assert not policy.decode_due(1, pending=1, decoded=4, now=10.1, until_decode=1)
  assert policy.decode_due(1, pending=2, decoded=4, now=10.2, until_decode=6)


def test_stall_and_cancellation_do_not_reuse_old_playback_time() -> None:
  """Late audio restarts playback now; cancellation removes its old deadline entirely."""
  policy = PlaybackPolicy()
  policy.received(1, 0.3, now=10)
  policy.received(1, 0.5, now=11)
  assert policy.can_admit([1], rows=1, now=11.1, until_decode=6)
  policy.forget(1)
  policy.forget(1)
  assert not policy.can_admit([1], rows=1, now=11.1, until_decode=6)
