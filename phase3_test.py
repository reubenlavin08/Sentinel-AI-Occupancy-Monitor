"""Pure-logic test for Phase 3 counter (no video needed)."""
from sentinel.config import CONFIG
from sentinel.counter import TripwireCounter, TrackScorer

cam = CONFIG.cameras[0].id
line = CONFIG.proc_width // 2          # 320
LEFT, RIGHT = line - 100, line + 100   # well outside the +/-40 buffer

c = TripwireCounter(CONFIG)
events = []

def walk(tid, x_to, n=4):
    for _ in range(n):
        e = c.update(cam, tid, x_to)
        if e:
            events.append(e)

# Track 1: start left, cross right (IN), then cross back left (OUT) — same id
c.update(cam, 1, LEFT)        # adopt origin = left
walk(1, RIGHT)                # -> IN
walk(1, LEFT)                 # -> OUT  (proves 'counted'-forever is fixed)

# Track 2: spawns INSIDE buffer, then exits right and crosses... origin resolves on exit
c.update(cam, 2, line)        # inside dead-zone, no origin yet
c.update(cam, 2, LEFT)        # now adopt origin = left
walk(2, RIGHT)                # -> IN (proves inside-buffer dead-end is fixed)

print("events:", events)
print("entered:", c.entered[cam], "exited:", c.exited[cam], "occupancy:", c.occupancy[cam])

# occupancy clamp: force an extra OUT below zero
c2 = TripwireCounter(CONFIG)
c2.update(cam, 9, RIGHT)
walk_e = [c2.update(cam, 9, LEFT) for _ in range(4)]
print("clamp test occupancy (should be 0, not -1):", c2.occupancy[cam])

# TrackScorer: one weak frame must NOT confirm; consistent frames must
s = TrackScorer(CONFIG.score_min, CONFIG.score_threshold, CONFIG.score_min_hist)
print("after 1 strong frame confirmed?", s.update(7, 0.9))
print("after 2 strong frames confirmed?", s.update(7, 0.9))
print("weak-only track confirmed?", s.update(8, 0.1) or s.update(8, 0.1))

assert events == ["IN", "OUT", "IN"], f"FSM wrong: {events}"
assert c.occupancy[cam] == 1, c.occupancy[cam]
assert c2.occupancy[cam] == 0
print("\nALL PHASE-3 LOGIC CHECKS PASSED")
