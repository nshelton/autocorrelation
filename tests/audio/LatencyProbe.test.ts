import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { LatencyProbe } from "../../src/audio/LatencyProbe";

const SR = 48000;
const HOP_SECS = 1024 / SR; // 21.33 ms

// Minimal AudioContext stand-in: the probe only needs sampleRate, currentTime,
// and a buffer source it can schedule. `started` records the scheduled context
// time so the test can drive frames relative to it.
function fakeCtx(currentTime = 10) {
  const started: number[] = [];
  const connected: unknown[] = [];
  return {
    started,
    connected,
    ctx: {
      sampleRate: SR,
      currentTime,
      createBuffer: (_ch: number, len: number) => ({
        getChannelData: () => new Float32Array(len),
      }),
      createBufferSource: () => ({
        buffer: null,
        onended: null,
        connect: (n: unknown) => connected.push(n),
        disconnect: () => {},
        start: (t: number) => started.push(t),
      }),
    } as unknown as AudioContext,
  };
}

const DEST = {} as AudioNode;
const onset = (v: number) => new Float32Array([0, 0, v]);

describe("LatencyProbe", () => {
  let now = 0;
  beforeEach(() => {
    now = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => vi.restoreAllMocks());

  it("tracks deliver and age on every frame without a probe armed", () => {
    const probe = new LatencyProbe(fakeCtx().ctx, DEST);
    now = 1002;
    probe.onFeatures(10.5, 1000, onset(0.1)); // published 2 ms ago
    expect(probe.deliverMs).toBeCloseTo(2, 6);
    now = 1009;
    probe.onConsume(now);
    expect(probe.ageMs).toBeCloseTo(9, 6); // 2 ms deliver + 7 ms RAF wait
  });

  it("age grows on a frame that got no new analysis", () => {
    const probe = new LatencyProbe(fakeCtx().ctx, DEST);
    now = 1002;
    probe.onFeatures(10.5, 1000, onset(0.1));
    probe.onConsume(1005);
    expect(probe.ageMs).toBeCloseTo(5, 6);
    // Next RAF frame, still no new message: the same hop is now 21 ms stale.
    probe.onConsume(1021);
    expect(probe.ageMs).toBeCloseTo(21, 6);
  });

  it("schedules the impulse ahead of currentTime and reports the detect lag in hops", () => {
    const { ctx, started, connected } = fakeCtx(10);
    const probe = new LatencyProbe(ctx, DEST);
    probe.fire();
    expect(connected).toEqual([DEST]);
    expect(started).toHaveLength(1);
    const clickT = started[0];
    expect(clickT).toBeCloseTo(10.2, 6);

    // Two lead-in frames establish the onset floor, then the burst is detected
    // on the second frame after it — one full hop of window-taper delay.
    probe.onFeatures(clickT - 2 * HOP_SECS, 1000, onset(0.05));
    probe.onFeatures(clickT - HOP_SECS, 1000 + HOP_SECS * 1000, onset(0.04));
    probe.onFeatures(clickT + 0.004, 1021, onset(0.06)); // burst at the taper edge
    now = 1044;
    probe.onFeatures(clickT + 0.004 + HOP_SECS, 1042, onset(0.98)); // detected

    // Report waits for consume + render; only the "armed" line so far.
    expect(console.log).toHaveBeenCalledTimes(1);

    probe.onConsume(1050); // scene read it 6 ms after receipt
    probe.onRendered(1053); // GPU submit 3 ms later

    // 4 ms into the hop + one full hop of window taper = 25.33 ms = 1.19 hops.
    const report = vi.mocked(console.log).mock.calls.at(-1)?.[0] as string;
    expect(report).toContain("1.19 hops");
    expect(report).toMatch(/impulse → onset rise\s+25\.3 ms/);
    expect(report).toMatch(/worklet → main thread\s+2\.0 ms/);
    expect(report).toMatch(/receipt → scene consume\s+6\.0 ms/);
    expect(report).toMatch(/consume → GPU submit\s+3\.0 ms/);
    expect(report).toMatch(/software path total\s+36\.3 ms/);
  });

  it("clears the floor relative to pre-impulse onset, not an absolute level", () => {
    const { ctx, started } = fakeCtx(10);
    const probe = new LatencyProbe(ctx, DEST);
    probe.fire();
    const clickT = started[0];
    // Loud input: the floor is already 0.9, so 0.95 must NOT count as a rise.
    probe.onFeatures(clickT - HOP_SECS, 1000, onset(0.9));
    probe.onFeatures(clickT + 0.001, 1021, onset(0.95));
    probe.onConsume(1030);
    probe.onRendered(1033);
    expect(console.log).toHaveBeenCalledTimes(1); // only the "armed" line
  });

  it("aborts instead of hanging when no rise ever arrives", () => {
    const { ctx, started } = fakeCtx(10);
    const probe = new LatencyProbe(ctx, DEST);
    probe.fire();
    const clickT = started[0];
    for (let i = 0; i < 30; i++) {
      probe.onFeatures(clickT + i * HOP_SECS, 1000 + i, onset(0.01));
    }
    expect(console.warn).toHaveBeenCalledOnce();
    // Disarmed: a later rise must not resurrect the aborted run.
    probe.onFeatures(clickT + 40 * HOP_SECS, 2000, onset(1.0));
    probe.onConsume(2010);
    probe.onRendered(2013);
    expect(console.log).toHaveBeenCalledTimes(1);
  });
});
