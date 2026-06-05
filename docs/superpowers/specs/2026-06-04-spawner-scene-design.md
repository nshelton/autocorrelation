# Spawner scene — design

**Date:** 2026-06-04
**Status:** Approved, pending implementation plan

## Goal

A new visualizer scene component that spawns discrete physics objects (cubes,
spheres, disks) on demand. Each object type has a GUI button that spawns one
object; because the component system already wires `paramButtons` to the audio
`Modulator`, each button is automatically drivable by an audio trigger
(source + threshold). Objects live under physics, have a finite lifetime, and
are pushed around by a selectable force field.

This component shares the same architecture as `ParticleView`: a Rapier
`World`, instanced rendering, a stable params bag, and the hot-param sweep
pattern.

## Component shape

- New file: `src/render/components/Spawner.ts`, implementing `Component`.
- `id = "spawner"`, `label = "Spawner"`, `paramPrefix = "spawner"`.
- Registered in the same component class list `ParticleView` is registered in
  (the array passed to `ComponentManager`).
- Owns: one Rapier `World`, three `InstancedMesh`es (one per shape), a params
  bag, and per-slot SoA state arrays.

## Object types & pools

Three shapes. Instancing requires a uniform geometry per `InstancedMesh`, so
each shape gets its own mesh AND its own fixed ring buffer of bodies.

| Type   | Three.js geometry            | Rapier collider |
|--------|------------------------------|-----------------|
| Cube   | `BoxGeometry`                | `cuboid`        |
| Sphere | `IcosahedronGeometry`        | `ball`          |
| Disk   | `CylinderGeometry` (thin)    | `cylinder`      |

- `MAX_PER_TYPE = 512` per shape (1536 total). Stays under the WebGPU
  1024-instance uniform-buffer cliff documented in `ParticleView.createInstancedMesh`.
- Bodies are pre-created as `dynamic` but `setEnabled(false)` until first
  spawned. Disabled bodies are out of the simulation, so empty slots cost
  nothing and don't collide.
- Each shape has a `next` ring index. Spawning activates slot `next`,
  advancing `next = (next + 1) % MAX_PER_TYPE`. If that slot is currently
  active (pool full), it is recycled — the oldest live object of that type is
  overwritten.
- Per-slot state (SoA, sized `MAX_PER_TYPE` per shape): `active` flag,
  `lifetime`, `maxLifetime`, `scale` (random size jitter). On expiry
  (`lifetime <= 0`): set `active = false`, `setEnabled(false)`, and write a
  zero-scale instance matrix so the object disappears.

## Spawning

Three `paramButtons`: **Spawn Cube**, **Spawn Sphere**, **Spawn Disk**.

`paramButtons[].onClick` receives only the `ParamStore`, not the live component
instance. To bridge the button to the instance, use a **module singleton
`spawnQueue`** — the same pattern OrbitalCloud uses with the `shTween`
singleton for "Randomize SH". Shape:

```ts
// pending spawn counts per type; drained by the live Spawner each frame
class SpawnQueue {
  cube = 0; sphere = 0; disk = 0;
  request(type: "cube" | "sphere" | "disk"): void { this[type]++; }
  reset(): void { this.cube = this.sphere = this.disk = 0; }
}
export const spawnQueue = new SpawnQueue();
```

- Each button's `onClick` calls `spawnQueue.request("<type>")`.
- The live `Spawner.update()` drains the queue: spawns that many objects of
  each type, then resets the counts.
- `Spawner` calls `spawnQueue.reset()` on construction so presses that
  accumulated while the component was disabled don't burst-spawn on enable.

**Audio triggers come for free.** `ComponentManager.bindUI` already registers
every `paramButton` with `modulator.registerTriggerCallback(triggerKey, fire)`
and injects the trigger popover via `bindTrigger`. So each spawn button gets a
∿ button → audio source + threshold popover with no extra code in `Spawner`.

### Spawn initial conditions

Spawn at the **world origin** `(0, 0, 0)` with a **random-direction impulse**
of magnitude `spawnImpulse`:

```
dir = random point on unit sphere
linvel = dir * spawnImpulse
```

Reset lifetime to `lifetime + random()*JITTER`, pick a random `scale` in
`[SCALE_MIN, SCALE_MAX]`, set the collider radius/half-extents from `scale *
objectScale`, `setEnabled(true)`, `active = true`.

## Force field

Discrete param `forceFieldType` (kind `discrete`, options `[0, 1]`, labels
`["Linear", "Curl"]`) — same mechanism as OrbitalCloud's `renderMode`.

- **Linear (0)** — constant direction force. Implemented as Rapier **world
  gravity** set to `(0, -forceStrength, 0)`. Combined with origin + impulse
  spawn this is a fountain/confetti look. (Direction is fixed -Y in v1.)
- **Curl (1)** — world gravity set to `(0, 0, 0)`; each active body gets a
  curl-noise velocity impulse scaled by `forceStrength`, reusing
  `createCurlNoise({ scale: noiseScale })` exactly as `ParticleView` does
  (recreate the noise fn only when `noiseScale` changes — it's closed over at
  construction).

`forceFieldType` is read each frame; switching sets `world.gravity` accordingly
(only on change). Adding more field types later = extend the option set + the
switch in `update()`.

## Parameters

All continuous unless noted. Defaults are starting points, tunable later.

| Key              | Kind       | Range / options          | Default | Notes |
|------------------|------------|--------------------------|---------|-------|
| `forceFieldType` | discrete   | `[0,1]` = Linear/Curl    | 0       | switches gravity vs curl |
| `forceStrength`  | continuous | 0..30                    | 9.8     | gravity magnitude (Linear) / curl impulse scale (Curl) |
| `noiseScale`     | continuous | 0.01..1.0                | 0.5     | curl spatial scale |
| `spawnImpulse`   | continuous | 0..10                    | 3       | initial random velocity magnitude |
| `lifetime`       | continuous | 1..15                    | 4       | seconds; per-object jitter added |
| `restitution`    | continuous | 0..1                     | 0.5     | bounciness; hot-swept |
| `damping`        | continuous | 0..2                     | 0.1     | linear + angular damping; hot-swept |
| `timescale`      | continuous | 0..3                     | 1.0     | physics dt multiplier; hot-swept |
| `objectScale`    | continuous | 0.2..3                   | 1.0     | global visual + collider size multiplier |

Hot params `timescale`, `damping`, `restitution` are swept across active bodies/
colliders only when their slider value changes (guarded by `lastFoo`), same as
`ParticleView`. `numParticles`-style reconfig is **not** needed — pool size is a
fixed constant.

## Rendering

- One `InstancedMesh` per shape, sized to `MAX_PER_TYPE`, built with
  `MeshBasicNodeMaterial` and the same `normalWorld` lambert lighting node
  `ParticleView` uses (`ndotl * 0.7 + 0.3`).
- Per-type distinct flat color (e.g. cube / sphere / disk each a different hue)
  so the three types read clearly. Color carried per-instance via
  `InstancedBufferAttribute` (matching `ParticleView`) OR a single material
  color per mesh (simpler, since each mesh is one type) — prefer one flat color
  per mesh.
- Each active slot writes its body translation + rotation + scale into the
  instance matrix each frame; inactive slots write a zero-scale matrix.

## Lifecycle

- `constructor`: `RAPIER.init()`, allocate SoA arrays, create the `World` (no
  ground plane), pre-create disabled bodies for all three pools, build the
  three `InstancedMesh`es, `spawnQueue.reset()`.
- `update()`: apply hot-param sweeps + force-field-type gravity switch, drain
  `spawnQueue`, `world.step()`, per-slot lifetime decrement / expiry / force
  field / matrix write.
- `dispose()`: free the Rapier world, dispose the three meshes' geometry +
  material, clear arrays. (Follows `ParticleView.dispose`.)

## Out of scope (v1)

- No ground plane — objects fall/drift away and recycle on lifetime expiry.
- No attractor or swirl field (that is `ParticleView`'s behavior).
- No per-shape lifetime — single shared `lifetime` param.
- No configurable linear-force direction (fixed -Y).
- No object-count reconfig — pool size is a compile-time constant.
