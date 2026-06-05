import { Pane } from "tweakpane";

// A single floating popover that hosts a parameter's modulation (or trigger)
// controls, anchored next to the small inline button on a slider/button row.
// Module singleton (like shTween) so bindParam/bindTrigger reach it directly —
// no need to thread it through the App/PostStack/main bind chain. The `build`
// callback captures the live store/modulator from the current panel build, so
// HMR stays correct; ParamPanel.close()s it on teardown.
class ModPopover {
  private root: HTMLDivElement;
  private pane: Pane;
  private anchor: HTMLElement | null = null;

  constructor() {
    this.root = document.createElement("div");
    this.root.style.cssText =
      "position:fixed; z-index:1000; display:none; width:240px;";
    document.body.appendChild(this.root);
    this.pane = new Pane({ container: this.root });

    // Capture phase so we see the click before the row/button handlers. Close
    // unless the click is inside the popover or on the current anchor (the
    // anchor's own click handler does the toggle).
    document.addEventListener(
      "pointerdown",
      (e) => {
        if (!this.isOpen()) return;
        const t = e.target as Node;
        if (this.root.contains(t)) return;
        if (this.anchor && this.anchor.contains(t)) return;
        this.close();
      },
      true,
    );
  }

  private isOpen(): boolean {
    return this.root.style.display !== "none";
  }

  toggle(anchor: HTMLElement, build: (pane: Pane) => void): void {
    if (this.isOpen() && this.anchor === anchor) {
      this.close();
      return;
    }
    this.clear();
    build(this.pane);
    this.anchor = anchor;
    this.root.style.display = "block";
    this.position(anchor);
  }

  close(): void {
    this.root.style.display = "none";
    this.anchor = null;
    this.clear();
  }

  private clear(): void {
    for (const child of [...this.pane.children]) child.dispose();
  }

  // Right-align the popover under the anchor button, clamped to the viewport.
  private position(anchor: HTMLElement): void {
    const r = anchor.getBoundingClientRect();
    const w = this.root.offsetWidth;
    const h = this.root.offsetHeight;
    const left = Math.max(4, Math.min(r.right - w, window.innerWidth - w - 4));
    const top = Math.max(4, Math.min(r.bottom + 2, window.innerHeight - h - 4));
    this.root.style.left = `${left}px`;
    this.root.style.top = `${top}px`;
  }
}

export const modPopover = new ModPopover();
