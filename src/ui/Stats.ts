import Stats from "stats.js";

/**
 * Thin wrapper around stats.js. Mounts a small DOM panel in the
 * top-left corner; begin/end called per render frame.
 */
export class FpsOverlay {
  private stats: Stats;

  constructor() {
    this.stats = new Stats();
    this.stats.showPanel(0); // 0: fps, 1: ms, 2: mb
  }

  mount(parent: HTMLElement = document.body): void {
    const dom = this.stats.dom;
    dom.style.position = "fixed";
    dom.style.top = "1rem";
    dom.style.right = "auto";
    dom.style.left = "1rem";
    dom.style.zIndex = "10";
    parent.appendChild(dom);
  }

  begin(): void {
    this.stats.begin();
  }

  end(): void {
    this.stats.end();
  }

  unmount(): void {
    const dom = this.stats.dom;
    if (dom.parentElement) dom.parentElement.removeChild(dom);
  }
}
