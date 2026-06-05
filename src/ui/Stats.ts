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
    dom.style.left = "auto";
    dom.style.right = "1rem";
    dom.style.zIndex = "10";
    // stats.js draws its panel on a <canvas>, so its dark backdrop can't be made
    // transparent — but it fades/hides with the rest of the GUI via .gui-el.
    dom.classList.add("gui-el");
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
