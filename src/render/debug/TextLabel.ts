import {
  CanvasTexture,
  Color,
  ColorRepresentation,
  LinearFilter,
  Sprite,
  SpriteMaterial,
} from "three";

export type TextLabelAnchorX = "left" | "center" | "right";
export type TextLabelAnchorY = "top" | "center" | "bottom";

export interface TextLabelOptions {
  text: string;
  color?: ColorRepresentation;
  background?: string;
  fontSize?: number;
  padding?: number;
  height?: number;
  textureWidth?: number;
  textureHeight?: number;
  maxTextureWidth?: number;
  anchorX?: TextLabelAnchorX;
  anchorY?: TextLabelAnchorY;
}

export class TextLabel {
  readonly object3d: Sprite;

  private canvas = document.createElement("canvas");
  private context: CanvasRenderingContext2D;
  private texture: CanvasTexture;
  private material: SpriteMaterial;
  private text = "";
  private color: ColorRepresentation;
  private background: string;
  private fontSize: number;
  private padding: number;
  private height: number;
  private fixedTextureWidth?: number;
  private textureHeight: number;
  private maxTextureWidth: number;
  private anchorX: TextLabelAnchorX;
  private anchorY: TextLabelAnchorY;

  constructor(opts: TextLabelOptions) {
    this.fixedTextureWidth = opts.textureWidth;
    this.textureHeight = Math.max(1, Math.ceil(opts.textureHeight ?? 128));
    this.maxTextureWidth = Math.max(1, Math.ceil(opts.maxTextureWidth ?? 4096));
    this.canvas.width = Math.max(1, Math.ceil(this.fixedTextureWidth ?? 1));
    this.canvas.height = this.textureHeight;

    const context = this.canvas.getContext("2d");
    if (!context) throw new Error("TextLabel: 2D canvas context unavailable");

    this.context = context;
    this.color = opts.color ?? 0xffffff;
    this.background = opts.background ?? "rgba(0, 0, 0, 0.45)";
    this.fontSize = opts.fontSize ?? 48;
    this.anchorX = opts.anchorX ?? "center";
    this.anchorY = opts.anchorY ?? "center";
    this.padding = opts.padding ?? 12;
    this.height = opts.height ?? 0.08;

    this.texture = new CanvasTexture(this.canvas);
    this.texture.minFilter = LinearFilter;
    this.texture.magFilter = LinearFilter;
    this.texture.generateMipmaps = false;

    this.material = new SpriteMaterial({
      map: this.texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.object3d = new Sprite(this.material);
    this.object3d.center.set(this.anchorCenterX(), this.anchorCenterY());
    this.object3d.frustumCulled = false;
    this.text = opts.text;
    this.redraw();
  }

  setText(text: string): void {
    if (text === this.text) return;
    this.text = text;
    this.redraw();
  }

  dispose(): void {
    this.texture.dispose();
    this.material.dispose();
    this.object3d.parent?.remove(this.object3d);
  }

  private redraw(): void {
    const font = `${this.fontSize}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
    this.context.font = font;

    const width = this.measureTextureWidth();
    const height = this.textureHeight;

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      // CanvasTexture allocates the GPU texture at the canvas's current
      // dims and reuses that allocation across uploads. Resizing the
      // source canvas afterward causes three's WebGPU renderer to copy
      // a too-large source into a too-small destination — validation
      // errors. Recreate the texture so the GPU side matches.
      this.texture.dispose();
      this.texture = new CanvasTexture(this.canvas);
      this.texture.minFilter = LinearFilter;
      this.texture.magFilter = LinearFilter;
      this.texture.generateMipmaps = false;
      this.material.map = this.texture;
      this.material.needsUpdate = true;
    }

    this.context.clearRect(0, 0, width, height);
    this.context.fillStyle = this.background;
    this.context.fillRect(0, 0, width, height);
    this.context.font = font;
    this.context.textBaseline = "middle";
    this.context.fillStyle = this.cssColor(this.color);
    this.context.fillText(
      this.text,
      this.padding,
      height / 2,
      Math.max(1, width - 2 * this.padding),
    );

    this.texture.needsUpdate = true;
    this.object3d.scale.set((width / height) * this.height, this.height, 1);
  }

  private measureTextureWidth(): number {
    if (this.fixedTextureWidth !== undefined) {
      return Math.max(1, Math.ceil(this.fixedTextureWidth));
    }

    const metrics = this.context.measureText(this.text);
    const boundsWidth =
      Math.abs(metrics.actualBoundingBoxLeft ?? 0) +
      Math.abs(metrics.actualBoundingBoxRight ?? 0);
    const measuredWidth = Math.max(metrics.width, boundsWidth);
    const paddedWidth = Math.ceil(measuredWidth + 2 * this.padding + 2);
    return Math.max(1, Math.min(this.maxTextureWidth, paddedWidth));
  }

  private anchorCenterY(): number {
    switch (this.anchorY) {
      case "top":
        return 1;
      case "bottom":
        return 0;
      case "center":
        return 0.5;
    }
  }

  private anchorCenterX(): number {
    switch (this.anchorX) {
      case "left":
        return 0;
      case "right":
        return 1;
      case "center":
        return 0.5;
    }
  }

  private cssColor(color: ColorRepresentation): string {
    const c = new Color(color);
    return `rgb(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(
      c.b * 255,
    )})`;
  }
}
