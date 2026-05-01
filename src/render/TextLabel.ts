import {
  CanvasTexture,
  Color,
  ColorRepresentation,
  LinearFilter,
  Sprite,
  SpriteMaterial,
} from "three";

export interface TextLabelOptions {
  text: string;
  color?: ColorRepresentation;
  background?: string;
  fontSize?: number;
  padding?: number;
  height?: number;
  textureWidth?: number;
  textureHeight?: number;
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
  private textureWidth: number;
  private textureHeight: number;

  constructor(opts: TextLabelOptions) {
    this.textureWidth = opts.textureWidth ?? 1024;
    this.textureHeight = opts.textureHeight ?? 128;
    this.canvas.width = this.textureWidth;
    this.canvas.height = this.textureHeight;

    const context = this.canvas.getContext("2d");
    if (!context) throw new Error("TextLabel: 2D canvas context unavailable");

    this.context = context;
    this.color = opts.color ?? 0xffffff;
    this.background = opts.background ?? "rgba(0, 0, 0, 0.45)";
    this.fontSize = opts.fontSize ?? 48;
    this.padding = opts.padding ?? 12;
    this.height = opts.height ?? 0.08;

    this.texture = new CanvasTexture(this.canvas);
    this.texture.minFilter = LinearFilter;
    this.texture.magFilter = LinearFilter;

    this.material = new SpriteMaterial({
      map: this.texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this.object3d = new Sprite(this.material);
    this.object3d.frustumCulled = false;
    this.setText(opts.text);
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
    const width = this.textureWidth;
    const height = this.textureHeight;
    const font = `${this.fontSize}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;

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
      width - 2 * this.padding,
    );

    this.texture.needsUpdate = true;
    this.object3d.scale.set((width / height) * this.height, this.height, 1);
  }

  private cssColor(color: ColorRepresentation): string {
    const c = new Color(color);
    return `rgb(${Math.round(c.r * 255)}, ${Math.round(c.g * 255)}, ${Math.round(
      c.b * 255,
    )})`;
  }
}
