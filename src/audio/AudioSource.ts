export interface AudioSourceBundle {
  context: AudioContext;
  source: MediaStreamAudioSourceNode;
  stream: MediaStream;
}

export async function createMicSource(): Promise<AudioSourceBundle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });

  const context = new AudioContext();
  if (context.state === "suspended") await context.resume();
  const source = context.createMediaStreamSource(stream);

  return { context, source, stream };
}
