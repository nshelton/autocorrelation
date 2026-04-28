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

/**
 * Create an audio source from a captured browser tab.
 *
 * `getDisplayMedia` requires both audio and video to be requested for
 * the picker to even show audio-capable surfaces. We drop the video
 * tracks immediately. The user must pick a tab (or window/screen on
 * platforms that support it) that is currently producing audio.
 */
export async function createTabSource(): Promise<AudioSourceBundle> {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    audio: true,
    video: true,
  });

  const audioTracks = stream.getAudioTracks();
  if (audioTracks.length === 0) {
    // Stop any video tracks we accidentally got
    stream.getTracks().forEach((t) => t.stop());
    throw new Error(
      "Selected source has no audio. Please pick a tab that's playing audio (and check the 'Share tab audio' box).",
    );
  }

  // Drop video tracks; we only want audio.
  stream.getVideoTracks().forEach((t) => {
    t.stop();
    stream.removeTrack(t);
  });

  const context = new AudioContext();
  if (context.state === "suspended") await context.resume();
  const source = context.createMediaStreamSource(stream);

  return { context, source, stream };
}
