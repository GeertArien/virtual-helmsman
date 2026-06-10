// AudioWorklet processor: PCM16 playback queue.
//
// The main thread posts incoming PCM16 frames (received over /ws/audio) as
// transferred ArrayBuffers; this processor converts them to Float32, queues
// them, and emits 128-sample quanta to the output. When the queue underruns
// it emits silence (a gap), which is the right behaviour for a live stream.
//
// Keeping the jitter buffer on the audio thread (rather than scheduling
// AudioBufferSourceNodes from the main thread) avoids clicks between chunks
// and main-thread timing jitter.

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = []; // array of Float32Array
    this._head = 0; // read offset into _queue[0]
    this.port.onmessage = (event) => {
      const buf = event.data && event.data.pcm;
      if (!buf) return;
      const ints = new Int16Array(buf);
      const floats = new Float32Array(ints.length);
      for (let i = 0; i < ints.length; i++) {
        floats[i] = ints[i] / (ints[i] < 0 ? 0x8000 : 0x7fff);
      }
      this._queue.push(floats);
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!out) return true;

    for (let i = 0; i < out.length; i++) {
      if (this._queue.length === 0) {
        out[i] = 0; // underrun -> silence
        continue;
      }
      const chunk = this._queue[0];
      out[i] = chunk[this._head++];
      if (this._head >= chunk.length) {
        this._queue.shift();
        this._head = 0;
      }
    }
    return true;
  }
}

registerProcessor('playback-processor', PlaybackProcessor);
