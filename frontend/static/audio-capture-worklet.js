// AudioWorklet processor: microphone capture -> PCM16.
//
// Runs on the audio render thread. Each 128-sample render quantum of Float32
// mono input is converted to little-endian 16-bit PCM and posted to the main
// thread, which forwards it over the /ws/audio WebSocket. We also post a
// coarse peak level so the UI can show a recording indicator without the main
// thread touching raw audio.
//
// Note: this captures at the AudioContext's native sample rate. The main
// thread tells the backend that rate in the `hello` handshake; backend-side
// resampling to the pipeline rate is a phase-two concern (loopback doesn't
// care about the rate).

class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    const pcm = new Int16Array(channel.length);
    let peak = 0;
    for (let i = 0; i < channel.length; i++) {
      let s = channel[i];
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      const a = s < 0 ? -s : s;
      if (a > peak) peak = a;
    }

    // Transfer the buffer (zero-copy) plus the peak level for the meter.
    this.port.postMessage({ pcm: pcm.buffer, peak }, [pcm.buffer]);
    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
