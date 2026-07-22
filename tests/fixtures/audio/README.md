# LibriSpeech ASR fixtures

These two unmodified FLAC files are LibriSpeech `dev-clean` utterances from
speaker 1272, chapter 128104. LibriSpeech was derived from LibriVox
public-domain audiobook recordings and is distributed under the Creative
Commons Attribution 4.0 International license. The complete legal code is
included as `LICENSE-CC-BY-4.0.txt`.

OpenSLR SLR12 is the authoritative LibriSpeech upstream:

- Corpus page: https://openslr.org/12/
- Corpus information: https://www.openslr.org/resources/12/README.TXT

Only the two individual utterances needed by the tests are checked in; no
LibriSpeech archive is redistributed. The pinned single-file mirrors below are
byte-identical to the corresponding OpenSLR SLR12 `dev-clean` files.

## `librispeech_1272-128104-0000.flac`

- Utterance ID: `1272-128104-0000`
- Format: FLAC, mono, 16 kHz
- Duration: 5.855 seconds
- Size: 120041 bytes
- SHA-256: `4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41`
- Pinned mirror: https://raw.githubusercontent.com/QwenLM/Qwen-Audio/b50fb958438081d36e1a14e93dbbc2f329c7f10e/assets/audio/1272-128104-0000.flac
- Transcript: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`

## `librispeech_1272-128104-0001.flac`

- Utterance ID: `1272-128104-0001`
- Format: FLAC, mono, 16 kHz
- Duration: 4.815 seconds
- Size: 101672 bytes
- SHA-256: `46a9d58622b4675b29564da2d9ba73e702241c5fa969f12c387cad4aa984276a`
- Pinned mirror: https://huggingface.co/patrickvonplaten/LibriSpeechTest/resolve/994387b7bba79ba3daeaf3089c971af3a4e06dce/dev-clean/1272/128104/1272-128104-0001.flac?download=true
- Transcript: `NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER`

The transcript sidecars contain the exact expected lexical text used by the
runtime tests. No audio samples were modified, resampled, or normalized.
