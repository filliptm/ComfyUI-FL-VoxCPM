# ComfyUI-FL-VoxCPM

ComfyUI nodes for **VoxCPM** text-to-speech with voice cloning and LoRA training support.

[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

## Features

- **VoxCPM 1.5** - 44.1kHz high-fidelity TTS with 800M parameters
- **Voice Cloning** - Clone any voice from a short audio sample
- **LoRA Support** - Load and train custom voice styles
- **Auto Model Download** - Models download automatically on first use

## Installation

**Via ComfyUI Manager:** Search for `ComfyUI-FL-VoxCPM`

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI-FL-VoxCPM.git
cd ComfyUI-FL-VoxCPM
pip install -r requirements.txt
```

## Nodes

| Node | Description |
|------|-------------|
| **FL VoxCPM TTS** | Main TTS node - text to speech with optional voice cloning |
| **FL VoxCPM Train Config** | Training hyperparameters (learning rate, LoRA rank, etc.) |
| **FL VoxCPM Dataset Maker** | Create training dataset from audio + transcript files |
| **FL VoxCPM LoRA Trainer** | Train custom LoRA voice models |

## Quick Start

### Text-to-Speech
1. Add **FL VoxCPM TTS** node
2. Enter text, queue prompt
3. Model downloads automatically on first run

### Voice Cloning
1. Add **FL VoxCPM TTS** + **Load Audio** nodes
2. Connect reference audio to `prompt_audio`
3. Enter exact transcript in `prompt_text`
4. Enter target text, queue prompt

### LoRA Training
1. Prepare folder with `.wav`/`.mp3`/`.flac` files + matching `.txt` transcripts
2. **FL VoxCPM Dataset Maker** → **FL VoxCPM Train Config** → **FL VoxCPM LoRA Trainer**
3. Output saves to `ComfyUI/models/loras/`

## Key Parameters

- **cfg_value** (2.0) - Higher = closer to prompt voice, lower = more natural
- **inference_timesteps** (10) - More steps = better quality, slower
- **lora_name** - Select trained LoRA from dropdown (auto-detects rank)

## Models

| Model | Sample Rate | Notes |
|-------|-------------|-------|
| VoxCPM1.5 | 44.1kHz | Recommended |
| VoxCPM-0.5B | 16kHz | Legacy |

Models auto-download to `ComfyUI/models/tts/VoxCPM/`

## License

[Apache-2.0](LICENSE) - Based on [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
