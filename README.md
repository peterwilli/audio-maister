# 🩺🎧 Fix all your podcast, video or live stream audio! 🎧🩺

## 👋 Intro

AudiomAIster is a fork of [VoiceFixer](https://github.com/haoheliu/voicefixer) with the following differences:

 - Focus on general purpose audio (where VoiceFixer focusses on voice alone, AudiomAIster will reduce noise while enhancing voice and sound effects, making it more suitable for talks and videos where more than just voices exist).
 - Pre-processing the slow parts of the original dataset to improve performance during training.
 - Training and inference from a single package.
 - Upgraded model to Lightning AI 2.0 v.s Pytorch Lightning 1.8.
 - Open sourced model and dataset on HuggingFace for easier inference.

Thanks to an improved augmentation and training loop, as well as a new dataset, I observed better performance than the original on my own videos with 100 epochs v.s 5000 and only 8% of the original dataset.

[Try it out now on 🤗 Spaces!](https://huggingface.co/spaces/Raaniel/Audiomaister)

<details>
<summary>⚙️ Run locally</summary>

## Requirements

 - ~~Basic knowledge of a (Linux) terminal~~
 - Working git, python3 and pytorch installation (other dependencies automatically are installed)

## Inference

- Install the cli tool: `pip install git+https://github.com/peterwilli/audio-maister.git`
- To restore an audio file called "input.wav" and save it to "fixed.wav", run `audiomaister --input_file input.wav --output_file fixed.wav`
- That's it! You're ready to go. For GPU acceleration, you can append `--accelerator=cuda` to the `audiomaister` command.
</details>

<details>
<summary>📓 Changelog</summary>

## V1.5

 - Improved removal of distortion
 - Makes voices sound less like a robot in some cases
 - Drops low-volume audio less likely

## V1

 - First model
</details>

<details>
<summary>🗃️ Dataset</summary>

The training dataset is largely self-made, and can be found on [HuggingFace](https://huggingface.co/datasets/peterwilli/audio-maister).
</details>

<details>
<summary>🙋 Why make this model?</summary>

I was doing a live stream in where I unboxed a new 3D printer. I was very happy with it, and wanted to edit it into a video to later upload on YouTube.

To my shock, when looking back the raw footage, the audio was ruined beyond repair. It was my first time using Twitch on Android so I guess that's where it went wrong.

Desperately looking for a way to fix my audio, I encountered the closest free model available: VoiceFixer. While it did manage to fix my voice in some areas, it completely erased effects like me opening a box, which set part of the vibe of the video. This likely is the noise canceling.

I decided this is the way to go, and started making small changes, with each change getting better results. Eventually, I decided to fork and train my own model.

Either way, I did fix the audio in the end. For those curious, [the video is here](https://www.youtube.com/watch?v=c5HmXQuj-WY).

</details>

## Credit where credit is due

- [VoiceFixer](https://github.com/haoheliu/voicefixer) for their great model and training framework, without them it would've taken a lot longer! 

## Support, sponsorship and thanks

Are you looking to make a positive impact and get some awesome perks in the process? **[Join me on Patreon!](https://www.patreon.com/emerald_show)** For just $3 per month, you can join our Patreon community and help a creative mind in the Netherlands bring their ideas to life.

Not only will you get the satisfaction of supporting an individual's passions, but you'll also receive a 50% discount on any paid services that result from the projects you sponsor. Plus, as a Patreon member, you'll have exclusive voting rights on new features and the opportunity to shape the direction of future projects. Don't miss out on this chance to make a difference and get some amazing benefits in return.

## To do

- [ ] Releasing training code + tutorial
- [x] HF space! (Thanks to @adolinska!)
- [ ] Colab demo 
