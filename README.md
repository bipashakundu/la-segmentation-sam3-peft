# Parameter-Efficient Adaptation of SAM3 for Left Atrium Segmentation

Code for the two parameter-efficient fine-tuning (PEFT) strategies evaluated in the paper:
LoRA and a prompt-free SAM3-Adapter, applied to left atrium segmentation from cardiac
LGE-MRI (LAScarQS 2022, LASC 2018).

## Structure

- `SAM3-LoRA/` — low-rank adaptation of SAM3's linear projection layers (vision encoder,
  text encoder, DETR encoder/decoder, mask decoder). See `SAM3-LoRA/README.md`.
- `SAM3-Adapter/` — prompt-free adapter modules conditioning the frozen SAM3 vision
  encoder on handcrafted high-frequency + learned patch-embedding features. See
  `SAM3-Adapter/README.md`.

## Dependencies not included in this release

- **SAM3 base model**: both methods build on Meta's official SAM3
  (`facebook/sam3`). Install/clone it separately per its own repository and license,
  then point the config paths (`sam_checkpoint`, model cache dir, etc.) at your local copy.
- **Datasets**: LAScarQS 2022 and LASC 2018 are third-party clinical MRI datasets and are
  not redistributed here; obtain them from the original challenge organizers and update the
  `data_dir` / `root_path_*` fields in the config files.
- **Pretrained/checkpoint weights and raw run logs** are excluded — only source code and
  hyperparameter configs are included.

## Notes

All file paths in configs and scripts have been converted to relative paths or
`<PLACEHOLDER>` markers for anonymity; substitute your own paths before running.
