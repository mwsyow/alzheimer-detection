# Pretrained Model Recommendations

Research date: 2026-08-23

This project classifies single-channel, three-dimensional T1-weighted OASIS MRI
volumes. The recommended pretrained-model evaluation order is:

1. **MedicalNet ResNet10**: first follow-up because it is a 3D medical-imaging
   model, is close to the existing ResNet10 architecture, and is directly supported
   by MONAI. Verification confirmed a one-channel `Conv3d` backbone and a finite
   forward pass for a `[1, 1, 96, 128, 96]` MRI tensor.
2. **3D-Neuro-SimCLR ResNet18**: strongest practical domain-specific follow-up.
   It was pretrained with SimCLR on 44,958 structural brain MRI scans. Port its
   encoder/checkpoint handling rather than assuming its weights match MONAI's
   ResNet18.
3. **BrainIAC base SSL encoder**: highly relevant 3D brain-MRI foundation model,
   but expensive and subject to possible OASIS overlap. Never use its released MCI
   downstream checkpoint for evaluation on this OASIS project because that model
   was trained/evaluated on OASIS-1.
4. **AnatCL**: brain-specific and MIT-licensed, but expects CAT12 VBM volumes of
   shape 121x128x121, so it requires a substantially different preprocessing path.
5. **SwinBrain / 3DINO**: lower-priority research comparisons. SwinBrain expects
   combined T1/T2/FLAIR channels, while 3DINO has high compute/integration cost and
   restrictive non-commercial/no-derivatives licensing.

Lower-priority generic alternatives such as Kinetics-pretrained R3D-18 are useful
only as transfer-learning baselines because their pretraining domain is RGB video,
not medical MRI.

## Existing checkpoint and loader notes

- `pretrained/DenseNet121/86_acc_model.pth` is structurally compatible with the
  configured MONAI DenseNet121: 725 of 727 checkpoint tensors match, with only the
  two classifier tensors excluded.
- The checkpoint's source and pretraining dataset should be documented before its
  results are reported.
- `PretrainedMixin.load_pretrained_weights` currently assumes a raw state dict. It
  should eventually unwrap common keys such as `state_dict` and
  `model_state_dict`, strip prefixes such as `module.`, report the number of loaded
  tensors, and fail if unexpectedly few tensors match.
- Compare head-only training (`freeze_backbone: true`) with full fine-tuning. Frozen
  backbones, including BatchNorm, must remain in evaluation mode.
- Before reporting an unbiased OASIS test result, audit every pretrained model for
  OASIS subject overlap, even when pretraining was self-supervised.

## Verified MedicalNet ResNet10 integration

- Official Hub source: `TencentMedicalNet/MedicalNet-Resnet10`, file
  `resnet_10_23dataset.pth`, pinned revision
  `2a0c8cd91b82beb69610b60cb76d9eb8cbf9eac7`.
- The 57,456,599-byte checkpoint SHA-256 is
  `afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.
- It contains 72 backbone tensors and no classifier. Compatibility requires 3D,
  one input channel, shortcut B, `widen_factor=1.0`, and bias-free downsampling.
- HPC jobs use Hugging Face directly. `huggingface-hub` is locked as a runtime
  dependency and `HF_HOME` is placed in Condor job-local scratch to avoid shared-NFS
  cache races. The repository is public and needs no `HF_TOKEN`.

## Sources

- MedicalNet: https://github.com/Tencent/MedicalNet
- 3D-Neuro-SimCLR: https://github.com/emilykaczmarek/3D-Neuro-SimCLR
- BrainIAC: https://github.com/AIM-KannLab/BrainIAC
- AnatCL: https://github.com/EIDOSLAB/AnatCL
- SwinBrain: https://github.com/MAI-Lab-West-China-Hospital/SwinBrain
- 3DINO: https://github.com/AICONSlab/3DINO
