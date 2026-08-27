# Finalization checklist

## Local evidence gates

- [x] Canonical identity contains three named members, instructor and group.
- [x] All three source versions have file-level inventories and component scorecards.
- [x] Raw audit covers 3.670 files with decode status, EXIF, original mode/format and SHA-256.
- [x] Split is 2.571/549/550 with zero path/hash overlap.
- [x] Surrogate artifacts and surrogate metrics are excluded.
- [x] Real MobileNetV2 checkpoint is persisted with SHA-256 metadata.
- [x] Training history records both stages, learning rate and epoch duration.
- [x] Enhancement parameters are selected only on Validation and locked before Test.
- [ ] Full results contain 49 conditions, 26.950 predictions, 245 per-class rows and paired-statistics rows (run later on user GPU).
- [ ] Final error analysis is derived from stored full predictions and is traceable per image.
- [x] Notebook executes sequentially in checkpoint-ready mode; full-run gate is conditional.
- [x] Streamlit supports single image and batch up to 20 images.
- [x] Word, PDF, PowerPoint and Excel use the selected best original (Version B); structural inspections are retained.
- [x] Tests, core validator and checkpoint-ready coherence audit PASS.
- [ ] Full-run validator PASS after user GPU execution.
- [x] Raw data, caches, junctions and transient QA outputs are excluded from the ZIP.

## External deployment gate

- [ ] Public Streamlit HTTPS URL exists.
- [ ] Deployed commit and model SHA-256 are recorded.
- [ ] Signed-out/incognito screenshot and single/batch smoke evidence exist.

Canonical status is `TRAINED_CHECKPOINT_READY — FULL_49_PENDING_USER_GPU — DEPLOY_PENDING_USER_ACTION`. No strict `SUBMISSION_READY` claim is made before the remaining items are verified.
