# External blocker

## Public Streamlit deployment verification

Local application, model checkpoint, locked enhancement parameters, Dockerfile and deployment guide are complete. A public deployment is **not** claimed because this workspace has no user-authorized hosting account/repository action.

Evidence still required to close this gate:

1. public HTTPS URL;
2. deployed commit identifier;
3. deployed model SHA-256 matching `models/model_metadata.json`;
4. UTC verification time;
5. screenshot from a signed-out/incognito browser session;
6. successful single-image and batch smoke tests on the public URL.

Until those items exist, canonical status remains `DEPLOY_READY_BUT_NOT_DEPLOYED` and strict `--require-final` is expected to fail only this external gate.
