# GitHub issue (draft — you post it)

Post at: https://github.com/BaopingLiu/TI2Net/issues/new

**Title:** Released FF++ ID vectors missing from Google Drive (only `*_lens.npy` present)

**Body:**

Hi, thanks for releasing the code.

The README says the extracted FF++ ID vectors are in the Google Drive folder `id_vectors`
(https://drive.google.com/drive/folders/1uEBSBGyjZC4DiseYiCRo4ULSauvb7wyw). The folder currently
contains only:

- `real_lens.npy` (1,000 videos)
- `fake_lens.npy` (4,000 videos)

The ID vector files (`real_ids.npy` / `fake_ids.npy`, or `ffpp_{real,df,ff,fs,nt}_ids.npy`) are not there.

Could you re-upload them? Three related questions so we can reproduce the extraction exactly:

1. `video_id_extraction` uses `glob(video + "/*")` without sorting. Were frames in temporal order for the
   released vectors?
2. The notebook reads `.../raw/aligned_faces`, while `utils/video2image_cropping.py` writes `images`.
   Is there an alignment step that is not in the repo?
3. When several faces are detected in a frame, which crop is used?

Thanks!
