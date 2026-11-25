# Email to the TI2Net authors (draft — you send it)

**To:** baoping.liu@student.uts.edu.au
**Cc (optional, in case the student address is no longer active):** Bo.liu@uts.edu.au, Xin.Yu@uts.edu.au
**Subject:** TI2Net (WACV 2023): released FF++ ID vectors are missing from the Google Drive folder

Dear Dr. Liu,

I am [YOUR NAME], a [ROLE] at [INSTITUTION]. We are reproducing TI2Net as the baseline for our work on
temporal deepfake detection, using your official repository (https://github.com/BaopingLiu/TI2Net).

The README says the extracted FF++ identity vectors are available in the Google Drive folder
`id_vectors`. Today that folder contains only two files, `real_lens.npy` and `fake_lens.npy`
(1,000 and 4,000 videos). The identity vector files themselves (`real_ids.npy` / `fake_ids.npy`, or the
per-method `ffpp_*_ids.npy` files) are not there.

Could you please:

1. Upload or share the missing FF++ identity vector files?
2. If possible, share a trained TI2Net checkpoint (listed as a to-do in the README)?

Three short questions would also help us match your setup exactly:

3. In `demo_offline_id.ipynb`, frames are read with `glob(video + "/*")` without sorting. Were the
   frames in temporal order when you extracted the released vectors?
4. The notebook reads from `.../raw/aligned_faces`, but `video2image_cropping.py` writes to `images`.
   Was there an extra alignment step between cropping and ArcFace, and if so, which one?
5. When a frame contains more than one detected face, which face was kept?

We will cite TI2Net and your repository in our work. Thank you very much for your time.

Best regards,
[YOUR NAME]
[INSTITUTION / EMAIL]
