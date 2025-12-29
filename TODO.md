# Face Recognition Demo TODO - CCTV Dataset Integration

This TODO list outlines the best datasets for testing your face recognition system with CCTV-style images. Focus on i-LIDS and CUHK for the most relevant surveillance scenarios. Complete these steps to enhance your demo.

## 1. Select Primary Dataset
- [ ] Choose **i-LIDS** as the top priority (most CCTV-specific).
- [ ] As backup, select **CUHK Face Dataset** for additional surveillance variations.
- [ ] Optionally, add **LFW** for broader testing with natural face variations.

## 2. Download Datasets
- [ ] **i-LIDS**:
  - Visit [AVSS Dataset Page](http://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html).
  - Register if required and download the dataset (small size, focused on surveillance).
  - Extract and organize into folders (e.g., by person or camera view).
- [ ] **CUHK Face Dataset**:
  - Go to [CUHK Vision Lab](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
  - Download the surveillance subset (look for "CUHK Face Sketch" or similar).
  - Extract and sort images for testing.
- [ ] **LFW (Optional)**:
  - Download from [LFW Website](http://vis-www.cs.umass.edu/lfw/).
  - Use subsets with lower-quality images to simulate CCTV.

## 3. Prepare Test Data
- [ ] Create a `test_cctv/` folder in your project root.
- [ ] Organize downloaded images into subfolders (e.g., by person if labeled).
- [ ] Resize images to CCTV resolution (e.g., 640x480) for realism.
- [ ] Include a mix of known (from your dataset) and unknown faces for comprehensive testing.

## 4. Integrate with Demo
- [ ] Update your Streamlit app (`app.py`) to include a "CCTV Test" section for uploading test images.
- [ ] Modify search logic to handle multiple faces or low-quality inputs if needed.
- [ ] Test recognition: Upload CCTV images and verify ID returns with similarity scores.

## 5. Demo Enhancements
- [ ] Add preprocessing (e.g., brightness adjustment) for poor-quality images.
- [ ] Display results with visuals: Query image vs. matched dataset image.
- [ ] Measure and display accuracy metrics (e.g., correct matches vs. total tests).
- [ ] Prepare edge case tests: Unknown faces, group photos, occlusions.

## 6. Final Checks
- [ ] Ensure all datasets are used ethically (demo/research only).
- [ ] Run full demo flow and note any issues for improvements.
- [ ] Update README.md with dataset details and demo instructions.

## Notes
- Start with i-LIDS for quick wins – it's directly applicable to CCTV.
- If downloads fail, check academic mirrors or Kaggle alternatives.
- Track progress here and mark tasks as completed.