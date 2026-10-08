## 5. Sanity-check on real Google Maps crops (Rizky's feedback)
Crop 15-20 SATELLITE (top-down) images, one building each, mostly normal houses, into `maps_crops/`.
```bash
python3 test_maps_imagery.py --images maps_crops \
    --model checkpoints/C2_resnet50/best.pt checkpoints/C1_resnet18/best.pt
open maps_review/review_gallery.html
```
Mark each crop OK / False alarm / Missed damage, then export `my_verdicts.csv`. No accuracy is computed: there is no ground truth.

## 6. Register the model to serve
```bash
python3 promote_serving_alias.py --list      # see runs, versions, aliases
python3 promote_serving_alias.py --dry-run
python3 promote_serving_alias.py             # C1_resnet18 -> alias 'serving'
```
`champion` stays on ResNet-50. `serving` points to ResNet-18.

## 7. Serve the API
```bash
pip install -r requirements-serve.txt
uvicorn serve_api:app --port 8000            # loads models:/property-dd-damage-cnn@serving
curl -s localhost:8000/health
curl -F "file=@house.jpg" localhost:8000/classify
```
Override with `CHECKPOINT=checkpoints/C1_resnet18/best.pt` or `MODEL_URI=models:/...@champion`.
