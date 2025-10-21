# indomain
python test.py \
  --tasks vit \
  --op_flag all_data \
  --reduce_rate "" \
  --output_dir ./results/indomain_result \
  --dino_flag dino \
  --domain indomain

# opendomain
python test.py \
  --tasks vit \
  --op_flag opendomain \
  --reduce_rate "" \
  --output_dir ./results/opendomain_result \
  --dino_flag dino \
  --domain opendomain