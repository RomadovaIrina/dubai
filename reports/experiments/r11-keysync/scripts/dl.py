from huggingface_hub import hf_hub_download
for f in ["WavLM-Base+.pt","keyframe_dub.pt","interpolation_dub.pt"]:
    print(hf_hub_download("toninio19/keysync", "checkpoints/"+f, local_dir="/workspace/dub/keysync/code/pretrained_models"), flush=True)
print("DL_DONE", flush=True)
