Here we finetune Qwen3-0.6B using reasoning data from "unsloth/OpenMathReasoning-mini". (also you can use non-reasoning data from "mlabonne/FineTome-100k") 

1.```git clone https://github.com/fuyuantan/qwen3-finetune```<br>

2.```python qwen3_finetune.py```<br>

3.When training finished, **adapter_config.json** saved in **/outputs/checkpoint-30/** (Or the directory where your actual model is saved)<br>

4.Config your **adapter_path = "./outputs/checkpoint-30"** in **qwen3_inference.py**<br>

5.```python qwen3_inference.py```<br>

Training Finished:
![微信截图_20250505163352](https://github.com/user-attachments/assets/e3988a18-1161-4a94-a3a9-80deca8234bb)

Saved in:<br>
![微信截图_20250505172600](https://github.com/user-attachments/assets/92963d97-1f8a-496e-8371-ca9c9f146047)


Inference:
![推理](https://github.com/user-attachments/assets/e584eeae-bafc-457d-b7ea-2d25d57cfd2a)

Thank your Stars!
