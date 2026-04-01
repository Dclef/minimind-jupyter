import json

with open('minimind_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 找到 SFT 初始化单元格并更新
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if '初始化 SFT 环境' in source and 'HuggingFace' in source:
            cell['source'] = [
                "# 初始化 SFT 环境\n",
                "setup_seed(42)\n",
                "\n",
                "# 创建模型配置\n",
                "sft_lm_config = MiniMindConfig(\n",
                "    hidden_size=sft_config['hidden_size'],\n",
                "    num_hidden_layers=sft_config['num_hidden_layers'],\n",
                "    use_moe=sft_config['use_moe']\n",
                ")\n",
                "\n",
                "# 加载 Tokenizer\n",
                "sft_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)\n",
                "\n",
                "# 创建模型并加载预训练权重\n",
                "sft_model = MiniMindForCausalLM(sft_lm_config)\n",
                "\n",
                "# 检查预训练权重是否存在\n",
                "pretrain_weight_path = f\"{OUTPUT_DIR}/{sft_config['from_weight']}_{sft_config['hidden_size']}.pth\"\n",
                "if os.path.exists(pretrain_weight_path):\n",
                "    weights = torch.load(pretrain_weight_path, map_location=device)\n",
                "    sft_model.load_state_dict(weights, strict=False)\n",
                "    Logger(f\"已加载预训练权重: {pretrain_weight_path}\")\n",
                "else:\n",
                "    Logger(\"⚠️ 本地未找到预训练权重\")\n",
                "    Logger(\"请确保您已运行预训练单元格并保存了权重\")\n",
                "    Logger(f\"预期权重路径: {pretrain_weight_path}\")\n",
                "    Logger(\"\")\n",
                "    Logger(\"如果您想从官方预训练权重开始，请:\")\n",
                "    Logger(\"1. 从 https://huggingface.co/jingyaogong/minimind-3 下载权重\")\n",
                "    Logger(\"2. 将权重文件放到 OUTPUT_DIR 目录\")\n",
                "    Logger(\"3. 或者设置 USE_DRIVE=True 使用 Google Drive 持久化保存\")\n",
                "    Logger(\"\")\n",
                "    Logger(\"将从随机初始化开始 SFT 训练（效果可能较差）\")\n",
                "\n",
                "get_model_params(sft_model, sft_lm_config)\n",
                "sft_model = sft_model.to(device)\n",
                "\n",
                "# 创建 SFT 数据集和数据加载器\n",
                "sft_ds = SFTDataset(sft_config['data_path'], sft_tokenizer, max_length=sft_config['max_seq_len'])\n",
                "sft_loader = DataLoader(sft_ds, batch_size=sft_config['batch_size'], shuffle=True, num_workers=sft_config['num_workers'])\n",
                "\n",
                "# 优化器和混合精度\n",
                "sft_optimizer = optim.AdamW(sft_model.parameters(), lr=sft_config['learning_rate'])\n",
                "sft_scaler = torch.cuda.amp.GradScaler(enabled=True)\n",
                "\n",
                "print(\"\\nSFT 环境初始化完成！\")\n",
                "print(f\"训练样本数: {len(sft_ds)}\")\n",
                "print(f\"每个 epoch 步数: {len(sft_loader)}\")"
            ]
            print(f"已更新 SFT 初始化单元格 {i}")
            break

with open('minimind_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook 已保存")
