import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 定义 CSV 文件及其对应的标题
# 请确保这些 CSV 文件与脚本在同一目录下，或在此处提供正确的文件路径
csv_files_info = [
    {"file": "./pronoun_probabilities_Qwen_7B.csv", "title": "(a) Base Model (Qwen2.5-7B Instruct)"},
    {"file": "./pronoun_probabilities_prompt_engineer.csv", "title": "(b) Prompt Engineered Model"},
    {"file": "./pronoun_probabilities_lora_winobias.csv", "title": "(c) Winobias Anti-stereotypical DataSet Fine-tuned Model"},
    {"file": "./pronoun_probabilities_lora_common.csv", "title": "(d) Custom Balanced Dataset Fine-tuned Model"}
]

# 创建一个 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

for i, csv_info in enumerate(csv_files_info):
    csv_file = csv_info["file"]
    plot_title = csv_info["title"]
    ax = axes[i]

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file}' 未找到。请提供正确的文件路径。")
        ax.text(0.5, 0.5, f"文件未找到:\n{csv_file}", ha='center', va='center', fontsize=10, color='red')
        ax.set_title(plot_title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    means = [df['male_prob'].mean(), df['female_prob'].mean()]
    stds = [df['male_prob'].std(), df['female_prob'].std()]
    
    if df['male_prob'].isnull().all() or len(df['male_prob'].dropna()) == 0:
        male_quartiles = [np.nan, np.nan, np.nan] 
    else:
        male_quartiles = [np.percentile(df['male_prob'].dropna(), 25),
                          np.percentile(df['male_prob'].dropna(), 50),
                          np.percentile(df['male_prob'].dropna(), 75)]
    if df['female_prob'].isnull().all() or len(df['female_prob'].dropna()) == 0:
        female_quartiles = [np.nan, np.nan, np.nan]
    else:
        female_quartiles = [np.percentile(df['female_prob'].dropna(), 25),
                            np.percentile(df['female_prob'].dropna(), 50),
                            np.percentile(df['female_prob'].dropna(), 75)]
    categories = ['Male', 'Female']

    bars = ax.bar(categories, means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral'], edgecolor='black')
    ax.set_xlabel('Gender', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title(plot_title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=8)

    for j, (bar, quartiles) in enumerate(zip(bars, [male_quartiles, female_quartiles])):
        x_coord = bar.get_x() + bar.get_width() / 2

        if not np.isnan(quartiles[0]):
            ax.plot(x_coord, quartiles[0], marker='o', color='blue', markersize=6, label='25th Percentile' if i == 0 and j == 0 else "")
        if not np.isnan(quartiles[1]):
            ax.plot(x_coord, quartiles[1], marker='o', color='green', markersize=6, label='50th Percentile (Median)' if i == 0 and j == 0 else "")
        if not np.isnan(quartiles[2]):
            ax.plot(x_coord, quartiles[2], marker='o', color='red', markersize=6, label='75th Percentile' if i == 0 and j == 0 else "")

        # --- 修改后的文本定位逻辑 ---
        font_size = 10
        text_content_mean = f'Mean: {means[j]:.3f}' if not np.isnan(means[j]) else 'Mean: N/A'
        text_content_std = f'Std: {stds[j]:.3f}' if not np.isnan(stds[j]) else 'Std: N/A'
        text_content = f'{text_content_mean}\n{text_content_std}'

        TEXT_HEIGHT_ESTIMATE = 0.07 # 文本块在数据单位中的估计高度 (针对两行文字)
        PADDING_ABOVE = 0.01       # 在误差棒或均值棒上方的小间距
        PADDING_FROM_PLOT_EDGE = 0.005 # 如果文本置于顶部，与绘图区域顶边的间距

        y_mean_val = means[j]
        y_std_val = stds[j]

        if np.isnan(y_mean_val):
            continue # 如果均值为NaN，则跳过此柱的文本

        # 参考 Y 坐标是误差棒的顶部，如果没有标准差，则是均值棒的顶部
        y_ref_for_text = y_mean_val + y_std_val if not np.isnan(y_std_val) else y_mean_val
        
        if np.isnan(y_ref_for_text): # 如果 y_ref_for_text 仍然是 NaN (例如均值有效但 std 是奇异的 NaN)
            y_ref_for_text = y_mean_val # 退回到仅使用均值

        # 默认位置：文本底部在 y_ref_for_text 上方 PADDING_ABOVE 处
        text_y = y_ref_for_text + PADDING_ABOVE
        va = 'bottom'

        # 如果此默认位置导致文本超出图表顶部 (ylim 为 0 到 1.0)
        if text_y + TEXT_HEIGHT_ESTIMATE > 1.0:
            # 回退方案：将文本顶部放置在图表顶部 (1.0) 向下 PADDING_FROM_PLOT_EDGE 处
            text_y = 1.0 - PADDING_FROM_PLOT_EDGE
            va = 'top'
            # 这种回退可能会导致文本与非常高的误差棒重叠。
            # 主要目标是“在柱状图上方”和“在图片内”。
            # 如果柱子+误差非常高，将文本放在绝对顶部边缘似乎是一个合理的折衷方案。

        # 再次检查 y_ref_for_text 是否为 NaN (理论上如果 mean_val 不是 NaN 则不应发生)
        if np.isnan(y_ref_for_text): 
            text_y = 0.5 # 绝对回退值
            va = 'center'
        
        # 防止文本被放置在 y=0 以下 (对概率不典型，但是好的做法)
        # 同时处理 y_ref_for_text 可能为负的情况 (虽然概率图中不太可能)
        min_y_text_bottom = PADDING_ABOVE
        min_y_text_top = TEXT_HEIGHT_ESTIMATE + PADDING_ABOVE

        if va == 'bottom' and text_y < min_y_text_bottom :
             # 如果文本底部低于允许的最低点，则上移
             text_y = min_y_text_bottom
             # 如果上移后超出顶部，则采用顶部对齐方式置于顶部
             if text_y + TEXT_HEIGHT_ESTIMATE > 1.0:
                 text_y = 1.0 - PADDING_FROM_PLOT_EDGE
                 va = 'top'

        elif va == 'top' and text_y - TEXT_HEIGHT_ESTIMATE < 0: # 如果文本顶部计算后导致文本底部低于0
            # 如果文本顶部低于允许的最低点(确保文本完整显示在0以上)，则上移
            text_y = min_y_text_top
            # 如果上移后超出顶部，则固定在顶部 (这种情况比较极端)
            if text_y > 1.0: # text_y 是文本的顶部
                 text_y = 1.0 - PADDING_FROM_PLOT_EDGE


        ax.text(x_coord, text_y, text_content, ha='center', va=va, fontsize=font_size,
                bbox=dict(facecolor='white', alpha=0.2, pad=0.1)) # bbox alpha 设为0.2
        # --- 文本定位逻辑结束 ---

    print(f"\n文件 {csv_file} 的统计信息:")
    print(f"  男性概率: Mean = {means[0]:.4f}, Std = {stds[0]:.4f}, Q1 = {male_quartiles[0]:.4f}, Median = {male_quartiles[1]:.4f}, Q3 = {male_quartiles[2]:.4f}")
    print(f"  女性概率: Mean = {means[1]:.4f}, Std = {stds[1]:.4f}, Q1 = {female_quartiles[0]:.4f}, Median = {female_quartiles[1]:.4f}, Q3 = {female_quartiles[2]:.4f}")

handles, labels = [], []
for ax_item in axes:
    h, l = ax_item.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

if handles and labels:
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9)

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

output_filename = 'combined_pronoun_probabilities.png'
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"\n图形已保存为 '{output_filename}'")