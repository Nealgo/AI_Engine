# AI_Engine

## DeformConvTranspose2d 可视化说明

项目中已提供可视化脚本 [scripts/visualize_deform.py](scripts/visualize_deform.py)，用于拆解 `DeformConvTranspose2d` 的中间过程，帮助理解“普通转置卷积 -> 偏移采样 -> 残差叠加”的变化。

### 运行方式

```bash
python scripts/visualize_deform.py \
	--image datas/archive/gt/gt_test/0_clean.png \
	--output_dir results/deform_vis_test \
	--device cpu \
	--in_channels 3 \
	--out_channels 3 \
	--stride 1 \
	--padding 1 \
	--output_padding 0
```

### 输出图像与含义（对应你看到的那些图）

1. `base_sampling_grid.png`
含义：基础采样网格（未加偏移）。
怎么看：规则直线网格，作为对照基准。

2. `warped_sampling_grid.png`
含义：加入 offset 后的采样网格。
怎么看：网格线出现弯曲/错位，弯曲越明显说明局部偏移越大。

3. `offset_x_color.png`
含义：偏移场 x 分量（水平位移，带正负方向）。
怎么看：不同颜色表示正负方向，颜色越“极端”表示水平偏移绝对值越大。

4. `offset_y_color.png`
含义：偏移场 y 分量（垂直位移，带正负方向）。
怎么看：与 `offset_x_color.png` 类似，但反映纵向偏移。

5. `offset_magnitude_color.png`
含义：偏移幅值图，只看位移强度，不区分方向。
计算：$\sqrt{dx^2 + dy^2}$
怎么看：亮色区域表示偏移更强，通常集中在边缘和纹理较复杂区域。

6. `deconv_output.png`
含义：普通转置卷积输出（还未进行偏移采样）。
用途：作为后续对比基线。

7. `offset_sampled_output.png`
含义：将 `deconv_output` 按偏移网格进行 `grid_sample` 后的输出。
用途：反映“仅偏移采样”带来的变化。

8. `final_residual_output.png`
含义：最终输出（偏移采样结果 + 转置卷积残差）。
关系：`final = sampled + deconv`
用途：反映模块真正输出特征。

9. `diff_sample_vs_deconv_gray.png` / `diff_sample_vs_deconv_overlay.png`
含义：偏移采样前后差异图。
计算：$|sampled - deconv|$
怎么看：亮区表示偏移采样改动较大；overlay 版本会叠回基准图，更容易定位变化位置。

10. `diff_final_vs_deconv_gray.png` / `diff_final_vs_deconv_overlay.png`
含义：最终输出与普通转置卷积的差异图。
计算：$|final - deconv|$
怎么看：用于判断残差叠加后哪些区域被增强最明显。

### 这批图对比的核心问题

1. 偏移场是否真的在“动网格”
对比 `base_sampling_grid.png` 与 `warped_sampling_grid.png`。

2. 偏移主要在哪些位置、哪个方向更强
对比 `offset_x_color.png`、`offset_y_color.png`、`offset_magnitude_color.png`。

3. 偏移采样本身改变了多少
看 `diff_sample_vs_deconv_overlay.png`。

4. 加上残差后最终增强了多少
看 `diff_final_vs_deconv_overlay.png`。

### 解读建议

1. 如果 `warped_sampling_grid.png` 几乎是规则直线，说明 offset 学得较弱。
2. 如果 `offset_magnitude_color.png` 全局过亮，可能偏移过大，需关注训练稳定性。
3. 如果 `diff_sample_vs_deconv` 很小但 `diff_final_vs_deconv` 很大，说明主要增强来自残差叠加而非采样形变。
4. 如果高亮主要集中在边缘和纹理区，通常是合理现象。
