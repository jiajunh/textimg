import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


from matplotlib.font_manager import FontProperties

times_bold = FontProperties(
    family="Times New Roman",
    weight="bold",
    size=15
)

def panel_label_under_xlabel(ax, label, y=-0.35):
    ax.text(
        0.5, y, label,
        transform=ax.transAxes,
        ha="center", va="top",
        fontproperties=times_bold,
    )


fig = plt.figure(figsize=(11, 6))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0])




# reasoning
r_gpt_low_a_1 = 0.02127659574468085
r_gpt_low_p_1 = 0.1698581560283688
r_gpt_low_a_2 = 0.02
r_gpt_low_p_2 = 0.13516666666666666
r_gpt_low_a_3 = 0.0
r_gpt_low_p_3 = 0.11033333333333334
r_gpt_low_a_4 = 0.01
r_gpt_low_p_4 = 0.11616666666666667
r_gpt_low_a_5 = 0.01
r_gpt_low_p_5 = 0.12283333333333334

r_gpt_2_a_1 = 0.97
r_gpt_2_p_1 = 0.9861666666666666
r_gpt_2_a_2 = 0.89
r_gpt_2_p_2 = 0.9345137085137085
r_gpt_2_a_3 = 0.78
r_gpt_2_p_3 = 0.8962896825396826
r_gpt_2_a_4 = 0.63
r_gpt_2_p_4 = 0.765821067821068
r_gpt_2_a_5 =0.37
r_gpt_2_p_5 = 0.6409787157287157


r_gemini_a_1 = 0.9
r_gemini_p_1 = 0.5869960317460318
r_gemini_a_2 = 0.8571428571428571
r_gemini_p_2 = 0.4040330417881438
r_gemini_a_3 = 0.7684210526315789
r_gemini_p_3 = 0.4270175146490936
r_gemini_a_4 = 0.6419753086419753
r_gemini_p_4 = 0.33120535157572195
r_gemini_a_5 = 0.5507246376811594
r_gemini_p_5 = 0.28691847604891085


r_qwen_a_1 = 0.84
r_qwen_p_1 = 0.6297222222222223
r_qwen_a_2 = 0.82
r_qwen_p_2 = 0.581
r_qwen_a_3 = 0.78
r_qwen_p_3 = 0.6217857142857143
r_qwen_a_4 = 0.55
r_qwen_p_4 = 0.4391428571428571
r_qwen_a_5 = 0.4
r_qwen_p_5 = 0.26445238095238094


r_flux_a_1 = 0.88
r_flux_p_1 = 0.5457261904761905
r_flux_a_2 = 0.74
r_flux_p_2 = 0.49349999999999994
r_flux_a_3 = 0.69
r_flux_p_3 = 0.3675
r_flux_a_4 = 0.46464646464646464
r_flux_p_4 = 0.2801346801346801
r_flux_a_5 = 0.26262626262626265
r_flux_p_5 = 0.19148629148629148


r_llm_a_1 = 0.99
r_llm_p_1 = 1.0
r_llm_a_2 = 0.99
r_llm_p_2 = 0.9795
r_llm_a_3 = 0.95
r_llm_p_3 = 0.9845833333333333
r_llm_a_4 = 0.93
r_llm_p_4 = 0.9618333333333333
r_llm_a_5 = 0.81
r_llm_p_6 = 0.9175409035409036


r_gpt_medium_a_1 = 0.87
r_gpt_medium_p_1 = 0.8336666666666668
r_gpt_medium_a_2 = 0.68
r_gpt_medium_p_2 = 0.7106428571428571
r_gpt_medium_a_3 = 0.55
r_gpt_medium_p_3 = 0.6705714285714285
r_gpt_medium_a_4 = 0.34
r_gpt_medium_p_4 = 0.5255714285714287
r_gpt_medium_a_5 = 0.16
r_gpt_medium_p_5 = 0.3355357142857142


r_llm_a = [0.99, 0.99, 0.95, 0.93, 0.81]
r_llm_p = [1.0, 0.9795, 0.9845833333333333, 0.9618333333333333, 0.9175409035409036]

r_llm_qwen3_a = [0.97, 0.95, 0.93, 0.74, 0.6]
r_llm_qwen3_p = [0.9846666666666667, 0.9598333333333333, 0.9754603174603175, 0.8660833333333332, 0.7990196747696747]


steps = np.arange(1, 6)

# ---- Pack data ----
acc = {
    "GPT-L": [r_gpt_low_a_1, r_gpt_low_a_2, r_gpt_low_a_3, r_gpt_low_a_4, r_gpt_low_a_5],
    "Gemini": [r_gemini_a_1, r_gemini_a_2, r_gemini_a_3, r_gemini_a_4, r_gemini_a_5],
    "Qwen-Img": [r_qwen_a_1, r_qwen_a_2, r_qwen_a_3, r_qwen_a_4, r_qwen_a_5],
    "Flux.2": [r_flux_a_1, r_flux_a_2, r_flux_a_3, r_flux_a_4, r_flux_a_5],
    "GPT-M": [r_gpt_medium_a_1, r_gpt_medium_a_2, r_gpt_medium_a_3, r_gpt_medium_a_4, r_gpt_medium_a_5],
    "GPT-Image-2": [r_gpt_2_a_1, r_gpt_2_a_2, r_gpt_2_a_3, r_gpt_2_a_4, r_gpt_2_a_5],
}

prec = {
    "GPT-L": [r_gpt_low_p_1, r_gpt_low_p_2, r_gpt_low_p_3, r_gpt_low_p_4, r_gpt_low_p_5],
    "Gemini": [r_gemini_p_1, r_gemini_p_2, r_gemini_p_3, r_gemini_p_4, r_gemini_p_5],
    "Qwen-Img": [r_qwen_p_1, r_qwen_p_2, r_qwen_p_3, r_qwen_p_4, r_qwen_p_5],
    "Flux.2": [r_flux_p_1, r_flux_p_2, r_flux_p_3, r_flux_p_4, r_flux_p_5],
    "GPT-M": [r_gpt_medium_p_1, r_gpt_medium_p_2, r_gpt_medium_p_3, r_gpt_medium_p_4, r_gpt_medium_p_5],
    "GPT-Image-2": [r_gpt_2_p_1, r_gpt_2_p_2, r_gpt_2_p_3, r_gpt_2_p_4, r_gpt_2_p_5],
}


markers = {
    "GPT-L": "o",
    "Gemini": "s",
    "Qwen-Img": "^",
    "Flux.2": "D",
    "GPT-M": "X",
    "GPT-Image-2": "P",
}


gs_mid = gs[0, 1].subgridspec(2, 1, hspace=0.2)
ax2a = fig.add_subplot(gs_mid[0, 0])
ax2b = fig.add_subplot(gs_mid[1, 0], sharex=ax2a)

for name, vals in acc.items():
    ax2a.plot(steps, vals, marker=markers[name],
              markersize=10, markerfacecolor="white",
              markeredgewidth=2, linewidth=2.5, label=name)

ax2a.plot(
    steps, r_llm_a,
    linestyle="--",
    linewidth=2.5,
    color="#1C1C1C",
    label="GPT-5.2 (LLM)",
    zorder=2,
    alpha=0.8
)

ax2a.plot(
    steps, r_llm_qwen3_a,
    linestyle="-.",
    linewidth=2.5,
    color="#6A4E42",
    label="Qwen3-8B (LLM)",
    zorder=2,
    alpha=0.8
)

# "#6A4E42", "#2EC4B6"

ax2a.set_ylabel("Answer Score")
ax2a.set_ylim(0.0, 1.1)
ax2a.set_xticks([1, 2, 3, 4, 5])
ax2a.grid(True, linestyle="--", alpha=0.4)
ax2a.set_title("(b) Math Reasoning")


ax2a.legend(
    ncol=2,
    fontsize=9,
    loc="lower left",
    bbox_to_anchor=(0.00, 0.05)
)



for name, vals in prec.items():
    ax2b.plot(steps, vals, marker=markers[name],
              markersize=10, markerfacecolor="white",
              markeredgewidth=2, linewidth=2.5)

ax2b.plot(
    steps, r_llm_p,
    linestyle="--",
    linewidth=2.5,
    color="black",
    label="LLM Baseline",
    zorder=2
)

ax2b.plot(
    steps, r_llm_qwen3_p,
    linestyle="-.",
    linewidth=2.5,
    color="#6A4E42",
    label="Qwen3-8b LLM Baseline",
    zorder=2
)


ax2b.set_ylabel("Process Score")
ax2b.set_xlabel("Difficulty Level")
ax2b.set_ylim(0, 1.1)
ax2a.set_xticks([1, 2, 3, 4, 5])
ax2b.grid(True, linestyle="--", alpha=0.4)



# identical
i_gpt_low_cer_64 = 0.12667133658622778
i_gpt_low_wer_64 = 0.547684998656997
i_gpt_low_cer_128 = 0.14806267616141244
i_gpt_low_wer_128 = 0.5596832482993197
i_gpt_low_cer_256 = 0.2307794744214584
i_gpt_low_wer_256 = 0.5659737240484429
i_gpt_low_cer_512 = 0.5165976089198959
i_gpt_low_wer_512 = 0.76617431640625


i_gpt_2_cer_64 = 0.04296288547178204
i_gpt_2_wer_64 = 0.22086148648648649
i_gpt_2_cer_128 = 0.045250095862034496
i_gpt_2_wer_128 = 0.32338362068965515
i_gpt_2_cer_256 = 0.0414382780168912
i_gpt_2_wer_256 = 0.2638767482517482
i_gpt_2_cer_512 = 0.07040806429620518
i_gpt_2_wer_512 = 0.3282199435763889


i_gemini_cer_64 = 0.5088746997424907
i_gemini_wer_64 = 0.6872859589041096
i_gemini_cer_128 = 0.3912749006401423
i_gemini_wer_128 = 0.6323309075342466
i_gemini_cer_256 = 0.497097177528128
i_gemini_wer_256 = 0.7501048657718121
i_gemini_cer_512 = 0.6239625325262782
i_gemini_wer_512 = 0.8536729236577181


i_qwen_cer_64 = 0.25287330774481803
i_qwen_wer_64 = 0.4628125
i_qwen_cer_128 = 0.2880357375551282
i_qwen_wer_128 = 0.5134079391891891
i_qwen_cer_256 = 0.47031159599395966
i_qwen_wer_256 = 0.6994270833333334
i_qwen_cer_512 = 0.6912767147192572
i_qwen_wer_512 = 0.8918098958333334


i_flux_cer_64 = 2.868082497178741
i_flux_wer_64 = 2.780625
i_flux_cer_128 = 1.0339202172552118
i_flux_wer_128 = 1.1443111795774648
i_flux_cer_256 = 0.688489178490883
i_flux_wer_256 = 0.881130246350365
i_flux_cer_512 = 0.6580758584739783
i_flux_wer_512 = 0.8581468986742424



i_gpt_medium_cer_64 = 0.04144199793241731
i_gpt_medium_wer_64 = 0.28996598639455784
i_gpt_medium_cer_128 = 0.04598680285570557
i_gpt_medium_wer_128 = 0.30872844827586204
i_gpt_medium_cer_256 = 0.06252688715672652
i_gpt_medium_wer_256 = 0.3213900862068966
i_gpt_medium_cer_512 = 0.21564198263477832
i_gpt_medium_wer_512 = 0.4727937940140845


lengths = [64, 128, 256, 512]

cer = {
    "GPT-L": [i_gpt_low_cer_64, i_gpt_low_cer_128, i_gpt_low_cer_256, i_gpt_low_cer_512],
    "Gemini": [i_gemini_cer_64, i_gemini_cer_128, i_gemini_cer_256, i_gemini_cer_512],
    "Qwen-Img": [i_qwen_cer_64, i_qwen_cer_128, i_qwen_cer_256, i_qwen_cer_512],
    "Flux.2": [i_flux_cer_64, i_flux_cer_128, i_flux_cer_256, i_flux_cer_512],
    "GPT-M": [i_gpt_medium_cer_64, i_gpt_medium_cer_128, i_gpt_medium_cer_256, i_gpt_medium_cer_512],
    "GPT-Image-2": [i_gpt_2_cer_64, i_gpt_2_cer_128, i_gpt_2_cer_256, i_gpt_2_cer_512],
}

wer = {
    "GPT-L": [i_gpt_low_wer_64, i_gpt_low_wer_128, i_gpt_low_wer_256, i_gpt_low_wer_512],
    "Gemini": [i_gemini_wer_64, i_gemini_wer_128, i_gemini_wer_256, i_gemini_wer_512],
    "Qwen-Img": [i_qwen_wer_64, i_qwen_wer_128, i_qwen_wer_256, i_qwen_wer_512],
    "Flux.2": [i_flux_wer_64, i_flux_wer_128, i_flux_wer_256, i_flux_wer_512],
    "GPT-M": [i_gpt_medium_wer_64, i_gpt_medium_wer_128, i_gpt_medium_wer_256, i_gpt_medium_wer_512],
    "GPT-Image-2": [i_gpt_2_wer_64, i_gpt_2_wer_128, i_gpt_2_wer_256, i_gpt_2_wer_512],
}

markers = {
    "GPT-L": "o",
    "Gemini": "s",
    "Qwen-Img": "^",
    "Flux.2": "D",
    "GPT-M": "X",
    "GPT-Image-2": "P",
}


from matplotlib.ticker import ScalarFormatter, NullLocator


gs_right = gs[0, 0].subgridspec(2, 1, hspace=0.2)
ax3a = fig.add_subplot(gs_right[0, 0])
ax3b = fig.add_subplot(gs_right[1, 0], sharex=ax3a)

for name, vals in cer.items():
    ax3a.plot(lengths, vals, marker=markers[name],
              markersize=10, markerfacecolor="white",
              markeredgewidth=2, linewidth=2.5, label=name)

ax3a.set_ylabel("CER")
ax3a.set_ylim(0, 1.5)
ax3a.grid(True, linestyle="--", alpha=0.4)
ax3a.set_title("(a) Text Rendering")
ax3a.set_xscale("log")
ax3a.set_xticks([64, 128, 256, 512])
ax3a.get_xaxis().set_major_formatter(ScalarFormatter())
ax3a.xaxis.set_minor_locator(NullLocator())  # remove minor ticks
ax3a.legend(ncol=2, fontsize=9)

for name, vals in wer.items():
    ax3b.plot(lengths, vals, marker=markers[name],
              markersize=10, markerfacecolor="white",
              markeredgewidth=2, linewidth=2.5)

ax3b.set_ylabel("WER")
ax3b.set_xlabel("Input Length (Words)")
ax3b.set_ylim(0, 1.5)
ax3b.set_xscale("log")
ax3b.set_xticks([64, 128, 256, 512])
ax3b.get_xaxis().set_major_formatter(ScalarFormatter())
ax3b.xaxis.set_minor_locator(NullLocator())  # remove minor ticks
ax3b.grid(True, linestyle="--", alpha=0.4)



plt.tight_layout()

plt.savefig("results_overview.pdf", bbox_inches="tight")
# plt.show()
